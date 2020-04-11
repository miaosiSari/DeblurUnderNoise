import torch
import torch.nn as nn
import math
import torch.nn.init as init
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision import utils as utils
from copy import deepcopy
import traceback
import sys

class _ResBLock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBLock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, stride, 1, bias=True)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = torch.add(residual, out)
        return out
    
    
class deblur(nn.Module):
    def __init__(self, numres, concatenate=False, rgb=3):
        super(deblur, self).__init__()
        if concatenate:
            self.conv1     = nn.Conv2d(2*rgb, 64, (9, 9), 1, padding=4)
        else:
            self.conv1     = nn.Conv2d(rgb, 64, (9, 9), 1, padding=4)
        self.relu      = nn.LeakyReLU(0.2, inplace=True)
        self.numres = numres
        self.resBlock1 = self._makelayers(64, 64, self.numres)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), 2, 1),
            nn.ReLU(inplace=True)
        )
        self.resBlock2 = self._makelayers(128, 128, self.numres)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), 2, 1),
            nn.ReLU(inplace=True)
        )
        self.resBlock3 = self._makelayers(256, 256, self.numres)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (4, 4), 2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4, 4), 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (7, 7), 1, padding=3)
        )
        self.convout = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, rgb, (3, 3), 1, 1)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLock(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        con1   = self.relu(self.conv1(x))
        res1   = self.resBlock1(con1)
        res1   = torch.add(res1, con1)
        con2   = self.conv2(res1)
        res2   = self.resBlock2(con2)
        res2   = torch.add(res2, con2)
        con3   = self.conv3(res2)
        res3   = self.resBlock3(con3)
        res3   = torch.add(res3, con3)
        decon1 = self.deconv1(res3)
        EC_feature = self.deconv2(decon1)
        EC_out = self.convout(torch.add(EC_feature, con1))
        return EC_out
    
    
class multiscale(nn.Module):
    def __init__(self, numres, rgb=1):
        super(multiscale, self).__init__()
        self.denoise = deblur(numres,rgb=rgb)
        self.A = deblur(numres,rgb=rgb)
        self.B = deblur(numres,concatenate=True,rgb=rgb)
        self.smoothing = deblur(0)
        self.rgb = rgb
        self.upconv = nn.Sequential(
            nn.Conv2d(rgb, 256, (3, 3), 1, 1),
            nn.ReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(64, rgb, (3, 3), 1, 1)
         )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()
    
    def forward(self, x, t='trainblur', smooth=False):
        if t != 'trainblur':
            x = self.denoise(x)
        if t != 'trainnoise':
            down = nn.functional.interpolate(x,scale_factor=0.5)
            resultdown = self.A(down)
            up = self.upconv(resultdown)
            uptensor = torch.cat([up, x], 1)
            resultup = self.B(uptensor)
            smoothtensor = self.smoothing(resultup) if (smooth and self.rgb == 1) else resultup
            '''
            if t == 'trainblur':
                return resultdown, resultup, x
            else:
                return resultdown, resultup, x
            '''
            return resultdown, resultup, x
        else:
            return 'trainnoise', x, True

        
if __name__ == '__main__':
    class util:
        @staticmethod
        def nearest(integer, M=8):
            mod = integer % M
            if integer < (M//2):
                return integer - mod
            else:
                return integer + M - mod
            
        @staticmethod
        def readimage(path, size=[1], cuda=True, togray=True):
            tensorlist = []
            img = Image.open(path)
            H, W = img.size
            d = dict()
            if togray:
                img = img.convert('L')
            for element in size:
                im = deepcopy(img)
                if isinstance(element, tuple):
                    nearest1 = util.nearest(element[0])
                    nearest2 = util.nearest(element[1])
                    if nearest1 != H or nearest2 != W:
                        im = im.resize((nearest1, nearest2))
                    original = element
                else:
                    size1 = int(H*element)
                    size2 = int(W*element)
                    nearest1 = util.nearest(size1)
                    nearest2 = util.nearest(size2)
                    im = im.resize((nearest1, nearest2))
                    original = (size1, size2)
                im = np.array(im).astype(np.float32)/255.0
                print(im.shape)
                if len(im.shape) == 3:
                    im = np.expand_dims(im.transpose((2, 0, 1)),0)
                else:
                    im = np.expand_dims(np.expand_dims(im, 0), 0)
                tensor = torch.from_numpy(im).float()
                if cuda:
                    tensor = tensor.cuda()
                d[element] = {'tensor':tensor, 'original':original}
            return d
        
        @staticmethod
        def run(net, d, cmd='trainjoint', modified = False):
            for key in d.keys():
                with torch.no_grad():
                    if modified:
                        waste, im, imb, ca, sa = net(d[key]['tensor'], cmd, False)
                    else:
                        waste, im, imb = net(d[key]['tensor'], cmd, False)
                im = util.toimage(im, save=False)
                im = im.resize(d[key]['original'])
                imb = util.toimage(imb, save=False)
                imb = imb.resize(d[key]['original'])
                d[key]['image'] = im
                d[key]['imblurry'] = imb
            return d
        
        
        @staticmethod
        def runall(net, path, size, togray=False, toimage=False, savepath='', cmd='trainjoint'):
            print(path)
            d = util.readimage(path, size, togray=togray)
            d = util.run(net, d, cmd=cmd)
            if toimage:
                for key in d.keys():
                    d[key]['image'].save(savepath[key]['image'])
                    d[key]['imblurry']. save(savepath[key]['imblurry'])
            return d
            
            
        @staticmethod
        def runall_multi(net, pathlist, size, togray=False, cmd='trainjoint'):
            d = dict()
            for path in pathlist:
                print(path)
                d[path] = util.runall(net, path, size, togray, cmd=cmd)
            return d
                
        @staticmethod
        def toimage(tensor,save=True, path=''):
            (filepath,tempfilename) = os.path.split(path)
            t = torch.clamp(tensor, min=0.0, max=1.0)
            t = transforms.ToPILImage()(t.cpu()[0])
            if save and path !='':
                if not os.path.isdir(filepath):
                    os.mkdir(filepath)
                t.save(path)
            return t
        
            
    try:
        gpu = int(sys.argv[1])
    except:
        gpu = 0
        pass
    gpus = torch.cuda.device_count()
    assert gpu < gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("gpu id:%s"%gpu)
    torch.cuda.set_device(gpu)
    dire = 'result'
    if not os.path.isdir(dire):
        os.mkdir(dire) 
    net = multiscale(6, rgb=3).cuda()
    net.load_state_dict(torch.load('nrgb_multiscale_final.pth', map_location={'cuda:2':'cuda:%s'%gpu}))
    sizes = [0.25, 0.50, 1]
    pathlist = []
    for cnt1 in range(1, 5):
        for cnt2 in range(1, 13):
            for sigma in [10, 25]:
                pathlist.append('BlurryImages/%sBlurry%s_%s.png'%(sigma, cnt1, cnt2))
    d = util.runall_multi(net, pathlist, sizes)
    for path in d.keys():
        for key in d[path].keys():
            d[path][key]['image'].save('resultjoint/%.02fl_%s'%(key, path.replace('BlurryImages/', '')))
            d[path][key]['imblurry'].save('resultjoint/%.02fb_%s'%(key, path.replace('BlurryImages/', '')))
    
 
   
