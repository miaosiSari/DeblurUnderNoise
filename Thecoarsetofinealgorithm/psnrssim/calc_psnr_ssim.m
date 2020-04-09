function [p,s] = calc_psnr_ssim(img1, img2, ispath)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%  FUNCTION:  metrix_psnr
%%%
%%%  INPUTS:    reference_image     - original image data  path
%%%
%%%             query_image         - modified image data path to be compared with
%%%                                   original image
%%%
%%%  OUTPUTS:   psnr        - PSNR value
%%%
%%%  OUTPUTS:   ssim        - SSIM value
%%%
%%%  CHANGES:   NONE
%%%             
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist('ispath', 'var')
   ispath = true;
end

if ispath
   img1 = im2uint8(imread(img1));
   img2 = im2uint8(imread(img2));
else
   img1 = im2uint8(img1);
   img2 = im2uint8(img2);
end


p = metrix_psnr(img1, img2);
s = metrix_ssim(img1, img2);