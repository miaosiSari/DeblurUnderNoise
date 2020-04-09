function t = deblurL0(s, sigma, grad, wei_exem, orig,  initk, prod)

fprintf("deblurL0 starts! s = %f, sigma = %d, grad = %d, wei_exem = %f, orig = %s, prod = %f!\n\n", s, sigma, grad, wei_exem, orig, prod);
disp(initk);
dir = 'result';
inits = s;

dx = [-1 1; 0 0];
dy = [-1 0; 1 0];
fx = [1, -1];
fy = [1; -1];
count = 0;
wei_grad = 4e-3;
betamax = 1e2;
init_wei_grad = wei_grad;
init_wei_exem = wei_exem;
decay = false;
synthesized = true;
usingconjgrad = true;
flag = true;
if s == 1.0
    flag = false;
end
if synthesized
     brgb1 = im2double(imread(sprintf('%s/1.00b_%d%s.png', dir, sigma, orig)));   
else
     brgb1 = im2double(imread(sprintf('%s.png', orig)));   
end
while s <= 1.0 || flag
   ksize = double(uint8(initk * s));
   for dim = 1:2
       if mod(ksize(dim), 2) == 0
           ksize(dim) = ksize(dim) + 1;
       end
   end  
   brgb = imresize(brgb1, s);
   b = rgb2gray2(brgb);
   [H, W] = size(b);
   if count == 0
      kernel = zeros(ksize);
      kernel(uint8((ksize(1)+1)/2), uint8((ksize(2)+1)/2)) = 1.0;
      if synthesized
           lrgb = im2double(imread(sprintf('%s/%0.2fl_%d%s.png', dir, s, sigma, orig)));
      else
           lrgb = im2double(imread(sprintf('%s.png', exem)));
           lrgb = imresize(lrgb, s);
      end
      l = rgb2gray2(lrgb);
   else
      kernel = imresize(kernel, ksize);
      kernel = refine_kernel(kernel, false, true);
      lrgb = imresize(t, [H, W]);
      if s == 1.0
          lrgb2 = im2double(imread(sprintf('%s/%0.2fl_%d%s.png', dir, s, sigma, orig)));
          F = abs(fft2(lrgb));
          F2 = abs(fft2(lrgb2));
          sumF = sum(F(:));
          sumF2 = sum(F2(:));
          fprintf('sharpness: CNN %f, upsampled %f!\n', sumF2, sumF);
          if sumF2 > sumF
               lrgb = lrgb2;
               fprintf('choose CNN as an exemplar!\n');
               kernel = zeros(ksize);
               kernel(uint8((ksize(1)+1)/2), uint8((ksize(2)+1)/2)) = 1.0;
               wei_exem = init_wei_exem;
               wei_grad = init_wei_grad;
          end
      end
      l = rgb2gray2(lrgb);
   end
   if grad
       %ls = L0Smoothing(l, 2e-2);
       ls = l;
       latentlsx = [diff(ls,1,2),ls(:,1,:) - ls(:,end,:)];
       latentlsy = [diff(ls,1,1); ls(1,:,:) - ls(end,:,:)];
       Normin = [latentlsx(:,end,:) - latentlsx(:, 1,:), -diff(latentlsx,1,2)];
       Normin = Normin + [latentlsy(end,:,:) - latentlsy(1, :,:); -diff(latentlsy,1,1)];
       fft2Normin = fft2(Normin);
   else 
       %ls = l;
       ls = L0Smoothing(l, 2e-2);
       fft2E = fft2(ls);
   end
   [~, ~, threshold]= threshold_pxpy_v1(b,max(ksize));
   
   blur_B_w = wrap_boundary_liu(b, opt_fft_size([H + ksize(1) - 1   W + ksize(2) - 1]));
   blur_B_w = blur_B_w(1:H, 1:W, :);
   Bx = conv2(blur_B_w, dx, 'valid');
   By = conv2(blur_B_w, dy, 'valid');
   
   iters = 5;

   for cnt = 1:iters %iteration
       S = b;
       otfFx = psf2otf(fx,[H, W]);
       otfFy = psf2otf(fy,[H, W]);
       Denormin2 = abs(otfFx).^2 + abs(otfFy).^2;
       fprintf('s = %f, cnt = %d, wei_exem=%f\n', s, cnt, wei_exem);
       KER = psf2otf(kernel,size(S));
       Den_KER = abs(KER).^2;
       Normin1 = conj(KER).*fft2(b);
       beta = 2 * wei_grad;
       while beta < betamax
           h = [diff(S,1,2), S(:,1,:) - S(:,end,:)];
           v = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
           t = (h.^2+v.^2) < wei_grad/beta;
           h(t)=0; v(t)=0;
           clear t;
           Normin2 = [h(:,end,:) - h(:, 1,:), -diff(h,1,2)];
           Normin2 = Normin2 + [v(end,:,:) - v(1, :,:); -diff(v,1,1)]; 
           if grad         
               normin = Normin1 + beta * fft2(Normin2) + wei_exem * fft2Normin;
               denormin = Den_KER + (beta + wei_exem) * Denormin2;
           else
               normin = Normin1 + beta * fft2(Normin2) + wei_exem * fft2E;
               denormin = Den_KER + beta * Denormin2 + wei_exem;
           end
           S = real(ifft2(normin./denormin));
           beta = 2 * beta;
       end
       [latent_x, latent_y, threshold] = threshold_pxpy_v1(S,max(ksize),threshold);
       if count == 0 || wei_exem == 0 || usingconjgrad
           kernel = estimate_psf(Bx, By, latent_x, latent_y, 2.0, ksize);
       else
           kernel = estimate_psf_conjgrad(count, 10.0, kernel, Bx, By, latent_x, latent_y, 2.0, false);
       end
       kernel = refine_kernel(kernel, true, true);
       subplot(1,3,1);
       imshow(kernel, []);
       subplot(1,3,2);
       imshow(S, []);
       drawnow;
       wei_grad = max(wei_grad/1.1, 1e-4);
       if wei_exem > 0 && decay
           wei_exem = max(wei_exem/1.1, 1e-3);
       end
   end
   kernel = refine_kernel(kernel, true, false);
   t = ringing_artifacts_removal(brgb, kernel, 0.003,5e-4, 1);
   subplot(1,3,3);
   imshow(t, []);
   drawnow;
   count = count + 1;
   s = s * prod;
   if s >= 1.0 && flag
       flag = false;
       s = 1.0;
   end
end
[~, orig, ~] = fileparts(orig);
if grad 
    if inits == 1.0
        fprintf('grad%d_%s_wocoarse.png\n', sigma, orig);
        imwrite(t, sprintf('grad%d_%s_wocoarse.png', sigma, orig));
    else
        if wei_exem == 0.0
            fprintf('grad%d_%s_woexem.png\n', sigma, orig);
            imwrite(t, sprintf('grad%d_%s_woexem.png', sigma, orig));
        else
            fprintf('grad%d_%s.png\n', sigma, orig);
            imwrite(t, sprintf('grad%d_%s.png', sigma, orig));
        end
    end
else
    if inits == 1.0
         fprintf('%d_%s_wocoarse.png', sigma, orig);
         imwrite(t, sprintf('%d_%s_wocoarse.png', sigma, orig));
    else
         if wei_exem == 0.0
             fprintf('%d_%s_woexem.png', sigma, orig);
             imwrite(t, sprintf('%d_%s_woexem.png', sigma, orig));
         else
             fprintf('%d_%s.png', sigma, orig);
             imwrite(t, sprintf('%d_%s_%f_%f.png', sigma, orig, wei_exem, betamax));
         end
    end
 end