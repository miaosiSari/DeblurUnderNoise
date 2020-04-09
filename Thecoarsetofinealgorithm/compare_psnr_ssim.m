clc;
clear;
addpath(genpath('psnrssim'));
[p, s] = calc_psnr_ssim('104_Zhong.jpg', '104.jpg');
[p1, s1] = calc_psnr_ssim('104_l_deblurred.jpg', '104.jpg');
[p2, s2] = calc_psnr_ssim('deblurred_.jpg', '104.jpg');
[p3, s3] = calc_psnr_ssim('104_SRN.jpg', '104.jpg');
[p4, s4] = calc_psnr_ssim('104_l_JointCNN.jpg', '104.jpg');