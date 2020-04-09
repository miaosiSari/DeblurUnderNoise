clc;
clear;
addpath(genpath('cho_code'));
for cnt1 = 1:4
    for cnt2 = 1:12
        if cnt2 <= 7 && cnt2 ~= 5
            initk = [35, 35];
            prod = 2;
        elseif cnt2 == 12 || cnt2 == 5
            initk = [55, 55];
            prod = 1.3;
        else
            initk = [95, 95];
            prod = 1.3;
        end
        %s, sigma, grad, wei_exem
        file = sprintf('Blurry%d_%d', cnt1, cnt2);
        fprintf("file name %s\n", file);
        fid = fopen('log.txt', 'a');
        fprintf(fid, '%s starts!\n', file);
        fclose(fid);
        %deblurL0(0.5, 10, 0, 1, file, initk, prod);%coarsetofine + wograd + exem
        deblurL0(0.5, 10, 0, 1, file, initk, prod);
        deblurL0(0.5, 25, 0, 1, file, initk, prod);
        %deblurL0(0.5, 10, 1, 1, file, initk, prod);
        %deblurL0(0.5, 25, 1, 1, file, initk, prod);
        %deblurL0(0.5, 10, 0, 0, file, initk, prod);
        %deblurL0(0.5, 25, 0, 0, file, initk, prod);
        
        %deblurL0(1.0, 10, 0, 1, file, initk, prod);
        %deblurL0(1.0, 25, 0, 1, file, initk, prod);
        fid = fopen('log.txt', 'a');
        fprintf(fid, '%s finishes!\n', file);
        fclose(fid);
    end
end