function y = rgb2gray2(x)

if size(x, 3) == 3
    y = rgb2gray(x);
else
    y = x;
end   