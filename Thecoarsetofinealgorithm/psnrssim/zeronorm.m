function l0 = zeronorm(x, eps)

if ~exist('eps', 'var')
    eps = 1e-4; 
end
b = (x > eps);
l0 = sum(b(:));




