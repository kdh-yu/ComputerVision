clear all;

%img = imread('peppers.png');
img = imread('./fig/cb.png');

% 1) RGB, Matlab
R = img(:, :, 1);
G = img(:, :, 2);
B = img(:, :, 3);
%imwrite([R, G, B], './fig/RGB_peppers.png');
imwrite([R, G, B], './fig/RGB_cb.png');

% 2) YCbCr, Matlab
YCbCr = [ 77  150   29; 
         -43  -84  127; 
         127 -106  -21;] / 256.;
img_YCbCr = reshape(double(img)/255, [], 3) * YCbCr';
img_YCbCr = reshape(img_YCbCr, size(img));
Y = img_YCbCr(:, :, 1);
Cb = min(img_YCbCr(:, :, 2) + 0.5, 1);
Cr = min(img_YCbCr(:, :, 3) + 0.5, 1);
%imwrite([Y, Cb, Cr], './fig/YCbCr_peppers.png');
imwrite([Y, Cb, Cr], './fig/YCbCr_cb.png');

% 3) HSI, Matlab
R = double(R) / 255;
G = double(G) / 255;
B = double(B) / 255;
I = (R + G + B) / 3;
S = 1 - min(min(R, G), B) ./ (I+eps);
H = 1 / (2*pi) * acos((2*R-G-B)./(2*sqrt((R-G).^2+(R-B).*(G-B))+eps));
H(B>G) = 1 - H(B>G);
H(I==0) = 0;
%imwrite([H, S, I], './fig/HSI_peppers.png');
imwrite([H, S, I], './fig/HSI_cb.png');

% 4) Modifying Image
S = S+40/255;
S(S>=1) = 1;
I = I+16/255;
I(I>=1) = 1;
hsi = zeros(size(R, 1), size(R, 2), 3);
for i = 1:size(R, 1)
    for j = 1:size(R, 2)
        if H(i, j)>=0 && H(i, j)<1/3
            b = I(i, j) * (1 - S(i, j));
            r = I(i, j) * (1 + S(i, j)*cos(H(i, j)*2*pi)/cos(pi/3 - H(i, j)*2*pi));
            g = 3*I(i, j) - (r + b);
        elseif H(i, j)>=1/3 && H(i, j)<2/3
            r = I(i, j) * (1 - S(i, j));
            g = I(i, j) * (1 + S(i, j)*cos(H(i, j)*2*pi - 2*pi/3)/cos(pi - H(i, j)*2*pi));
            b = 3*I(i, j) - (r + g);
        else
            g = I(i, j) * (1 - S(i, j));
            b = I(i, j) * (1 + S(i, j)*cos(H(i, j)*2*pi - 4*pi/3)/cos(5*pi/3 - H(i, j)*2*pi));
            r = 3*I(i, j) - (g + b);
        end
        hsi(i, j, 1) = r;
        hsi(i, j, 2) = g;
        hsi(i, j, 3) = b;
    end
end
%imwrite(hsi, './fig/HSI2RGB_peppers.png');
imwrite(hsi, './fig/HSI2RGB_cb.png');