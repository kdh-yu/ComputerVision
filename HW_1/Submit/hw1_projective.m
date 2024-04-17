%{
2022094093 Kim Dohoon
Assignment#1
%}

% Load peppers.png
img = imread('peppers.png');

% Load Own Image - Cherry Blossom
%img = imread('./fig/cb.png');

% 1) Scaling, Matlab
s = 1.5;
S = [s 0 0; 
     0 s 0; 
     0 0 1];
scaled = imwarp(img, projective2d(S), 'OutputView', imref2d(size(img)));
imwrite(scaled, './fig/scaling_peppers.png', 'png');
%imwrite(scaled, './fig/scaling_cb.png', 'png')

% 2) Rotation, Matlab
theta = -pi / 6;
R = [cos(theta) -sin(theta) 0; 
     sin(theta) cos(theta)  0; 
     0          0           1];
rotated = imwarp(img, projective2d(R), 'OutputView', imref2d(size(img)));
imwrite(rotated, './fig/rotation_peppers.png', 'png');
%imwrite(rotated, './fig/rotation_cb.png', 'png')

% 3) Similarity, Matlab
s = 3;
theta = pi / 3;
Sim = [s*cos(theta) -sin(theta)  0; 
       sin(theta)   s*cos(theta) 0; 
       0            0            1];
similarity = imwarp(img, projective2d(Sim), 'OutputView', imref2d(size(img)));
imwrite(similarity, './fig/similarity_peppers.png', 'png');
%imwrite(similarity, './fig/similarity_cb.png', 'png');

% 4) Affine, Matlab
Aff = [   2 0.33 0; 
        0.2    1 0; 
       -100   50 1;];
affine = imwarp(img, projective2d(Aff), 'OutputView', imref2d(size(img)));
imwrite(affine, './fig/affine_peppers.png', 'png');
%imwrite(affine, './fig/affine_cb.png', 'png');
