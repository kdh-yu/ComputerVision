import cv2, skimage.data
import numpy as np
import matplotlib.pyplot as plt

img = skimage.data.astronaut()

# 1) Scaling, Python
s = 1.5
S = np.array([[s, 0, 0],
              [0, s, 0],
              [0, 0, 1]])
scaled = cv2.warpPerspective(img, S, (0, 0))
cv2.imwrite('./fig/scaling_astronaut.png', scaled[:, :, ::-1])

# 2) Rotation, Python
theta = np.pi / 6
R = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta),  np.cos(theta), 0],
              [0,              0,             1]])
rotated = cv2.warpPerspective(img, R, (0, 0))
cv2.imwrite('./fig/rotation_astronaut.png', rotated[:, :, ::-1])

# 3) Similarity, Python
s = 3
theta = -np.pi / 3
Sim = np.array([[s*np.cos(theta), -np.sin(theta),   0],
                [np.sin(theta),   s*np.cos(theta),  0],
                [0,                0,               1]])
similarity = cv2.warpPerspective(img, Sim, (0, 0))
cv2.imwrite('./fig/similarity_astronaut.png', similarity[:, :, ::-1])

# 4) Affine, Python
Aff = np.array([[  2, 0.33, -100], 
                [0.2,    1,   50], 
                [  0,    0,    1]]);
affine = cv2.warpPerspective(img, Aff, (0, 0))
cv2.imwrite('./fig/affine_astronaut.png', affine[:, :, ::-1])