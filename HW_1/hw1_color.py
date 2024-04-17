import cv2, skimage.data
import numpy as np
import matplotlib.pyplot as plt

# Load Image
img = skimage.data.astronaut()

# 1) RGB, Python
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]
cv2.imwrite('./fig/RGB_astronaut.png', cv2.hconcat([R, G, B]))

# 2) YCbCr, Python
YCbCr = np.array([[ 77,  150,  29],
                  [-43,  -84, 127],
                  [127, -106, -21]]) / 256.
img_YCbCr = img.dot(YCbCr.T)
Y, Cb, Cr = np.split(img_YCbCr, 3, axis=2)
Cb = np.clip(Cb+128, 0, 255)
Cr = np.clip(Cr+128, 0, 255)
#Y = np.clip(Y, 0, 255).astype(np.uint8)
#Cb = np.clip(Cb, 0, 255).astype(np.uint8)
#Cr = np.clip(Cr, 0, 255).astype(np.uint8)
cv2.imwrite('./fig/YCbCr_astronaut.png', cv2.hconcat([Y, Cb, Cr])) 

# 3) HSI, Python
eps = 1e-8
R, G, B = R/255, G/255, B/255
I = (R + G + B) / 3
S = 1 - np.minimum(np.minimum(R, G), B) / (I+eps)
H = 1 / (2*np.pi) * np.arccos((2*R-G-B)/(2*np.sqrt((R-G)**2+(R-B)*(G-B)+eps)))
H[B>G] = 1 - H[B>G]
H[I==0] = 0
cv2.imwrite('./fig/HSI_astronaut.png', cv2.hconcat([H*255, S*255, I*255]))

# 4) Modifying image
S += 40/255
S[S>=1] = 1
I += 16/255
I[I>=1] = 1
hsi = np.zeros(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if H[i, j]>=0 and H[i, j]<1/3:
            b = I[i, j] * (1 - S[i, j])
            r = I[i, j] * (1 + S[i, j]*np.cos(H[i, j]*2*np.pi)/np.cos(np.pi/3 - H[i, j]*2*np.pi))
            g = 3*I[i, j] - (r + b)
        elif H[i, j]>=1/3 and H[i, j]<2/3:
            r = I[i, j] * (1 - S[i, j])
            g = I[i, j] * (1 + S[i, j]*np.cos(H[i, j]*2*np.pi-2*np.pi/3)/np.cos(np.pi - H[i, j]*2*np.pi))
            b = 3*I[i, j] - (r + g)
        else:
            g = I[i, j] * (1 - S[i, j])
            b = I[i, j] * (1 + S[i, j]*np.cos(H[i, j]*2*np.pi-4*np.pi/3)/np.cos(5*np.pi/3-H[i, j]*2*np.pi))
            r = 3*I[i, j] - (g + b)
        hsi[i, j, 0], hsi[i, j, 1], hsi[i, j, 2] = r, g, b
hsi *= 255
cv2.imwrite('./fig/HSI2RGB_astronaut.png', hsi[:, :, ::-1])