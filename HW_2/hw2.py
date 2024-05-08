# 2022094093 Kim Dohoon
# Computer Vision Assignmnet2
import numpy as np
import matplotlib.pyplot as plt
import cv2

####### Load origianl image
img = cv2.imread('./fig/cat.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')

####### 1. Mean Filtering
def Conv(img, filter_, stride=1):
    conv = np.zeros(img.shape)
    H, W, C = conv.shape
    pad_size = int((stride*(img.shape[0]-1)-img.shape[0]+filter_.shape[0])/2)
    img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)))
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            for c in range(C):
                conv[h, w, c] = np.sum(filter_ * img[h:h+filter_.shape[0], w:w+filter_.shape[1], c])
    return conv

def GaussianBlur(n, sig=1):
    matrix = np.zeros((n, n))
    r = n // 2
    s = 0
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            G = (1/(2*np.pi*sig**2)) * np.exp(-(i**2+j**2)/(2*sig**2))
            matrix[i+r, j+r] = G
            s += G
    # Normalize to make sum=1
    for i in range(n):
        for j in range(n):
            matrix[i][j] = matrix[i][j] / s
    return matrix

def PSNR(before, after):
    mse = np.mean((before-after)**2)
    psnr = 20 * np.log10(255.0/np.sqrt(mse))
    return psnr

plt.figure(dpi=100, figsize=(24, 8))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    LPF = GaussianBlur(2*i+1, 5)
    LFI = Conv(img, LPF).astype(np.uint8)
    plt.imshow(LFI)
    plt.axis('off')
    plt.title(f"kernel size = {2*i+1}")
    print(PSNR(img, LFI))
#plt.savefig('./fig/mean_filtering.png', bbox_inches='tight')

####### 2. Unsharp Mask
plt.figure(dpi=100, figsize=(24, 8))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    LPF = GaussianBlur(2*i+1, 5)
    LFI = Conv(img, LPF)
    HFD = img + (-1)*LFI
    HFI = img + 5*HFD
    HFI = np.clip(HFI, 0, 255).astype(np.uint8)
    plt.imshow(HFI)
    plt.axis('off')
    plt.title(f"kernel size = {2*i+1}")
#plt.savefig('./fig/unsharp_mask.png', bbox_inches='tight')

####### 3. Contrast Stretching
# 3-1. contrast stretch
def ContrastStretch(image, func, *args):
    img = image.copy()
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            for c in range(image.shape[2]):
                img[h, w, c] = func(image[h, w, c], *args)
    return img

def contrast(u, a, va, b, vb):
    if 0<=u and u<a:
        return (va/a)*u
    elif a<=u and u<b:
        return ((vb-va)/(b-a))*(u-a) + va
    else:
        return ((255-vb)/(255-b))*(u-b) + vb

cs = ContrastStretch(img, contrast, 64, 16, 192, 240)
plt.imshow(cs)
plt.axis('off')
#plt.savefig('./fig/contrast_stretch.png', bbox_inches='tight')

# 3-2. gamma correction
def GammaCorr(u, c, gamma):
    return c * u**gamma
gc = img / 255.0
gc = ContrastStretch(gc, GammaCorr, 1, .3)
plt.imshow(gc)
plt.axis('off')
plt.savefig('./fig/gamma_corr.png', bbox_inches='tight')

####### 4. Histogram Equalization
def HistogramEqualization(im):
    image = im.copy()
    cdf_r = np.array([np.count_nonzero(image[:, :, 0]==i) for i in range(256)]).cumsum()
    cdf_g = np.array([np.count_nonzero(image[:, :, 1]==i) for i in range(256)]).cumsum()
    cdf_b = np.array([np.count_nonzero(image[:, :, 2]==i) for i in range(256)]).cumsum()
    cnt = image.shape[0] * image.shape[1]
    table_r = {i : round(255*cdf_r[i]/cnt) for i in range(256)}
    table_g = {i : round(255*cdf_g[i]/cnt) for i in range(256)}
    table_b = {i : round(255*cdf_b[i]/cnt) for i in range(256)}
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            image[h, w, 0] = table_r[im[h, w, 0]]
            image[h, w, 1] = table_g[im[h, w, 1]]
            image[h, w, 2] = table_b[im[h, w, 2]]
    return image

plt.figure(dpi=100)
plt.figure(figsize=(16, 8))
plt.axis('off')
plt.subplot(2, 4, 1)
plt.imshow(img)
plt.axis('off')
plt.rcParams['font.size'] = 12
plt.title('Image')
color = ['R', 'G', 'B']
plt.rcParams['font.size'] = 8
for n in range(2, 5):
    plt.subplot(2, 4, n)
    cnt = np.array([np.count_nonzero(img[:, :, n-2]==i) for i in range(256)])
    plt.bar(range(256), cnt)
    plt.title(color[n-2])

plt.subplot(2, 4, 5)
equalized = HistogramEqualization(img)
plt.imshow(equalized)
plt.axis('off')
for n in range(6, 9):
    plt.subplot(2, 4, n)
    cnt = np.array([np.count_nonzero(equalized[:, :, n-6]==i) for i in range(256)])
    plt.bar(range(256), cnt)
#plt.savefig('./fig/hist_equalization.png', bbox_inches='tight')

####### 5. Image Upsampling
# downsample
H = int(img.shape[0]/4)
W = int(img.shape[1]/4)
downsampled = np.zeros((H, W, 3), dtype=np.uint8)
for h in range(H):
    for w in range(W):
        downsampled[h, w] = img[h*(img.shape[0] // H), w*(img.shape[1] // W)]
plt.imshow(downsampled)
plt.axis('off')
plt.savefig('./fig/downsampled.png', bbox_inches='tight')
plt.figure(figsize=(24, 8), dpi=100)

plt.subplot(1,3,1)
upscaling_nn = cv2.resize(downsampled, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
plt.imshow(upscaling_nn)
plt.axis('off')
plt.title(f'Nearest Neighbor, PSNR={round(PSNR(img, upscaling_nn), 4)}')
print(PSNR(img, upscaling_nn))

plt.subplot(1,3,2)
upscaling_bilinear = cv2.resize(downsampled, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
plt.imshow(upscaling_bilinear)
plt.axis('off')
plt.title(f'Bilinear, PSNR={round(PSNR(img, upscaling_bilinear), 4)}')
print(PSNR(img, upscaling_bilinear))

plt.subplot(1,3,3)
upscaling_bicubic = cv2.resize(downsampled, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
plt.imshow(upscaling_bicubic)
plt.axis('off')
plt.title(f'Bicubic, PSNR={round(PSNR(img, upscaling_bicubic), 4)}')
print(PSNR(img, upscaling_bicubic))
plt.savefig('./fig/interpolation.png', bbox_inches='tight')