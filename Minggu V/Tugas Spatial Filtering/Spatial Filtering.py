import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

from skimage import data
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity

# ==============================
# 1. LOAD ORIGINAL IMAGE
# ==============================

image = data.camera()

# ==============================
# 2. ADD NOISE
# ==============================

gaussian_noise = random_noise(image, mode='gaussian', var=0.01)
sp_noise = random_noise(image, mode='s&p', amount=0.05)
speckle_noise = random_noise(image, mode='speckle', var=0.05)

# convert ke uint8
gaussian_img = (gaussian_noise*255).astype(np.uint8)
sp_img = (sp_noise*255).astype(np.uint8)
speckle_img = (speckle_noise*255).astype(np.uint8)

# ==============================
# 3. VISUALISASI NOISE
# ==============================

fig, ax = plt.subplots(1,4, figsize=(16,4))

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original")

ax[1].imshow(gaussian_img, cmap='gray')
ax[1].set_title("Gaussian Noise")

ax[2].imshow(sp_img, cmap='gray')
ax[2].set_title("Salt & Pepper")

ax[3].imshow(speckle_img, cmap='gray')
ax[3].set_title("Speckle Noise")

for a in ax:
    a.axis("off")

plt.show()

# ==============================
# 4. FUNGSI EVALUASI
# ==============================

def evaluate(original, filtered):
    
    psnr = peak_signal_noise_ratio(original, filtered)
    mse = mean_squared_error(original, filtered)
    ssim = structural_similarity(original, filtered)
    
    return psnr, mse, ssim

# ==============================
# 5. LIST HASIL
# ==============================

results = []

# ==============================
# 6. MEAN FILTER
# ==============================

for k in [3,5]:

    start = time.time()
    filtered = cv2.blur(gaussian_img,(k,k))
    end = time.time()

    psnr, mse, ssim = evaluate(image, filtered)

    results.append({
        "Filter":"Mean",
        "Noise":"Gaussian",
        "Kernel":f"{k}x{k}",
        "PSNR":psnr,
        "MSE":mse,
        "SSIM":ssim,
        "Time":end-start
    })

# ==============================
# 7. GAUSSIAN FILTER
# ==============================

for sigma in [1,2]:

    start = time.time()
    filtered = cv2.GaussianBlur(gaussian_img,(5,5),sigma)
    end = time.time()

    psnr, mse, ssim = evaluate(image, filtered)

    results.append({
        "Filter":"Gaussian",
        "Noise":"Gaussian",
        "Kernel":f"sigma={sigma}",
        "PSNR":psnr,
        "MSE":mse,
        "SSIM":ssim,
        "Time":end-start
    })

# ==============================
# 8. MEDIAN FILTER
# ==============================

for k in [3,5]:

    start = time.time()
    filtered = cv2.medianBlur(sp_img,k)
    end = time.time()

    psnr, mse, ssim = evaluate(image, filtered)

    results.append({
        "Filter":"Median",
        "Noise":"SaltPepper",
        "Kernel":f"{k}x{k}",
        "PSNR":psnr,
        "MSE":mse,
        "SSIM":ssim,
        "Time":end-start
    })

# ==============================
# 9. MIN FILTER
# ==============================

kernel = np.ones((3,3),np.uint8)

start = time.time()
min_filter = cv2.erode(sp_img, kernel)
end = time.time()

psnr, mse, ssim = evaluate(image, min_filter)

results.append({
    "Filter":"Min",
    "Noise":"SaltPepper",
    "Kernel":"3x3",
    "PSNR":psnr,
    "MSE":mse,
    "SSIM":ssim,
    "Time":end-start
})

# ==============================
# 10. MAX FILTER
# ==============================

start = time.time()
max_filter = cv2.dilate(sp_img, kernel)
end = time.time()

psnr, mse, ssim = evaluate(image, max_filter)

results.append({
    "Filter":"Max",
    "Noise":"SaltPepper",
    "Kernel":"3x3",
    "PSNR":psnr,
    "MSE":mse,
    "SSIM":ssim,
    "Time":end-start
})

# ==============================
# 11. TABEL HASIL
# ==============================

df = pd.DataFrame(results)

print("\nHasil Evaluasi Spatial Filtering:\n")
print(df)

# ==============================
# 12. VISUAL INSPECTION (LENGKAP)
# ==============================

plt.figure(figsize=(14,10))

# ===== ROW 1 : Gaussian Noise =====
plt.subplot(3,4,1)
plt.imshow(image, cmap='gray')
plt.title("Original")

plt.subplot(3,4,2)
plt.imshow(gaussian_img, cmap='gray')
plt.title("Gaussian Noise")

plt.subplot(3,4,3)
plt.imshow(cv2.blur(gaussian_img,(5,5)), cmap='gray')
plt.title("Mean Filter 5x5")

plt.subplot(3,4,4)
plt.imshow(cv2.GaussianBlur(gaussian_img,(5,5),1), cmap='gray')
plt.title("Gaussian Filter")


# ===== ROW 2 : Salt Pepper Noise =====
plt.subplot(3,4,5)
plt.imshow(image, cmap='gray')
plt.title("Original")

plt.subplot(3,4,6)
plt.imshow(sp_img, cmap='gray')
plt.title("Salt & Pepper Noise")

plt.subplot(3,4,7)
plt.imshow(cv2.medianBlur(sp_img,3), cmap='gray')
plt.title("Median 3x3")

plt.subplot(3,4,8)
plt.imshow(cv2.medianBlur(sp_img,5), cmap='gray')
plt.title("Median 5x5")


# ===== ROW 3 : Speckle Noise =====
plt.subplot(3,4,9)
plt.imshow(image, cmap='gray')
plt.title("Original")

plt.subplot(3,4,10)
plt.imshow(speckle_img, cmap='gray')
plt.title("Speckle Noise")

plt.subplot(3,4,11)
plt.imshow(cv2.GaussianBlur(speckle_img,(5,5),1), cmap='gray')
plt.title("Gaussian Filter")

plt.subplot(3,4,12)
plt.imshow(cv2.blur(speckle_img,(3,3)), cmap='gray')
plt.title("Mean Filter 3x3")


for i in range(1,13):
    plt.subplot(3,4,i).axis("off")

plt.suptitle("Visual Inspection Spatial Filtering", fontsize=16)

plt.tight_layout()
plt.show()

# ==============================
# 13. VISUAL INSPECTION MIN / MAX FILTER
# ==============================

plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.imshow(image, cmap='gray')
plt.title("Original")

plt.subplot(1,4,2)
plt.imshow(sp_img, cmap='gray')
plt.title("Salt & Pepper Noise")

plt.subplot(1,4,3)
plt.imshow(cv2.erode(sp_img, np.ones((3,3),np.uint8)), cmap='gray')
plt.title("Min Filter")

plt.subplot(1,4,4)
plt.imshow(cv2.dilate(sp_img, np.ones((3,3),np.uint8)), cmap='gray')
plt.title("Max Filter")

for i in range(1,5):
    plt.subplot(1,4,i).axis("off")

plt.suptitle("Visual Inspection Min / Max Filter")

plt.show()