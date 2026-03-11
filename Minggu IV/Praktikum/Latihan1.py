import cv2
import numpy as np
import matplotlib.pyplot as plt

def manual_histogram_equalization(image):
    """
    Manual implementation of histogram equalization
    
    Parameters:
    image: Input grayscale image (0-255)
    
    Returns:
    equalized_image: Image after histogram equalization
    transform: transformation function
    hist: original histogram
    """

    # 1. Hitung histogram
    hist = np.zeros(256)

    for pixel in image.flatten():
        hist[pixel] += 1

    # 2. Hitung cumulative histogram (CDF)
    cdf = np.zeros(256)
    cdf[0] = hist[0]

    for i in range(1,256):
        cdf[i] = cdf[i-1] + hist[i]

    # 3. Hitung transformation function
    cdf_min = np.min(cdf[cdf > 0])
    total_pixels = image.shape[0] * image.shape[1]

    transform = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255)
    transform = transform.astype('uint8')

    # 4. Apply transformation
    equalized_image = transform[image]

    # 5. Return equalized image and transformation function
    return equalized_image, transform, hist


# ====================================================
# MAIN PROGRAM
# ====================================================

# Load image
image = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise Exception("Image tidak ditemukan. Pastikan file image.jpg ada.")

# Jalankan histogram equalization manual
equalized_image, transform, hist_original = manual_histogram_equalization(image)

# Hitung histogram hasil
hist_equalized = np.zeros(256)
for pixel in equalized_image.flatten():
    hist_equalized[pixel] += 1

# ====================================================
# VISUALISASI
# ====================================================

plt.figure(figsize=(12,8))

# Original Image
plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Histogram Original
plt.subplot(2,2,2)
plt.title("Histogram Original")
plt.plot(hist_original)

# Equalized Image
plt.subplot(2,2,3)
plt.title("Equalized Image (Manual)")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

# Histogram Equalized
plt.subplot(2,2,4)
plt.title("Histogram Equalized")
plt.plot(hist_equalized)

plt.tight_layout()
plt.show()