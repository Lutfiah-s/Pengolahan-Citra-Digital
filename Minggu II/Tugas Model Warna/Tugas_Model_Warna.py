import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans

# ======================================================
# FUNGSI KUANTISASI
# ======================================================

def uniform_quantization(image, levels=16):
    step = 256 // levels
    return (image // step) * step

def non_uniform_quantization(image, clusters=16):
    if len(image.shape) == 2:  # grayscale
        data = image.reshape(-1,1)
    else:
        data = image.reshape((-1, image.shape[-1]))

    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=10)
    kmeans.fit(data)

    new_pixels = kmeans.cluster_centers_[kmeans.labels_]
    quantized = new_pixels.reshape(image.shape)

    return quantized.astype(np.uint8)

# ======================================================
# DAFTAR GAMBAR
# ======================================================

image_paths = ["apple_bright.jpg", "apple_normal.jpg", "apple_low.jpg"]

# ======================================================
# PROSES SETIAP GAMBAR
# ======================================================

for path in image_paths:

    print("\n====================================")
    print("Processing:", path)
    print("====================================")

    # Load image
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # ======================================================
    # TAMPILKAN ORIGINAL & MODEL WARNA
    # ======================================================

    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    plt.title("Original RGB")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.title("Grayscale")
    plt.imshow(img_gray, cmap='gray')
    plt.axis("off")

    plt.subplot(2,2,3)
    plt.title("HSV")
    plt.imshow(img_hsv)
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.title("LAB")
    plt.imshow(img_lab)
    plt.axis("off")

    plt.show()

    # ======================================================
    # KUANTISASI UNIFORM
    # ======================================================

    start = time.time()
    gray_uniform = uniform_quantization(img_gray, 16)
    hsv_uniform = uniform_quantization(img_hsv, 16)
    rgb_uniform = uniform_quantization(img_rgb, 16)
    uniform_time = time.time() - start

    # ======================================================
    # KUANTISASI NON-UNIFORM (KMEANS)
    # ======================================================

    start = time.time()
    gray_kmeans = non_uniform_quantization(img_gray, 16)
    hsv_kmeans = non_uniform_quantization(img_hsv, 16)
    rgb_kmeans = non_uniform_quantization(img_rgb, 16)
    kmeans_time = time.time() - start

    # ======================================================
    # PERBANDINGAN RGB
    # ======================================================

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Original RGB")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Uniform 16")
    plt.imshow(rgb_uniform)
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("KMeans 16")
    plt.imshow(rgb_kmeans)
    plt.axis("off")

    plt.show()

    # ======================================================
    # HISTOGRAM GRAYSCALE
    # ======================================================

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("Gray Original Histogram")
    plt.hist(img_gray.ravel(), bins=256)

    plt.subplot(1,2,2)
    plt.title("Gray Uniform Histogram")
    plt.hist(gray_uniform.ravel(), bins=16)

    plt.show()

    # ======================================================
    # SEGMENTASI WARNA MERAH (HSV)
    # ======================================================

    lower = np.array([0, 50, 50])
    upper = np.array([10, 255, 255])

    mask_original = cv2.inRange(img_hsv, lower, upper)
    mask_uniform = cv2.inRange(hsv_uniform, lower, upper)
    mask_kmeans = cv2.inRange(hsv_kmeans, lower, upper)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Mask Original")
    plt.imshow(mask_original, cmap='gray')
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Mask Uniform")
    plt.imshow(mask_uniform, cmap='gray')
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Mask KMeans")
    plt.imshow(mask_kmeans, cmap='gray')
    plt.axis("off")

    plt.show()

    # ======================================================
    # HITUNG PIXEL TERDETEKSI
    # ======================================================

    original_pixels = np.sum(mask_original > 0)
    uniform_pixels = np.sum(mask_uniform > 0)
    kmeans_pixels = np.sum(mask_kmeans > 0)

    print("\nDetected Pixels:")
    print("Original :", original_pixels)
    print("Uniform  :", uniform_pixels)
    print("KMeans   :", kmeans_pixels)

    # ======================================================
    # ANALISIS MEMORI & RASIO KOMPRESI
    # ======================================================

    h, w = img_rgb.shape[:2]

    original_bits = h * w * 3 * 8
    quantized_bits = h * w * 3 * 4

    compression_ratio = original_bits / quantized_bits

    print("\nMemory Analysis:")
    print("Original bits:", original_bits)
    print("Quantized bits (theoretical):", quantized_bits)
    print("Compression ratio:", compression_ratio)

    # ======================================================
    # WAKTU KOMPUTASI
    # ======================================================

    print("\nComputation Time:")
    print("Uniform:", uniform_time, "seconds")
    print("KMeans :", kmeans_time, "seconds")