# ============================================
# Nama: Lutfiah Sahira
# NIM : 24343039
# ============================================

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import richardson_lucy
import time

# ============================================
# 1. LOAD IMAGE
# ============================================
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Gambar tidak ditemukan! Cek path file.")

img = img.astype(np.float32)

# ============================================
# 2. PSF MOTION BLUR
# ============================================
def motion_blur_psf(length=15, angle=30):
    psf = np.zeros((length, length))
    center = length // 2
    
    for i in range(length):
        x = int(center + (i - center) * np.cos(np.deg2rad(angle)))
        y = int(center + (i - center) * np.sin(np.deg2rad(angle)))
        if 0 <= x < length and 0 <= y < length:
            psf[y, x] = 1

    psf /= psf.sum()
    return psf

psf = motion_blur_psf(15, 30)

# ============================================
# 3. DEGRADASI
# ============================================
blurred = convolve2d(img, psf, mode='same', boundary='wrap')

def add_gaussian_noise(image, sigma=20):
    noise = np.random.normal(0, sigma, image.shape)
    return image + noise

gaussian_blur = add_gaussian_noise(blurred, 20)

def add_salt_pepper(image, prob=0.05):
    noisy = image.copy()
    rnd = np.random.rand(*image.shape)
    noisy[rnd < prob/2] = 0
    noisy[rnd > 1 - prob/2] = 255
    return noisy

sp_blur = add_salt_pepper(blurred, 0.05)

# ============================================
# 4. RESTORASI (FIX FFT)
# ============================================
def inverse_filter(image, psf):
    eps = 1e-3
    G = np.fft.fft2(image)
    H = np.fft.fft2(psf, s=image.shape)
    
    F_hat = G / (H + eps)
    result = np.abs(np.fft.ifft2(F_hat))
    return result
    return np.clip(result, 0, 255)

def wiener_filter(image, psf, K=0.01):
    G = np.fft.fft2(image)
    H = np.fft.fft2(psf, s=image.shape)
    
    H_conj = np.conj(H)
    F_hat = (H_conj / (np.abs(H)**2 + K)) * G
    
    result = np.abs(np.fft.ifft2(F_hat))
    return result
    return np.clip(result, 0, 255)

def lucy_filter(image, psf, iter=20):
    image_norm = np.clip(image / 255.0, 0, 1)
    result = richardson_lucy(image_norm, psf, num_iter=iter)
    return np.clip(result * 255, 0, 255)

# ============================================
# 5. EVALUASI
# ============================================
def evaluate(original, restored):
    original = np.clip(original, 0, 255).astype(np.float32)
    restored = np.clip(restored, 0, 255).astype(np.float32)
    
    return {
        "PSNR": psnr(original, restored, data_range=255),
        "MSE": mse(original, restored),
        "SSIM": ssim(original, restored, data_range=255)
    }

# ============================================
# 6. RUN & TIMING
# ============================================
def run_restoration(image, label):
    print(f"\n===== {label} =====")

    results = {}

    # Inverse
    start = time.time()
    inv = inverse_filter(image, psf)
    t_inv = time.time() - start

    # Wiener
    start = time.time()
    wnr = wiener_filter(image, psf)
    t_wnr = time.time() - start

    # Lucy
    start = time.time()
    lucy = lucy_filter(image, psf)
    t_lucy = time.time() - start

    # Evaluasi
    results['Inverse'] = evaluate(img, inv)
    results['Wiener'] = evaluate(img, wnr)
    results['Lucy'] = evaluate(img, lucy)

    # Simpan waktu
    times = {
        'Inverse': t_inv,
        'Wiener': t_wnr,
        'Lucy': t_lucy
    }

    return inv, wnr, lucy, results, times

# ============================================
# 7. EKSEKUSI
# ============================================
inv1, wnr1, lucy1, res1, time1 = run_restoration(blurred, "Motion Blur")
inv2, wnr2, lucy2, res2, time2 = run_restoration(gaussian_blur, "Gaussian + Blur")
inv3, wnr3, lucy3, res3, time3 = run_restoration(sp_blur, "Salt & Pepper + Blur")

# ============================================
# 8. VISUALISASI
# ============================================
def show_results(original, degraded, inv, wnr, lucy, title):
    plt.figure(figsize=(12,8))
    
    plt.subplot(2,3,1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')

    plt.subplot(2,3,2)
    plt.title("Degraded")
    plt.imshow(degraded, cmap='gray')

    plt.subplot(2,3,3)
    plt.title("Inverse")
    plt.imshow(inv, cmap='gray')

    plt.subplot(2,3,5)
    plt.title("Wiener")
    plt.imshow(wnr, cmap='gray')

    plt.subplot(2,3,6)
    plt.title("Lucy")
    plt.imshow(lucy, cmap='gray')

    plt.suptitle(title)
    plt.show()

print("\n========== TABEL HASIL ==========")

def print_table(res, times, label):
    print(f"\n--- {label} ---")
    print("Metode\t\tPSNR\t\tMSE\t\tSSIM\t\tWaktu")
    
    for method in res:
        print(f"{method}\t\t"
              f"{res[method]['PSNR']:.2f}\t\t"
              f"{res[method]['MSE']:.2f}\t\t"
              f"{res[method]['SSIM']:.4f}\t\t"
              f"{times[method]:.4f}")

print_table(res1, time1, "Motion Blur")
print_table(res2, time2, "Gaussian + Blur")
print_table(res3, time3, "Salt & Pepper + Blur")
show_results(img, blurred, inv1, wnr1, lucy1, "Motion Blur")
show_results(img, gaussian_blur, inv2, wnr2, lucy2, "Gaussian + Blur")
show_results(img, sp_blur, inv3, wnr3, lucy3, "Salt & Pepper + Blur")

def show_fft(image, title):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    plt.imshow(magnitude, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()

show_fft(img, "FFT Original")
show_fft(blurred, "FFT Motion Blur")
show_fft(gaussian_blur, "FFT Gaussian + Blur")
show_fft(sp_blur, "FFT S&P + Blur")