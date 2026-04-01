# ==========================================================
# NAMA : LUTFIAH SAHIRA
# NIM  : 24343039
# ==========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import time

from skimage import data

# ==========================================================
# 1. LOAD IMAGE
# ==========================================================
img = data.camera()

# ==========================================================
# 2. TAMBAH NOISE PERIODIK
# ==========================================================
rows, cols = img.shape
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)

noise = 30 * np.sin(2 * np.pi * X / 20)
noisy_img = np.clip(img + noise, 0, 255)

# ==========================================================
# 3. FFT
# ==========================================================
def compute_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    phase = np.angle(fshift)
    return fshift, magnitude, phase

fshift, magnitude, phase = compute_fft(img)

# ==========================================================
# 4. IDENTIFIKASI FREKUENSI DOMINAN
# ==========================================================
threshold = np.percentile(magnitude, 99)
dominant_freq = magnitude > threshold

# ==========================================================
# 5. REKONSTRUKSI
# ==========================================================
mag_only = np.abs(fshift)
recon_mag = np.abs(np.fft.ifft2(np.fft.ifftshift(mag_only)))

phase_only = np.exp(1j * phase)
recon_phase = np.abs(np.fft.ifft2(np.fft.ifftshift(phase_only)))

# ==========================================================
# 6. FILTER FUNCTIONS
# ==========================================================
def ideal_lowpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i-crow)**2 + (j-ccol)**2) <= cutoff:
                mask[i,j] = 1
    return mask

def gaussian_lowpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            d = (i-crow)**2 + (j-ccol)**2
            mask[i,j] = np.exp(-d/(2*(cutoff**2)))
    return mask

def ideal_highpass(shape, cutoff):
    return 1 - ideal_lowpass(shape, cutoff)

def gaussian_highpass(shape, cutoff):
    return 1 - gaussian_lowpass(shape, cutoff)

def bandpass_filter(shape, low, high):
    return ideal_lowpass(shape, high) - ideal_lowpass(shape, low)

def apply_filter(fshift, mask):
    result = fshift * mask
    return np.abs(np.fft.ifft2(np.fft.ifftshift(result)))

# ==========================================================
# 7. VARIASI CUTOFF
# ==========================================================
cutoffs = [10, 30, 60]

results = {}
times = {}

for c in cutoffs:
    start = time.time()
    mask = gaussian_lowpass(img.shape, c)
    results[f'gauss_lp_{c}'] = apply_filter(fshift, mask)
    times[f'gauss_lp_{c}'] = time.time() - start

# Highpass
mask_hp = gaussian_highpass(img.shape, 30)
img_high = apply_filter(fshift, mask_hp)

# Bandpass
mask_bp = bandpass_filter(img.shape, 10, 50)
img_band = apply_filter(fshift, mask_bp)

# ==========================================================
# 8. NOTCH FILTER (NOISE PERIODIK)
# ==========================================================
fshift_noisy, mag_noisy, _ = compute_fft(noisy_img)
mask_notch = np.ones_like(fshift_noisy)

center = (rows//2, cols//2)
mask_notch[center[0]-10:center[0]+10, center[1]+50:center[1]+70] = 0
mask_notch[center[0]-10:center[0]+10, center[1]-70:center[1]-50] = 0

img_notch = apply_filter(fshift_noisy, mask_notch)

# ==========================================================
# 9. FILTERING SPASIAL (PERBANDINGAN)
# ==========================================================
start = time.time()
spatial = cv2.GaussianBlur(img, (5,5), 0)
time_spatial = time.time() - start

# ==========================================================
# 10. WAVELET (HAAR & DB4)
# ==========================================================
start = time.time()
coeffs_haar = pywt.wavedec2(img, 'haar', level=2)
recon_haar = pywt.waverec2(coeffs_haar, 'haar')
time_haar = time.time() - start

start = time.time()
coeffs_db4 = pywt.wavedec2(img, 'db4', level=2)
recon_db4 = pywt.waverec2(coeffs_db4, 'db4')
time_db4 = time.time() - start

# ambil detail wavelet
cA, (cH, cV, cD), _ = coeffs_haar

# ==========================================================
# 11. PSNR
# ==========================================================
def psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

metrics = {}

metrics['spatial'] = psnr(img, spatial)
metrics['notch'] = psnr(img, img_notch)
metrics['highpass'] = psnr(img, img_high)
metrics['bandpass'] = psnr(img, img_band)
metrics['haar'] = psnr(img, recon_haar)
metrics['db4'] = psnr(img, recon_db4)

for c in cutoffs:
    metrics[f'gauss_lp_{c}'] = psnr(img, results[f'gauss_lp_{c}'])

# ==========================================================
# 12. VISUALISASI
# ==========================================================
plt.figure(figsize=(14,10))

plt.subplot(3,4,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(3,4,2), plt.imshow(noisy_img, cmap='gray'), plt.title("Noisy")
plt.subplot(3,4,3), plt.imshow(magnitude, cmap='gray'), plt.title("Spectrum")
plt.subplot(3,4,4), plt.imshow(dominant_freq, cmap='gray'), plt.title("Dominant Freq")

plt.subplot(3,4,5), plt.imshow(results['gauss_lp_10'], cmap='gray'), plt.title("LPF 10")
plt.subplot(3,4,6), plt.imshow(results['gauss_lp_30'], cmap='gray'), plt.title("LPF 30")
plt.subplot(3,4,7), plt.imshow(results['gauss_lp_60'], cmap='gray'), plt.title("LPF 60")
plt.subplot(3,4,8), plt.imshow(img_high, cmap='gray'), plt.title("Highpass")

plt.subplot(3,4,9), plt.imshow(img_band, cmap='gray'), plt.title("Bandpass")
plt.subplot(3,4,10), plt.imshow(img_notch, cmap='gray'), plt.title("Notch")
plt.subplot(3,4,11), plt.imshow(spatial, cmap='gray'), plt.title("Spatial")
plt.subplot(3,4,12), plt.imshow(recon_db4, cmap='gray'), plt.title("Wavelet DB4")

plt.tight_layout()
plt.show()

# ==========================================================
# 13. VISUALISASI WAVELET DETAIL
# ==========================================================
plt.figure(figsize=(6,6))
plt.subplot(2,2,1), plt.imshow(cA, cmap='gray'), plt.title("Approx")
plt.subplot(2,2,2), plt.imshow(cH, cmap='gray'), plt.title("Horizontal")
plt.subplot(2,2,3), plt.imshow(cV, cmap='gray'), plt.title("Vertical")
plt.subplot(2,2,4), plt.imshow(cD, cmap='gray'), plt.title("Diagonal")
plt.tight_layout()
plt.show()

# ==========================================================
# 14. PRINT METRIK
# ==========================================================
print("\n===== HASIL PERBANDINGAN =====")
for k, v in metrics.items():
    print(f"{k:15s} : PSNR = {v:.2f}")

print("\n===== WAKTU KOMPUTASI =====")
print(f"Spatial   : {time_spatial:.4f} detik")
print(f"Haar      : {time_haar:.4f} detik")
print(f"DB4       : {time_db4:.4f} detik")
for k, v in times.items():
    print(f"{k:15s} : {v:.4f} detik")