# LUTFIAH SAHIRA (24343039)
# PENGOLAHAN CITRA DIGITAL
# TUGAS TRANFORMASI GEOMETRI

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ======================================================
# 1. LOAD IMAGE
# ======================================================

ref = cv2.imread("doc_lurus.jpg")
target = cv2.imread("doc_miring.jpg")

if ref is None or target is None:
    raise Exception("Pastikan file doc_lurus.jpg dan doc_miring.jpg ada.")

h, w = ref.shape[:2]

ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

# ======================================================
# 2. TRANSFORMASI DASAR BERDASARKAN PERBEDAAN KEDUA GAMBAR
# ======================================================

# Translasi berdasarkan selisih titik kiri atas
tx = -250
ty = -200
T = np.array([
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1]
], dtype=np.float32)

translated_ref = cv2.warpPerspective(ref, T, (w, h))
translated_target = cv2.warpPerspective(target, T, (w, h))

# Rotasi target sudut dari kemiringan)
angle = -10  # dibalik agar sejajar
center = (w//2, h//2)

theta = np.deg2rad(angle)

R = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0, 0, 1]
], dtype=np.float32)

rotated_ref = cv2.warpPerspective(ref, R, (w, h))
rotated_target = cv2.warpPerspective(target, R, (w, h))

# Scaling berdasarkan rasio lebar dokumen
scale_x = (950 - 250) / (990 - 340)
scale_y = (2200 - 100) / (1110 - 120)

S = np.array([
    [scale_x, 0, 0],
    [0, scale_y, 0],
    [0, 0, 1]
], dtype=np.float32)

scaled_ref = cv2.warpPerspective(ref, S, (w, h))
scaled_target = cv2.warpPerspective(target, S, (w, h))

# ======================================================
# 3. TITIK SUDUT MANUAL (RELASI DIPERTAHANKAN)
# ======================================================

pts_target = np.float32([
    [340, 120],
    [890, 190],
    [300, 1040],
    [930, 1110]
])

pts_ref = np.float32([
    [250, 100],
    [850, 100],
    [250, 1100],
    [850, 1100]
])

# ======================================================
# 4. AFFINE (3 TITIK)
# ======================================================

# Titik sumber (ambil area dalam gambar)
pts_affine_src = np.float32([
    [100, 100],
    [600, 100],
    [100, 600]
])

# Titik tujuan (dibuat lebih miring dan tertarik)
pts_affine_dst = np.float32([
    [50, 250],
    [750, 50],
    [250, 750]
])

M_affine = cv2.getAffineTransform(pts_affine_src, pts_affine_dst)

affine_ref = cv2.warpAffine(ref, M_affine, (w, h))
affine_target = cv2.warpAffine(target, M_affine, (w, h))

# ======================================================
# 5. PERSPECTIVE REGISTRATION (4 TITIK)
# ======================================================

# Titik sudut dokumen
M_persp = cv2.getPerspectiveTransform(pts_target, pts_ref)

# Nearest Neighbor
start = time.time()
nearest = cv2.warpPerspective(target, M_persp, (w, h), flags=cv2.INTER_NEAREST)
time_nearest = time.time() - start

# Bilinear
start = time.time()
bilinear = cv2.warpPerspective(target, M_persp, (w, h), flags=cv2.INTER_LINEAR)
time_bilinear = time.time() - start

# Bicubic
start = time.time()
bicubic = cv2.warpPerspective(target, M_persp, (w, h), flags=cv2.INTER_CUBIC)
time_bicubic = time.time() - start

# ======================================================
# 6. METRIK EVALUASI
# ======================================================

def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def psnr(img1, img2):
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse_val))

nearest_gray = cv2.cvtColor(nearest, cv2.COLOR_BGR2GRAY)
bilinear_gray = cv2.cvtColor(bilinear, cv2.COLOR_BGR2GRAY)
bicubic_gray = cv2.cvtColor(bicubic, cv2.COLOR_BGR2GRAY)

print("===== HASIL REGISTRASI =====")
print("Nearest  PSNR:", psnr(ref_gray, nearest_gray), "MSE:",mse(ref_gray, nearest_gray), "Computation Time:", time_nearest)
print("Bilinear PSNR:", psnr(ref_gray, bilinear_gray),"MSE:",mse(ref_gray, bilinear_gray),"Computation Time:", time_bilinear)
print("Bicubic  PSNR:", psnr(ref_gray, bicubic_gray), "MSE:",mse(ref_gray, bicubic_gray), "Computation Time:", time_bicubic)

# ======================================================
# 7. VISUALISASI
# ======================================================

plt.figure(figsize=(18,15))

images = [
    ("Reference", ref),
    ("Target", target),
    ("Translasi (Ref)", translated_ref),
    ("Translasi (Target)", translated_target),
    ("Rotasi (Ref)", rotated_ref),
    ("Rotasi (Target)", rotated_target),
    ("Scaling (Ref)", scaled_ref),
    ("Scaling (Target)", scaled_target),
    ("Affine (Ref)", affine_ref),
    ("Affine (Target)", affine_target),
    ("Nearest Registered", nearest),
    ("Bilinear Registered", bilinear),
    ("Bicubic Registered", bicubic)
]

for i, (title, img) in enumerate(images):
    plt.subplot(3,5,i+1)
    plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

plt.tight_layout()
plt.show()