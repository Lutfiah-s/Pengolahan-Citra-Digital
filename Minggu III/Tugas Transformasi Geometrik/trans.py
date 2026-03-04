#24343031_DEALEXA FATIKA DZIKRA_TUGAS TRANSFORMASI GEOMETRIK_MINGGU 3

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# =========================
# 1. Load Dua Gambar
# =========================
img_ref = cv2.imread("doc_lurus.jpg")      # referensi
img_dist = cv2.imread("doc_miring.jpg")    # miring

if img_ref is None or img_dist is None:
    raise Exception("Pastikan kedua gambar ada!")

h, w = img_ref.shape[:2]

# Resize agar ukuran sama
img_dist = cv2.resize(img_dist, (w, h))

gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
gray_dist = cv2.cvtColor(img_dist, cv2.COLOR_BGR2GRAY)

# =========================
# 2. Transformasi Dasar (pada kedua gambar)
# =========================
tx, ty = 60, 40
T = np.array([[1, 0, tx],
              [0, 1, ty],
              [0, 0, 1]], dtype=np.float32)

translated_ref = cv2.warpPerspective(img_ref, T, (w, h))
translated_dist = cv2.warpPerspective(img_dist, T, (w, h))

# Rotasi
angle = 25
center = (w//2, h//2)
R = cv2.getRotationMatrix2D(center, angle, 1.0)

rotated_ref = cv2.warpAffine(img_ref, R, (w, h))
rotated_dist = cv2.warpAffine(img_dist, R, (w, h))

# Scaling
sx, sy = 1.2, 0.8
S = np.array([[sx, 0, 0],
              [0, sy, 0],
              [0, 0, 1]], dtype=np.float32)

scaled_ref = cv2.warpPerspective(img_ref, S, (w, h))
scaled_dist = cv2.warpPerspective(img_dist, S, (w, h))

# =========================
# 3. Affine (3 Titik)
# =========================
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[220,50],[100,250]])

M_affine = cv2.getAffineTransform(pts1, pts2)

affine_ref = cv2.warpAffine(img_ref, M_affine, (w, h))
affine_dist = cv2.warpAffine(img_dist, M_affine, (w, h))

# =========================
# 4. Perspektif (4 Titik)
# =========================
pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
pts2 = np.float32([[50,50],[w-50,30],[30,h-30],[w-30,h-50]])

M_persp = cv2.getPerspectiveTransform(pts1, pts2)

persp_ref = cv2.warpPerspective(img_ref, M_persp, (w, h))

# Transformasi perspektif pada gambar miring + timing
start = time.time()
persp_dist = cv2.warpPerspective(img_dist, M_persp, (w, h), flags=cv2.INTER_LINEAR)
end = time.time()
waktu_persp = end - start

# =========================
# 5. Evaluasi Kualitas (ref vs hasil dist)
# =========================
def mse(imgA, imgB):
    return np.mean((imgA.astype("float") - imgB.astype("float")) ** 2)

def psnr(imgA, imgB):
    m = mse(imgA, imgB)
    if m == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(m))

gray_persp_dist = cv2.cvtColor(persp_dist, cv2.COLOR_BGR2GRAY)

nilai_mse = mse(gray_ref, gray_persp_dist)
nilai_psnr = psnr(gray_ref, gray_persp_dist)

print("\n===== EVALUASI PERSPEKTIF (DIST vs REF) =====")
print("MSE  :", nilai_mse)
print("PSNR :", nilai_psnr)
print("Waktu Komputasi :", waktu_persp)

# =========================
# 6. Tampilkan Semua Hasil
# =========================
plt.figure(figsize=(16,12))

images = [
    ("Ref - Original", img_ref),
    ("Dist - Original", img_dist),
    ("Ref - Translasi", translated_ref),
    ("Dist - Translasi", translated_dist),
    ("Ref - Rotasi", rotated_ref),
    ("Dist - Rotasi", rotated_dist),
    ("Ref - Scaling", scaled_ref),
    ("Dist - Scaling", scaled_dist),
    ("Ref - Affine", affine_ref),
    ("Dist - Affine", affine_dist),
    ("Ref - Perspektif", persp_ref),
    ("Dist - Perspektif", persp_dist),
]

for i, (title, image) in enumerate(images):
    plt.subplot(4,3,i+1)
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

plt.tight_layout()
plt.show()