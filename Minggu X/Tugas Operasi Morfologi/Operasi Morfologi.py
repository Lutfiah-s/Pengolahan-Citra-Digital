import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# =========================
# GENERATE CITRA A (TEXT + NOISE)
# =========================
img_text = np.ones((300, 600), dtype=np.uint8) * 255

cv2.putText(img_text, 'OCR TEST 123', (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0), 3)

noise = np.random.randint(0, 256, img_text.shape, dtype=np.uint8)
img_text[noise < 20] = 0
img_text[noise > 235] = 255

img_text_color = cv2.cvtColor(img_text, cv2.COLOR_GRAY2BGR)

# =========================
# GENERATE CITRA B (OBJECT)
# =========================
img_obj = np.zeros((400, 400), dtype=np.uint8)

cv2.circle(img_obj, (150, 200), 80, 255, -1)
cv2.circle(img_obj, (230, 200), 80, 255, -1)
cv2.circle(img_obj, (190, 120), 70, 255, -1)
cv2.circle(img_obj, (190, 280), 70, 255, -1)

img_obj_color = cv2.cvtColor(img_obj, cv2.COLOR_GRAY2BGR)

# =========================
# STRUCTURING ELEMENT EXPERIMENT
# =========================
gray = img_text.copy()

sizes = [3,5,7]
shapes = {
    "RECT": cv2.MORPH_RECT,
    "ELLIPSE": cv2.MORPH_ELLIPSE,
    "CROSS": cv2.MORPH_CROSS
}

print("\n=== TABEL STRUCTURING ELEMENT ===")
print("Bentuk | Ukuran | Waktu | Noise Level")

for s in sizes:
    for name, shape in shapes.items():
        kernel = cv2.getStructuringElement(shape, (s,s))
        
        start = time.time()
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        elapsed = time.time() - start
        
        noise_level = np.sum(result < 128)
        print(f"{name} | {s}x{s} | {elapsed:.5f} | {noise_level}")

# =========================
# EROSI & DILASI
# =========================
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

print("\n=== TABEL EROSI & DILASI ===")
print("Iterasi | Waktu Erosi | Waktu Dilasi")

for i in [1,2,3]:
    t1 = time.time()
    er = cv2.erode(gray, kernel, iterations=i)
    t1 = time.time() - t1
    
    t2 = time.time()
    dl = cv2.dilate(gray, kernel, iterations=i)
    t2 = time.time() - t2
    
    print(f"{i} | {t1:.5f} | {t2:.5f}")

# =========================
# OPERASI MORFOLOGI
# =========================
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

# =========================
# PERBANDINGAN NUMERIK
# =========================
print("\n=== PERBANDINGAN MORFOLOGI ===")
print("Jenis | Jumlah Pixel (<128)")

print(f"Original | {np.sum(gray < 128)}")
print(f"Opening  | {np.sum(opening < 128)}")
print(f"Closing  | {np.sum(closing < 128)}")
print(f"Gradient | {np.sum(gradient < 128)}")
print(f"TopHat   | {np.sum(tophat < 128)}")
print(f"BlackHat | {np.sum(blackhat < 128)}")

# =========================
# OCR PREPROCESSING
# =========================
start = time.time()

_, thresh = cv2.threshold(opening, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

ocr_time = time.time() - start

before = np.sum(gray < 128)
after = np.sum(thresh < 128)

print("\n=== TABEL OCR ===")
print("Kondisi | Pixel Teks")
print(f"Sebelum | {before}")
print(f"Sesudah | {after}")
print(f"Waktu OCR | {ocr_time:.5f} detik")

# =========================
# OBJECT COUNTING
# =========================
start = time.time()

_, thresh_obj = cv2.threshold(img_obj, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

opening_obj = cv2.morphologyEx(thresh_obj, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening_obj, kernel, iterations=3)

dist_transform = cv2.distanceTransform(opening_obj, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform,
                           0.5 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(img_obj_color, markers)

object_count = len(np.unique(markers)) - 2
manual_count = 4

count_time = time.time() - start
accuracy = (object_count / manual_count) * 100

print("\n=== TABEL COUNTING ===")
print("Metode | Jumlah")
print(f"Manual | {manual_count}")
print(f"Otomatis | {object_count}")
print(f"Akurasi | {accuracy:.2f}%")
print(f"Waktu Counting | {count_time:.5f} detik")

# =========================
# VISUALISASI TEXT
# =========================
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(gray, cmap='gray')

plt.subplot(2,3,2)
plt.title("Opening")
plt.imshow(opening, cmap='gray')

plt.subplot(2,3,3)
plt.title("Closing")
plt.imshow(closing, cmap='gray')

plt.subplot(2,3,4)
plt.title("Gradient")
plt.imshow(gradient, cmap='gray')

plt.subplot(2,3,5)
plt.title("Top Hat")
plt.imshow(tophat, cmap='gray')

plt.subplot(2,3,6)
plt.title("Black Hat")
plt.imshow(blackhat, cmap='gray')

plt.tight_layout()
plt.show()

# =========================
# VISUALISASI OBJECT
# =========================
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.title("Original Object")
plt.imshow(img_obj, cmap='gray')

plt.subplot(1,3,2)
plt.title("Threshold")
plt.imshow(thresh_obj, cmap='gray')

plt.subplot(1,3,3)
plt.title("Watershed")

img_ws = img_obj_color.copy()
img_ws[markers == -1] = [255,0,0]

plt.imshow(cv2.cvtColor(img_ws, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()