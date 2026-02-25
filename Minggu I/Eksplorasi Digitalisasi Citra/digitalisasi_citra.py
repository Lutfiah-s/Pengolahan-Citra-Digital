import cv2
import numpy as np
import matplotlib.pyplot as plt

#|| ANALISIS PARAMETER CITRA ||

# Baca citra
img = cv2.imread('citra_analog.jpg')

# Konversi BGR ke RGB (agar warna benar di matplotlib)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Tampilkan citra
plt.imshow(img_rgb)
plt.title("Citra Digital")
plt.axis('off')
plt.show()

# Cetak dimensi
print("Dimensi Citra:", img.shape)

# Tampilkan 5x5 piksel pertama
print("\nRepresentasi Matriks (5x5 piksel pertama):")
print(img[:5, :5])

# Flatten menjadi vektor
img_vector = img.flatten()
print("\nRepresentasi Vektor (20 elemen pertama):")
print(img_vector[:20])

height, width, channels = img.shape

# Resolusi
print("\nResolusi Spasial:", width, "x", height)

# Bit depth
bit_depth = img.dtype
print("Tipe data:", bit_depth)

# Asumsi 8 bit per channel
bits_per_pixel = 8 * channels
print("Bit depth total:", bits_per_pixel, "bit")

# Aspect Ratio
aspect_ratio = width / height
print("Aspect Ratio:", aspect_ratio)

# Ukuran memori
memory_bytes = img.nbytes
memory_kb = memory_bytes / 1024
memory_mb = memory_kb / 1024

print("Ukuran Memori:", round(memory_kb,2), "KB")
print("Ukuran Memori:", round(memory_mb,2), "MB")

# Jika resolusi naik 2x dan bit depth setengah
new_width = width * 2
new_height = height * 2
new_bits = bits_per_pixel / 2

new_memory = new_width * new_height * (new_bits / 8)
print("\nPerkiraan memori baru (byte):", new_memory)
print("Perkiraan memori baru (MB):", round(new_memory/(1024*1024),2))


#|| MANIPULASI DASAR ||

# CROPPING
crop = img[100:400, 100:400]

# RESIZING
resize = cv2.resize(img, (width//2, height//2))

# FLIP
flip = cv2.flip(img, 1)

# Tampilkan hasil
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title("Original")

plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
plt.title("Crop")

plt.subplot(2,2,3)
plt.imshow(cv2.cvtColor(resize, cv2.COLOR_BGR2RGB))
plt.title("Resize")

plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(flip, cv2.COLOR_BGR2RGB))
plt.title("Flip")

plt.tight_layout()
plt.show()