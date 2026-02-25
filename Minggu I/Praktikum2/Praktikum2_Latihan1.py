import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_my_image(image_path, compare_path=None):
    """Analyze your own image"""
    
    img = cv2.imread(image_path)
    
    if img is None:
        print("Gambar tidak ditemukan!")
        return
    
    # 1. Dimensi & Resolusi
    height, width, channels = img.shape
    print("Resolusi :", width, "x", height)
    print("Jumlah Channel :", channels)
    
    # 2. Aspect Ratio
    aspect_ratio = width / height
    print("Aspect Ratio :", round(aspect_ratio, 2))
    
    # 3. Konversi ke Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Ukuran Grayscale :", gray.shape)
    
    # 4. Statistik
    stats = {
        "Mean": np.mean(img),
        "Std Dev": np.std(img),
        "Min": np.min(img),
        "Max": np.max(img)
    }
    
    print("\nStatistik Citra:")
    for k, v in stats.items():
        print(k, ":", round(v, 2))
    
    # 5. Histogram semua channel
    colors = ('b', 'g', 'r')
    plt.figure()
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.title("Histogram RGB")
    plt.xlabel("Intensitas")
    plt.ylabel("Jumlah Piksel")
    plt.show()
    
    # 6. Bandingkan dengan citra lain (opsional)
    if compare_path:
        img2 = cv2.imread(compare_path)
        if img2 is not None:
            print("\nPerbandingan Mean:")
            print("Citra 1:", round(np.mean(img),2))
            print("Citra 2:", round(np.mean(img2),2))
    
    return {
        "resolution": (width, height),
        "aspect_ratio": aspect_ratio,
        "statistics": stats
    }


# Contoh penggunaan
# Ganti dengan path fotomu
result = analyze_my_image("Bunga_Saya.jpg", "Sample.jpg")
