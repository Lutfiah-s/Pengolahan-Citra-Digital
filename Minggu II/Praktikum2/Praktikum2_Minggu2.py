import cv2
import numpy as np
import matplotlib.pyplot as plt

#=========ANALISIS MODEL WARNA==========

def analyze_color_model_suitability(image, application):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    if application == "skin_detection":
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        mask = ((cr > 135) & (cr < 180) &
                (cb > 85) & (cb < 135))
        return "YCrCb cocok untuk deteksi kulit", mask.astype(np.uint8)*255

    elif application == "shadow_removal":
        v = hsv[:, :, 2]
        equalized = cv2.equalizeHist(v)
        return "HSV cocok untuk manipulasi bayangan", equalized

    elif application == "text_extraction":
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return "Grayscale cukup untuk ekstraksi teks", thresh

    elif application == "object_detection":
        return "RGB cocok untuk deteksi objek", image

    else:
        return "Aplikasi tidak dikenali", image

#==========SIMULASI ALIASING==========

def simulate_image_aliasing(image, factors):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    results = {}

    for f in factors:
        downsampled = gray[::f, ::f]

        upsampled = cv2.resize(downsampled,
                               (gray.shape[1], gray.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        mse = np.mean((gray - upsampled) ** 2)

        results[f] = (upsampled, mse)

    return results

#==========PANGGIL FUNGSI==========

image = cv2.imread("kulit.jpg")

# Analisis warna
info, result_img = analyze_color_model_suitability(image, "skin_detection")
print(info)

plt.imshow(result_img, cmap='gray')
plt.title("Hasil Analisis Warna")
plt.show()


# Simulasi aliasing
aliasing = simulate_image_aliasing(image, [2,4,8])

for factor in aliasing:
    print("Factor:", factor)
    print("MSE:", aliasing[factor][1])

    plt.imshow(aliasing[factor][0], cmap='gray')
    plt.title(f"Downsampling factor {factor}")
    plt.show()