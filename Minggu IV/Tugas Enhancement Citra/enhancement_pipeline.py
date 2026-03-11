import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
import os

# ==============================
# Membuat Dataset Citra
# ==============================

def generate_dataset():

    img = cv2.imread("images/normal.jpg")

    # Underexposed
    under = cv2.convertScaleAbs(img, alpha=0.4, beta=-40)

    # Overexposed
    over = cv2.convertScaleAbs(img, alpha=1.6, beta=60)

    # Uneven Illumination
    rows, cols, ch = img.shape
    gradient = np.tile(np.linspace(0.4,1.6,cols),(rows,1))

    uneven = img.astype(float)
    for i in range(3):
        uneven[:,:,i] *= gradient

    uneven = np.clip(uneven,0,255).astype(np.uint8)

    cv2.imwrite("images/underexposed.jpg", under)
    cv2.imwrite("images/overexposed.jpg", over)
    cv2.imwrite("images/uneven.jpg", uneven)

    print("Dataset berhasil dibuat")


# ==============================
# Transformasi Titik
# ==============================

def negative_transform(image):
    return 255 - image


def log_transform(image):
    image_float = image.astype(np.float32)
    c = 255 / np.log(1 + np.max(image_float))
    log_img = c * np.log(1 + image_float)
    log_img = np.clip(log_img, 0, 255)
    return log_img.astype(np.uint8)

def gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([(i/255.0)**invGamma * 255
        for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


# ==============================
# Histogram Enhancement
# ==============================

def contrast_stretch(image):

    min_val = np.min(image)
    max_val = np.max(image)

    stretched = (image - min_val) * (255/(max_val - min_val))

    return np.array(stretched, dtype=np.uint8)


def histogram_equalization(gray):

    return cv2.equalizeHist(gray)


def clahe_enhancement(gray):

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    return clahe.apply(gray)


# ==============================
# Histogram
# ==============================

def show_histogram(image, title):

    plt.figure()
    plt.hist(image.ravel(), bins=256, range=(0,256))
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()


# ==============================
# Evaluasi
# ==============================

def evaluate(image,name):

    entropy = shannon_entropy(image)
    contrast = image.std()

    print(name)
    print("Entropy :",entropy)
    print("Contrast:",contrast)
    print("-------------------")


# ==============================
# Pipeline Enhancement
# ==============================

def process_image(path):

    print("\nProcessing:",path)

    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # ======================
    # Transformasi Titik
    # ======================

    negative = negative_transform(img)
    log_img = log_transform(img)

    gamma1 = gamma_correction(img,0.5)
    gamma2 = gamma_correction(img,1.5)
    gamma3 = gamma_correction(img,2.0)

    # ======================
    # Histogram Enhancement
    # ======================

    stretch = contrast_stretch(gray)
    equalized = histogram_equalization(gray)
    clahe = clahe_enhancement(gray)

    # ======================
    # Evaluasi
    # ======================

    evaluate(gray,"Original")
    evaluate(stretch,"Contrast Stretching")
    evaluate(equalized,"Histogram Equalization")
    evaluate(clahe,"CLAHE")

    # ======================
    # Visualisasi
    # ======================

    titles = [
        "Original","Negative","Log Transform",
        "Gamma 0.5","Gamma 1.5","Gamma 2.0",
        "Contrast Stretch","Hist Equalization","CLAHE"
    ]

    images = [
        img,negative,log_img,
        gamma1,gamma2,gamma3,
        stretch,equalized,clahe
    ]

    plt.figure(figsize=(12,10))

    for i in range(9):

        plt.subplot(3,3,i+1)

        if len(images[i].shape)==2:
            plt.imshow(images[i],cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB))

        plt.title(titles[i])
        plt.axis("off")

    plt.show()

    show_histogram(gray,"Histogram Original")
    show_histogram(equalized,"Histogram Equalized")
    show_histogram(clahe,"Histogram CLAHE")


# ==============================
# MAIN PROGRAM
# ==============================

if __name__ == "__main__":

    generate_dataset()

    image_paths = [
        "images/underexposed.jpg",
        "images/overexposed.jpg",
        "images/uneven.jpg"
    ]

    for path in image_paths:

        process_image(path)