import cv2
import numpy as np
import matplotlib.pyplot as plt


def medical_image_enhancement(medical_image, modality='X-ray'):
    """
    Adaptive enhancement for medical images
    
    Parameters:
    medical_image: Input medical image
    modality: Image modality ('X-ray', 'MRI', 'CT', 'Ultrasound')
    
    Returns:
    Enhanced image and enhancement report
    """

    report = {}

    # Implement enhancement pipeline based on modality
    if modality == 'X-ray':
        # meningkatkan kontras tulang
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(medical_image)

    elif modality == 'MRI':
        # sharpening untuk memperjelas jaringan
        blur = cv2.GaussianBlur(medical_image,(5,5),0)
        enhanced = cv2.addWeighted(medical_image,1.5,blur,-0.5,0)

    elif modality == 'CT':
        # normalisasi intensitas
        enhanced = cv2.normalize(medical_image,None,0,255,cv2.NORM_MINMAX)

    elif modality == 'Ultrasound':
        # mengurangi speckle noise
        filtered = cv2.medianBlur(medical_image,5)
        enhanced = cv2.equalizeHist(filtered)

    else:
        enhanced = medical_image

    # Generate enhancement report with metrics
    report['mean_intensity'] = np.mean(enhanced)
    report['std_intensity'] = np.std(enhanced)
    report['min_intensity'] = np.min(enhanced)
    report['max_intensity'] = np.max(enhanced)

    return enhanced, report


# =====================================================
# MAIN PROGRAM
# =====================================================

# Load image
image = cv2.imread("medical2_image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise Exception("File medical2_image.jpg tidak ditemukan")

# Pilih modality
modality = 'X-ray'

# Jalankan enhancement
enhanced_image, report = medical_image_enhancement(image, modality)

# Tampilkan report
print("Enhancement Report")
print("======================")

for key, value in report.items():
    print(f"{key} : {value:.2f}")

# =====================================================
# VISUALISASI
# =====================================================

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original Medical Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Enhanced Image")
plt.imshow(enhanced_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()