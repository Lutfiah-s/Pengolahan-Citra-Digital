import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. FUNGSI EKSTRAKSI ---

def get_chain_code(contour, directions=8):
    """Implementasi Chain Codes 4/8-directional & Normalisasi First Difference"""
    lookup8 = {(1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3, 
               (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7}
    lookup4 = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}
    
    lookup = lookup8 if directions == 8 else lookup4
    chain_code = []
    
    for i in range(len(contour) - 1):
        delta = tuple(contour[i+1][0] - contour[i][0])
        if delta in lookup:
            chain_code.append(lookup[delta])
    
    if not chain_code: return [0]
    # Normalisasi (First Difference) untuk Rotational Invariance
    first_diff = [(chain_code[i] - chain_code[i-1]) % directions for i in range(len(chain_code))]
    return first_diff

def extract_fourier_descriptors(contour, num_descriptors=20):
    contour_complex = contour[:, 0, 0] + 1j * contour[:, 0, 1]
    fft_res = np.fft.fft(contour_complex)
    magnitudes = np.abs(fft_res)
    if len(magnitudes) > 1 and magnitudes[1] != 0:
        normalized_fd = magnitudes / magnitudes[1]
    else:
        normalized_fd = magnitudes
    return normalized_fd[1:num_descriptors+1], fft_res

def reconstruct_fourier(fft_res, num_descriptors):
    fft_copy = np.zeros_like(fft_res)
    fft_copy[:num_descriptors] = fft_res[:num_descriptors]
    fft_copy[-num_descriptors:] = fft_res[-num_descriptors:]
    reconstructed_complex = np.fft.ifft(fft_copy)
    reconstructed_contour = np.zeros((len(reconstructed_complex), 1, 2), dtype=np.int32)
    reconstructed_contour[:, 0, 0] = reconstructed_complex.real
    reconstructed_contour[:, 0, 1] = reconstructed_complex.imag
    return reconstructed_contour

# --- 2. MAIN ---

DATASET_DIR = "dataset"
CATEGORIES = ["apple", "banana", "orange"]
data_features = []
data_labels = []
viz_storage = {cat: [] for cat in CATEGORIES}

print("Sedang mengekstraksi fitur dan menyiapkan visualisasi...")

for label_idx, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATASET_DIR, category)
    if not os.path.exists(folder_path): continue
    
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])
    
    for i, filename in enumerate(files):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is None: continue
        
        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours: continue
        c = max(contours, key=cv2.contourArea)
        
        # Fitur-fitur
        area = cv2.contourArea(c)
        solidity = area / cv2.contourArea(cv2.convexHull(c))
        w, h = cv2.boundingRect(c)[2:]
        aspect_ratio = w/h
        extent = area/(w*h)
        
        M = cv2.moments(c)
        hu = cv2.HuMoments(M).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        
        fd, fft_res = extract_fourier_descriptors(c, 20)
        
        # Simpan Fitur untuk k-NN
        data_features.append([solidity, aspect_ratio, extent, hu_log[0], fd[0], fd[1]])
        data_labels.append(label_idx)
        
        print(f"{category.upper()} | {filename} | Area: {int(area)} | Sol: {solidity:.3f} | AR: {aspect_ratio:.2f} | Hu1: {hu_log[0]:.4f}")
        
        # Simpan data untuk Visualisasi Grid
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.drawContours(img_rgb, [c], -1, (255, 0, 0), 2)
        viz_storage[category].append((img_rgb, area, solidity, fft_res, c))

# --- 3. VISUALISASI ---

for category in CATEGORIES:
    samples = viz_storage[category]
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"GRID DATASET: {category.upper()}", fontsize=16, fontweight='bold')
    axs = axs.flatten()
    for idx, (img_v, a, s, _, _) in enumerate(samples[:9]):
        axs[idx].imshow(img_v)
        axs[idx].set_title(f"Sampel Ke-{idx+1}\n")
        axs[idx].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- 4. ANALISIS DETAIL ---

print("\n--- ANALISIS DETAIL REPRESENTASI BATAS & FOURIER ---")
for category in CATEGORIES:
    img_v, _, _, fft_res, c = viz_storage[category][0] # Ambil sampel pertama
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    
    # Contour
    axs[0].imshow(img_v)
    axs[0].set_title(f"Contour {category}")
    
    # Douglas Peucker
    epsilon = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    img_poly = np.ones_like(img_v)*255
    cv2.drawContours(img_poly, [approx], -1, (255, 0, 0), 2)
    axs[1].imshow(img_poly)
    axs[1].set_title("Douglas-Peucker")
    
    # Fourier N=5
    recon5 = reconstruct_fourier(fft_res, 5)
    img_f5 = np.ones_like(img_v)*255
    cv2.drawContours(img_f5, [recon5], -1, (0, 128, 0), 2)
    axs[2].imshow(img_f5)
    axs[2].set_title("Fourier (N=5)")
    
    # Fourier N=20
    recon20 = reconstruct_fourier(fft_res, 20)
    img_f20 = np.ones_like(img_v)*255
    cv2.drawContours(img_f20, [recon20], -1, (255, 140, 0), 2)
    axs[3].imshow(img_f20)
    axs[3].set_title("Fourier (N=20)")
    plt.show()

# --- 5. KLASIFIKASI ---

X_train, X_test, y_train, y_test = train_test_split(np.array(data_features), np.array(data_labels), test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"\nAKURASI K-NN: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))