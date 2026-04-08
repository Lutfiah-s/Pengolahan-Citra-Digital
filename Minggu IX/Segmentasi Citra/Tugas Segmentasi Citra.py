# ========================================
# ======== Nama : Lutfiah Sahira =========
# ======== NIM  : 24343039 ===============
# ========================================

import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from skimage import data

#==========================================================
# ==============IMPLEMENTASI METODE THRESHOLDING===========
#==========================================================
# 1. Global Threshold (Manual)
def global_threshold(image, T=127):
    _, thresh = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
    return thresh

# 2. Otsu's Thresholding
def otsu_threshold(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# 3. Adaptive Mean
def adaptive_mean(image):
    return cv2.adaptiveThreshold(image, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY,
                                 11, 2)
# 4. Adaptive Gaussian
def adaptive_gaussian(image):
    return cv2.adaptiveThreshold(image, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,
                                 11, 2)

#==========================================================
# ===============IMPLEMENTASI EDGE DETECTION===============
#==========================================================
# 1.Sobel
def sobel_edge(image):
    # Gradient X dan Y
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    
    return sobelx, sobely, magnitude

# 2. Prewitt
def prewitt_edge(image):
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    
    prewittx = cv2.filter2D(image, -1, kernelx)
    prewitty = cv2.filter2D(image, -1, kernely)
    
    magnitude = np.sqrt(prewittx**2 + prewitty**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return prewittx, prewitty, magnitude

# 3. Canny
def canny_edge(image):
    c1 = cv2.Canny(image, 50, 150)
    c2 = cv2.Canny(image, 100, 200)
    c3 = cv2.Canny(image, 150, 250)
    
    return c1, c2, c3

#==========================================================
# ===============IMPLEMENTASI REGION BASED=================
#==========================================================
# 1. Region Growing
def region_growing(image, seed):
    h, w = image.shape
    visited = np.zeros((h,w), dtype=bool)
    region = np.zeros((h,w), dtype=np.uint8)

    threshold = 10
    seed_value = image[seed]

    stack = [seed]

    while stack:
        x, y = stack.pop()

        if visited[x, y]:
            continue

        visited[x, y] = True

        if abs(int(image[x,y]) - int(seed_value)) < threshold:
            region[x,y] = 255

            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < h and 0 <= ny < w:
                        stack.append((nx, ny))

    return region

# 2. Watershed
def watershed_segmentation(image):
    import cv2
    import numpy as np

    # 1. Threshold (pakai Otsu)
    _, thresh = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Background pasti
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 4. Foreground pasti (distance transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)

    # 5. Area tidak pasti
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    # 7. Watershed
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(image_color, markers)

    # Boundary = -1
    image_color[markers == -1] = [255, 0, 0]

    return image_color, markers

# 3. Connected
def connected_components(image):
    _, thresh = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels = cv2.connectedComponents(thresh)
    
    return labels

def ensure_binary(mask):
    if mask.dtype != np.uint8:
        mask = np.uint8(mask)
    return np.where(mask > 0, 255, 0).astype(np.uint8)

def edge_to_binary(edge_image):
    if edge_image.dtype != np.uint8:
        edge_image = np.uint8(np.clip(edge_image, 0, 255))
    _, binary = cv2.threshold(edge_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def markers_to_mask(markers):
    return np.where(markers > 1, 255, 0).astype(np.uint8)

def build_reference_mask(image, image_name):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)

    if image_name == "Data Text":
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    elif image_name == "Coin":
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    else:
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return ensure_binary(mask)

def build_reference_edge(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return ensure_binary(cv2.Canny(blurred, 100, 200))

def compute_metrics(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0

    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "IoU": iou,
        "Dice": dice,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    }

def add_gaussian_noise(image, sigma=20):
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.uint8(np.clip(noisy, 0, 255))

def change_illumination(image, alpha=1.2, beta=20):
    illuminated = image.astype(np.float32) * alpha + beta
    return np.uint8(np.clip(illuminated, 0, 255))

def format_table(rows, headers):
    str_rows = []
    for row in rows:
        formatted_row = []
        for item in row:
            if isinstance(item, float):
                formatted_row.append(f"{item:.4f}")
            else:
                formatted_row.append(str(item))
        str_rows.append(formatted_row)

    widths = []
    for idx, header in enumerate(headers):
        max_row_width = max((len(row[idx]) for row in str_rows), default=0)
        widths.append(max(len(header), max_row_width))

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header_line = "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |"

    lines = [separator, header_line, separator]
    for row in str_rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    lines.append(separator)
    return "\n".join(lines)

def print_metric_table(title, rows):
    print(f"\n{title}")
    print(format_table(
        rows,
        ["Image", "Method", "IoU", "Dice", "Accuracy", "Precision", "Recall"]
    ))

def print_timing_table(title, rows):
    print(f"\n{title}")
    print(format_table(
        rows,
        ["Image", "Method", "Time (ms)"]
    ))

def print_robustness_table(title, rows):
    print(f"\n{title}")
    print(format_table(
        rows,
        ["Image", "Method", "Base IoU", "Noise IoU", "Illum IoU"]
    ))

def create_overlay(image, gt_mask, pred_mask):
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 1)
    cv2.drawContours(overlay, pred_contours, -1, (255, 0, 0), 1)
    return overlay

# ==========================================================
# ======================MAIN PROGRAM========================
# ==========================================================

# Load dataset
img1 = data.text()
img2 = data.camera()
img3 = data.coins()

images = [img1, img2, img3]
titles = ["Bimodal (Text)", "Iluminasi Tidak Merata (Camera)", "Overlapping (Coins)"]
row_titles = ["Data Text", "Camera", "Coin"]
seeds = [(75,85), (100,150), (150,200)]

# Visualisasi Original Citra
plt.figure(figsize=(10,4))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# Jalankan ke Semua Citra (Thresholding)
results = []

for img in images:
    g = global_threshold(img)
    o = otsu_threshold(img)
    am = adaptive_mean(img)
    ag = adaptive_gaussian(img)
    
    results.append([g, o, am, ag])

# Visualisasi Hasil Thresholding
methods = ["Global", "Otsu", "Adaptive Mean", "Adaptive Gaussian"]

plt.figure(figsize=(15,9))
plt.suptitle("Implementasi Metode Thresholding", fontsize=16)
for i in range(3):
    threshold_images = [images[i], results[i][0], results[i][1], results[i][2], results[i][3]]
    threshold_titles = ["Original", "Global", "Otsu", "Adaptive Mean", "Adaptive Gaussian"]

    for j in range(5):
        ax = plt.subplot(3,5,i*5 + j + 1)
        ax.imshow(threshold_images[j], cmap='gray')
        ax.set_title(threshold_titles[j])
        ax.axis('off')

        if j == 0:
            ax.set_ylabel(row_titles[i], rotation=90, size='large')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Jalankan ke Semua Citra (Edge Detection)
edge_results = []

for img in images:
    sobel = sobel_edge(img)
    prewitt = prewitt_edge(img)
    canny = canny_edge(img)
    
    edge_results.append({
        "sobel": sobel,
        "prewitt": prewitt,
        "canny": canny
    })

# Visualisasi Hasil Edge Detection
plt.figure(figsize=(18,9))
plt.suptitle("Implementasi Edge Detection", fontsize=16)
for i in range(3):
    edge_images = [
        images[i],
        edge_results[i]["sobel"][2],
        edge_results[i]["prewitt"][2],
        edge_results[i]["canny"][0],
        edge_results[i]["canny"][1],
        edge_results[i]["canny"][2]
    ]
    edge_titles = [
        "Original",
        "Sobel",
        "Prewitt",
        "Canny 50-150",
        "Canny 100-200",
        "Canny 150-250"
    ]

    for j in range(6):
        ax = plt.subplot(3,6,i*6 + j + 1)
        ax.imshow(edge_images[j], cmap='gray')
        ax.set_title(edge_titles[j])
        ax.axis('off')

        if j == 0:
            ax.set_ylabel(row_titles[i], rotation=90, size='large')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Jalankan ke Semua Citra (Region Based)
region_based_results = []

for img, seed in zip(images, seeds):
    regionGrowing = region_growing(img, seed)
    watershed, markers = watershed_segmentation(img)
    connected = connected_components(img)
    
    region_based_results.append({
        "seed": seed,
        "region_growing": regionGrowing,
        "watershed": watershed,
        "markers": markers,
        "connected": connected
    })

# Visualisasi Hasil Region Based
plt.figure(figsize=(12,9))
plt.suptitle("Implementasi Region-Based", fontsize=16)
for i in range(3):
    data = region_based_results[i]
    region_images = [
        images[i],
        data["region_growing"],
        data["watershed"],
        data["connected"]
    ]
    region_titles = ["Original", "Region Growing", "Watershed", "Connected"]

    for j in range(4):
        ax = plt.subplot(3,4,i*4 + j + 1)
        if j == 3:
            ax.imshow(region_images[j], cmap='nipy_spectral')
        elif j == 2:
            ax.imshow(region_images[j])
        else:
            ax.imshow(region_images[j], cmap='gray')
        ax.set_title(region_titles[j])
        ax.axis('off')

        if j == 0:
            ax.set_ylabel(row_titles[i], rotation=90, size='large')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#==========================================================
#===================EVALUASI KOMPREHENSIF==================
#==========================================================
np.random.seed(42)

reference_masks = [build_reference_mask(img, row_titles[i]) for i, img in enumerate(images)]
reference_edges = [build_reference_edge(img) for img in images]

threshold_specs = [
    ("Global", global_threshold),
    ("Otsu", otsu_threshold),
    ("Adaptive Mean", adaptive_mean),
    ("Adaptive Gaussian", adaptive_gaussian)
]

edge_specs = [
    ("Sobel", lambda image, idx: edge_to_binary(sobel_edge(image)[2])),
    ("Prewitt", lambda image, idx: edge_to_binary(prewitt_edge(image)[2])),
    ("Canny 50-150", lambda image, idx: ensure_binary(canny_edge(image)[0])),
    ("Canny 100-200", lambda image, idx: ensure_binary(canny_edge(image)[1])),
    ("Canny 150-250", lambda image, idx: ensure_binary(canny_edge(image)[2]))
]

region_specs = [
    ("Region Growing", lambda image, idx: ensure_binary(region_growing(image, seeds[idx]))),
    ("Watershed", lambda image, idx: markers_to_mask(watershed_segmentation(image)[1])),
    ("Connected", lambda image, idx: ensure_binary(connected_components(image)))
]

threshold_metric_rows = []
edge_metric_rows = []
region_metric_rows = []

threshold_timing_rows = []
edge_timing_rows = []
region_timing_rows = []

threshold_robustness_rows = []
edge_robustness_rows = []
region_robustness_rows = []

overlay_examples = {
    image_name: {
        "threshold": None,
        "edge": None,
        "region": None
    }
    for image_name in row_titles
}

for i, image in enumerate(images):
    image_name = row_titles[i]
    gt_mask = reference_masks[i]
    gt_edge = reference_edges[i]
    noisy_image = add_gaussian_noise(image)
    illuminated_image = change_illumination(image)

    best_threshold = None
    best_threshold_iou = -1
    for method_name, method_fn in threshold_specs:
        start = time.perf_counter()
        prediction = ensure_binary(method_fn(image))
        elapsed_ms = (time.perf_counter() - start) * 1000

        metrics = compute_metrics(prediction, gt_mask)
        noise_metrics = compute_metrics(ensure_binary(method_fn(noisy_image)), gt_mask)
        illum_metrics = compute_metrics(ensure_binary(method_fn(illuminated_image)), gt_mask)

        threshold_metric_rows.append([
            image_name, method_name, metrics["IoU"], metrics["Dice"],
            metrics["Accuracy"], metrics["Precision"], metrics["Recall"]
        ])
        threshold_timing_rows.append([image_name, method_name, elapsed_ms])
        threshold_robustness_rows.append([
            image_name, method_name, metrics["IoU"],
            noise_metrics["IoU"], illum_metrics["IoU"]
        ])

        if metrics["IoU"] > best_threshold_iou:
            best_threshold_iou = metrics["IoU"]
            best_threshold = (method_name, prediction, gt_mask)

    if best_threshold is not None:
        overlay_examples[image_name]["threshold"] = best_threshold

    best_edge = None
    best_edge_iou = -1
    for method_name, method_fn in edge_specs:
        start = time.perf_counter()
        prediction = method_fn(image, i)
        elapsed_ms = (time.perf_counter() - start) * 1000

        metrics = compute_metrics(prediction, gt_edge)
        noise_metrics = compute_metrics(method_fn(noisy_image, i), gt_edge)
        illum_metrics = compute_metrics(method_fn(illuminated_image, i), gt_edge)

        edge_metric_rows.append([
            image_name, method_name, metrics["IoU"], metrics["Dice"],
            metrics["Accuracy"], metrics["Precision"], metrics["Recall"]
        ])
        edge_timing_rows.append([image_name, method_name, elapsed_ms])
        edge_robustness_rows.append([
            image_name, method_name, metrics["IoU"],
            noise_metrics["IoU"], illum_metrics["IoU"]
        ])

        if metrics["IoU"] > best_edge_iou:
            best_edge_iou = metrics["IoU"]
            best_edge = (method_name, prediction, gt_edge)

    if best_edge is not None:
        overlay_examples[image_name]["edge"] = best_edge

    best_region = None
    best_region_iou = -1
    for method_name, method_fn in region_specs:
        start = time.perf_counter()
        prediction = method_fn(image, i)
        elapsed_ms = (time.perf_counter() - start) * 1000

        metrics = compute_metrics(prediction, gt_mask)
        noise_metrics = compute_metrics(method_fn(noisy_image, i), gt_mask)
        illum_metrics = compute_metrics(method_fn(illuminated_image, i), gt_mask)

        region_metric_rows.append([
            image_name, method_name, metrics["IoU"], metrics["Dice"],
            metrics["Accuracy"], metrics["Precision"], metrics["Recall"]
        ])
        region_timing_rows.append([image_name, method_name, elapsed_ms])
        region_robustness_rows.append([
            image_name, method_name, metrics["IoU"],
            noise_metrics["IoU"], illum_metrics["IoU"]
        ])

        if metrics["IoU"] > best_region_iou:
            best_region_iou = metrics["IoU"]
            best_region = (method_name, prediction, gt_mask)

    if best_region is not None:
        overlay_examples[image_name]["region"] = best_region


# Visualisasi Overlay Contour dan Perbandingan Ground Truth
for image_name in row_titles:
    image_index = row_titles.index(image_name)
    threshold_example = overlay_examples[image_name]["threshold"]
    edge_example = overlay_examples[image_name]["edge"]
    region_example = overlay_examples[image_name]["region"]

    plt.figure(figsize=(18, 4))
    plt.suptitle(f"Overlay Contour dan Perbandingan Ground Truth - {image_name}", fontsize=16)

    plt.subplot(1, 6, 1)
    plt.imshow(images[image_index], cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 6, 2)
    plt.imshow(reference_masks[image_index], cmap='gray')
    plt.title("Pseudo GT Mask")
    plt.axis('off')

    plt.subplot(1, 6, 3)
    plt.imshow(reference_edges[image_index], cmap='gray')
    plt.title("Pseudo GT Edge")
    plt.axis('off')

    plt.subplot(1, 6, 4)
    plt.imshow(create_overlay(images[image_index], threshold_example[2], threshold_example[1]))
    plt.title(f"Threshold\n{threshold_example[0]}")
    plt.axis('off')

    plt.subplot(1, 6, 5)
    plt.imshow(create_overlay(images[image_index], edge_example[2], edge_example[1]))
    plt.title(f"Edge\n{edge_example[0]}")
    plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.imshow(create_overlay(images[image_index], region_example[2], region_example[1]))
    plt.title(f"Region\n{region_example[0]}")
    plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
    
print("\n=== EVALUASI KOMPREHENSIF SEGMENTASI CITRA ===")

print_metric_table("Tabel Metrik Thresholding", threshold_metric_rows)
print_metric_table("Tabel Metrik Edge Detection", edge_metric_rows)
print_metric_table("Tabel Metrik Region-Based", region_metric_rows)

print_timing_table("Tabel Waktu Komputasi Thresholding", threshold_timing_rows)
print_timing_table("Tabel Waktu Komputasi Edge Detection", edge_timing_rows)
print_timing_table("Tabel Waktu Komputasi Region-Based", region_timing_rows)

print_robustness_table("Tabel Robustness Thresholding", threshold_robustness_rows)
print_robustness_table("Tabel Robustness Edge Detection", edge_robustness_rows)
print_robustness_table("Tabel Robustness Region-Based", region_robustness_rows)