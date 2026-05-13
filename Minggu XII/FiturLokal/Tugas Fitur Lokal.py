import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# =====================================================
# CONFIG
# =====================================================

DATASET_PATH = 'dataset'

CLASSES = ['botol', 'mug', 'buku', 'remote', 'mainan']

TEST_IMAGES = [
    'test_rot.jpg',
    'test_scale.jpg',
    'test_dark.jpg',
    'test_occ.jpg'
]

VOCAB_SIZES = [10, 20, 50, 100]
# Akan disesuaikan otomatis dengan ukuran dataset
PCA_COMPONENTS = [2, 4, 8, 16, 32, 64, 128]

# =====================================================
# FEATURE EXTRACTOR
# =====================================================

def get_extractor(method):

    if method == 'SIFT':
        return cv2.SIFT_create()

    elif method == 'ORB':
        return cv2.ORB_create(nfeatures=500)

    elif method == 'SURF':
        try:
            return cv2.xfeatures2d.SURF_create()
        except:
            return None

# =====================================================
# LOAD IMAGE
# =====================================================

def load_gray(path):

    img = cv2.imread(path)

    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(gray, extractor):

    start = time.time()

    kp, des = extractor.detectAndCompute(gray, None)

    elapsed = time.time() - start

    return kp, des, elapsed

# =====================================================
# VISUALIZATION
# =====================================================

def draw_keypoints_image(img, kp):

    result = cv2.drawKeypoints(
        img,
        kp,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result

# =====================================================
# BF MATCHING
# =====================================================

def bf_matching(des1, des2, method):

    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

    if method == 'ORB':
        norm = cv2.NORM_HAMMING
    else:
        norm = cv2.NORM_L2

    bf = cv2.BFMatcher(norm)

    matches = bf.knnMatch(des1, des2, k=2)

    good = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    return good

# =====================================================
# FLANN MATCHING
# =====================================================

def flann_matching(des1, des2, method):

    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

    good = []

    try:

        if method == 'ORB':

            FLANN_INDEX_LSH = 6

            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,
                key_size=12,
                multi_probe_level=1
            )

            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(
                index_params,
                search_params
            )

            des1 = np.asarray(des1, np.uint8)
            des2 = np.asarray(des2, np.uint8)

        else:

            FLANN_INDEX_KDTREE = 1

            index_params = dict(
                algorithm=FLANN_INDEX_KDTREE,
                trees=5
            )

            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(
                index_params,
                search_params
            )

            des1 = np.asarray(des1, np.float32)
            des2 = np.asarray(des2, np.float32)

        matches = flann.knnMatch(des1, des2, k=2)

        for pair in matches:

            if len(pair) != 2:
                continue

            m, n = pair

            if m.distance < 0.75 * n.distance:
                good.append(m)

    except Exception as e:

        print(f'FLANN Error ({method}):', e)

    return good

# =====================================================
# HOMOGRAPHY + RANSAC
# =====================================================

def compute_homography(kp1, kp2, matches):

    if len(matches) < 4:
        return None, 0, None

    try:

        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in matches]
        ).reshape(-1,1,2)

        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in matches]
        ).reshape(-1,1,2)

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )

        if mask is None:
            return H, 0, None

        inliers = np.sum(mask)

        return H, int(inliers), mask

    except:
        return None, 0, None

# =====================================================
# MATCH VISUALIZATION
# =====================================================

def draw_match_visualization(img1, kp1, img2, kp2, matches, mask, title):

    if mask is not None:
        matchesMask = mask.ravel().tolist()
    else:
        matchesMask = None

    result = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        matchesMask=matchesMask,
        flags=2
    )

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12,6))
    plt.imshow(result)
    plt.title(title)
    plt.axis('off')
    plt.show()

# =====================================================
# MAIN PROCESS
# =====================================================

results = []
all_descriptors = []
all_labels = []
method_visuals = {}

methods = ['SIFT', 'ORB', 'SURF']

for method in methods:

    extractor = get_extractor(method)

    if extractor is None:
        print(f'\n{method} not available')
        continue

    print(f'\n========== {method} ==========')

    method_visuals[method] = []

    for cls in CLASSES:

        ref_path = os.path.join(
            DATASET_PATH,
            cls,
            'ref.jpg'
        )

        ref_img, ref_gray = load_gray(ref_path)

        if ref_img is None:
            continue

        kp1, des1, t1 = extract_features(
            ref_gray,
            extractor
        )

        if des1 is None:
            continue

        print(f'\nClass: {cls}')
        print(f'Keypoints: {len(kp1)}')
        print(f'Descriptor Shape: {des1.shape}')
        print(f'Extraction Time: {t1:.4f} sec')

        vis_img = draw_keypoints_image(ref_img, kp1)
        method_visuals[method].append((cls, vis_img))

        if method == 'SIFT':

            all_descriptors.append(des1)
            all_labels.append(cls)

            for test_name in TEST_IMAGES:

                test_path = os.path.join(
                    DATASET_PATH,
                    cls,
                    test_name
                )

                _, test_gray = load_gray(test_path)

                if test_gray is None:
                    continue

                _, des_test, _ = extract_features(
                    test_gray,
                    extractor
                )

                if des_test is not None:
                    all_descriptors.append(des_test)
                    all_labels.append(cls)

        for test_name in TEST_IMAGES:

            test_path = os.path.join(
                DATASET_PATH,
                cls,
                test_name
            )

            test_img, test_gray = load_gray(test_path)

            if test_img is None:
                continue

            kp2, des2, t2 = extract_features(
                test_gray,
                extractor
            )

            if des2 is None:
                continue

            good_bf = bf_matching(des1, des2, method)
            good_flann = flann_matching(des1, des2, method)

            H, inliers, mask = compute_homography(
                kp1,
                kp2,
                good_bf
            )

            print(f'{test_name}')
            print(f'BF Matches: {len(good_bf)}')
            print(f'FLANN Matches: {len(good_flann)}')
            print(f'RANSAC Inliers: {inliers}')

            results.append([
                method,
                cls,
                test_name,
                len(kp1),
                len(kp2),
                len(good_bf),
                len(good_flann),
                inliers,
                t1,
                t2
            ])

            if (
                (method == 'SIFT' and cls == 'mug' and test_name == 'test_rot.jpg')
                or
                (method == 'ORB' and cls == 'mug' and test_name == 'test_rot.jpg')
            ):

                 draw_match_visualization(
                    ref_img,
                    kp1,
                    test_img,
                    kp2,
                    good_bf,
                    mask,
                    f'{method} Matching - {cls} - {test_name}'
                )

# =====================================================
# VISUALIZATION PER METHOD
# =====================================================

for method, visuals in method_visuals.items():

    n = len(visuals)

    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5*cols, 5*rows)
    )

    axes = np.array(axes).reshape(-1)

    for ax, (cls, img) in zip(axes, visuals):

        ax.imshow(img)
        ax.set_title(f'{method} - {cls}')
        ax.axis('off')

    for ax in axes[len(visuals):]:
        ax.remove()

    plt.suptitle(f'Keypoints Visualization - {method}')

    plt.tight_layout()
    plt.show()

# =====================================================
# RESULT TABLE
# =====================================================

df = pd.DataFrame(results, columns=[
    'Method',
    'Class',
    'Test Image',
    'Ref Keypoints',
    'Test Keypoints',
    'BF Matches',
    'FLANN Matches',
    'RANSAC Inliers',
    'Ref Time',
    'Test Time'
])

print(df)

df.to_csv('matching_results.csv', index=False)

# =====================================================
# ROBUSTNESS ANALYSIS
# =====================================================

summary = df.groupby('Method')[[
    'BF Matches',
    'FLANN Matches',
    'RANSAC Inliers',
    'Ref Time'
]].mean()

print('\n========== ROBUSTNESS SUMMARY ==========')
print(summary)

ax = summary.plot(
    kind='bar',
    figsize=(12,6),
    width=0.8
)

plt.title('Method Comparison')
plt.ylabel('Average Value')
plt.xticks(rotation=0)
plt.grid(axis='y')

# Tambahkan nilai di atas batang
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f')

plt.tight_layout()
plt.show()
# =====================================================
# BOVW + PCA + SVM
# =====================================================

bovw_results = []

for k in VOCAB_SIZES:

    print(f'\n========== VOCAB SIZE {k} ==========')

    descriptor_stack = np.vstack(all_descriptors)

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(descriptor_stack)

    histograms = []

    for des in all_descriptors:

        words = kmeans.predict(des)

        hist, _ = np.histogram(
            words,
            bins=np.arange(k+1)
        )

        histograms.append(hist)

    X = np.array(histograms)
    y = np.array(all_labels)

    for n_comp in PCA_COMPONENTS:

        # Pastikan PCA valid
        max_comp = min(X.shape[0]-1, X.shape[1])

        if n_comp > max_comp:
            continue

        print(f'PCA Components: {n_comp}')

        try:

            pca = PCA(n_components=n_comp)

            X_pca = pca.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_pca,
                y,
                test_size=0.3,
                random_state=42,
                stratify=y
            )

            clf = SVC(
                kernel='linear',
                probability=True
            )

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            print(f'Accuracy: {acc:.4f}')

        except Exception as e:

            print('PCA/SVM Error:', e)
            continue

        bovw_results.append([
            k,
            n_comp,
            acc
        ])

        if k == 20 and n_comp == 16:

            print('\nClassification Report:')
            print(
                classification_report(
                    y_test,
                    y_pred,
                    zero_division=0
                )
            )

            cm = confusion_matrix(y_test, y_pred)

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=clf.classes_
            )

            disp.plot(cmap='Blues')

            plt.title('Confusion Matrix')
            plt.show()

            # =========================================
            # PRECISION RECALL CURVE
            # =========================================

            try:

                y_bin = label_binarize(y_test, classes=CLASSES)

                clf_pr = OneVsRestClassifier(
                    SVC(kernel='linear', probability=True)
                )

                clf_pr.fit(
                    X_train,
                    label_binarize(y_train, classes=CLASSES)
                )

                y_score = clf_pr.predict_proba(X_test)

                plt.figure(figsize=(8,6))

                for i, cls in enumerate(CLASSES):

                    if i >= y_bin.shape[1]:
                        continue

                    # Skip jika class tidak ada
                    if np.sum(y_bin[:, i]) == 0:
                        continue

                    precision, recall, _ = precision_recall_curve(
                        y_bin[:, i],
                        y_score[:, i]
                    )

                    ap = average_precision_score(
                        y_bin[:, i],
                        y_score[:, i]
                    )

                    plt.plot(
                        recall,
                        precision,
                        label=f'{cls} AP={ap:.2f}'
                    )

                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                plt.grid()
                plt.show()

            except Exception as e:

                print('PR Curve Error:', e)

# =====================================================
# EVALUATION GRAPH
# =====================================================

bovw_df = pd.DataFrame(
    bovw_results,
    columns=['Vocabulary Size', 'PCA Components', 'Accuracy']
)

print('\n========== BOVW RESULT ==========' )
print(bovw_df)

# =====================================================
# VOCABULARY GRAPH
# =====================================================

plt.figure(figsize=(10,6))

for pca_comp in bovw_df['PCA Components'].unique():

    subset = bovw_df[
        bovw_df['PCA Components'] == pca_comp
    ]

    plt.scatter(
        subset['Vocabulary Size'],
        subset['Accuracy'] + np.random.uniform(-0.01, 0.01, len(subset)),
        s=120,
        label=f'PCA {pca_comp}'
    )

plt.title('Vocabulary Size vs Accuracy')
plt.xlabel('Vocabulary Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# =====================================================
# PCA GRAPH
# =====================================================

plt.figure(figsize=(10,6))

for vocab in bovw_df['Vocabulary Size'].unique():

    subset = bovw_df[
        bovw_df['Vocabulary Size'] == vocab
    ]

    plt.scatter(
        subset['PCA Components'],
        subset['Accuracy'] + np.random.uniform(-0.01, 0.01, len(subset)),
        s=120,
        label=f'Vocab {vocab}'
    )

plt.title('PCA Components vs Accuracy')
plt.xlabel('PCA Components')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# =====================================================
# MATCHING GRAPH
# =====================================================

plt.figure(figsize=(10,6))

for method in df['Method'].unique():

    subset = df[df['Method'] == method]

    plt.plot(
        subset['BF Matches'].values,
        label=method
    )

plt.title('BF Matching Comparison')
plt.xlabel('Sample')
plt.ylabel('Good Matches')
plt.legend()
plt.grid()
plt.show()

print('\nProgram selesai.')