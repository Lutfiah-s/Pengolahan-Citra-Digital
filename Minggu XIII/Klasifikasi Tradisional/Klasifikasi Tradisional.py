# =========================================================
# KOMPARASI KLASIFIKASI KNN vs SVM
# PENGENALAN OBJEK CITRA FASHION-MNIST

# =========================================================
# IMPORT LIBRARY
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import warnings

warnings.filterwarnings('ignore')

from tensorflow.keras.datasets import fashion_mnist

from skimage.feature import hog, local_binary_pattern

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    GridSearchCV
)

from sklearn.preprocessing import (
    StandardScaler,
    label_binarize
)

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# =========================================================
# LOAD DATASET
# =========================================================

print("=" * 60)
print("LOAD DATASET")
print("=" * 60)

(X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()

# Gabungkan dataset
X = np.concatenate((X_train_full, X_test_full), axis=0)
y = np.concatenate((y_train_full, y_test_full), axis=0)

# Ambil 1000 sample
np.random.seed(42)

indices = np.random.choice(len(X), 1000, replace=False)

X = X[indices]
y = y[indices]

print("Jumlah Data :", X.shape)
print("Jumlah Label:", y.shape)

# =========================================================
# NAMA KELAS
# =========================================================

class_names = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

# =========================================================
# VISUALISASI DATASET
# =========================================================

print("\nMenampilkan contoh dataset...")

plt.figure(figsize=(12, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(X[i], cmap='gray')

    plt.title(class_names[y[i]])

    plt.axis('off')

plt.suptitle("Sample Fashion-MNIST Dataset")

plt.tight_layout()

plt.show()

# =========================================================
# EKSTRAKSI FITUR
# HOG + LBP
# =========================================================

print("\n" + "=" * 60)
print("EKSTRAKSI FITUR")
print("=" * 60)

def extract_features(images):

    feature_list = []

    for image in images:

        # -------------------------
        # HOG FEATURE
        # -------------------------

        hog_feature = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )

        # -------------------------
        # LBP FEATURE
        # -------------------------

        lbp = local_binary_pattern(
            image,
            P=8,
            R=1,
            method='uniform'
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, 11),
            range=(0, 10)
        )

        hist = hist.astype("float")

        hist /= (hist.sum() + 1e-6)

        # -------------------------
        # COMBINE FEATURE
        # -------------------------

        combined_feature = np.hstack([
            hog_feature,
            hist
        ])

        feature_list.append(combined_feature)

    return np.array(feature_list)

print("Proses ekstraksi fitur...")

X_features = extract_features(X)

print("Shape fitur:", X_features.shape)

# =========================================================
# SPLIT DATA
# =========================================================

print("\n" + "=" * 60)
print("SPLIT DATA")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training Data :", X_train.shape)
print("Testing Data  :", X_test.shape)

# =========================================================
# STANDARDISASI
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# =========================================================
# KNN EXPERIMENT
# =========================================================

print("\n" + "=" * 60)
print("KNN EXPERIMENT")
print("=" * 60)

k_values = [1, 3, 5, 7, 9, 11]

distance_metrics = [
    'euclidean',
    'manhattan',
    'minkowski'
]

knn_results = []

best_knn_model = None
best_knn_accuracy = 0

for metric in distance_metrics:

    for k in k_values:

        print(f"KNN -> k={k}, metric={metric}")

        knn_model = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric
        )

        # Training Time
        start_train = time.time()

        knn_model.fit(X_train, y_train)

        train_time = time.time() - start_train

        # Inference Time
        start_test = time.time()

        y_pred = knn_model.predict(X_test)

        inference_time = time.time() - start_test

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average='weighted'
        )

        recall = recall_score(
            y_test,
            y_pred,
            average='weighted'
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average='weighted'
        )

        knn_results.append([
            k,
            metric,
            accuracy,
            precision,
            recall,
            f1,
            train_time,
            inference_time
        ])

        # Best Model
        if accuracy > best_knn_accuracy:

            best_knn_accuracy = accuracy

            best_knn_model = knn_model

# DataFrame
knn_df = pd.DataFrame(
    knn_results,
    columns=[
        'K',
        'Metric',
        'Accuracy',
        'Precision',
        'Recall',
        'F1-Score',
        'Train Time',
        'Inference Time'
    ]
)

print("\nKNN RESULT")
print(knn_df)

# =========================================================
# VISUALISASI KNN
# =========================================================

plt.figure(figsize=(10, 6))

for metric in distance_metrics:

    subset = knn_df[knn_df['Metric'] == metric]

    plt.plot(
        subset['K'],
        subset['Accuracy'],
        marker='o',
        label=metric
    )

plt.title("KNN Accuracy vs K")

plt.xlabel("K")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.show()

# =========================================================
# SVM EXPERIMENT
# =========================================================

print("\n" + "=" * 60)
print("SVM EXPERIMENT")
print("=" * 60)

kernels = ['linear', 'poly', 'rbf']

C_values = [0.1, 1, 10, 100]

svm_results = []

best_svm_model = None
best_svm_accuracy = 0

for kernel in kernels:

    for C in C_values:

        print(f"SVM -> kernel={kernel}, C={C}")

        if kernel == 'rbf':
            gamma = 0.01
        else:
            gamma = 'scale'

        svm_model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True
        )

        # Training Time
        start_train = time.time()

        svm_model.fit(X_train, y_train)

        train_time = time.time() - start_train

        # Inference Time
        start_test = time.time()

        y_pred = svm_model.predict(X_test)

        inference_time = time.time() - start_test

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average='weighted'
        )

        recall = recall_score(
            y_test,
            y_pred,
            average='weighted'
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average='weighted'
        )

        svm_results.append([
            kernel,
            C,
            accuracy,
            precision,
            recall,
            f1,
            train_time,
            inference_time
        ])

        # Best Model
        if accuracy > best_svm_accuracy:

            best_svm_accuracy = accuracy

            best_svm_model = svm_model

# DataFrame
svm_df = pd.DataFrame(
    svm_results,
    columns=[
        'Kernel',
        'C',
        'Accuracy',
        'Precision',
        'Recall',
        'F1-Score',
        'Train Time',
        'Inference Time'
    ]
)

print("\nSVM RESULT")
print(svm_df)

# =========================================================
# HASIL TERBAIK
# =========================================================

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best KNN Accuracy :", best_knn_accuracy)

print("Best SVM Accuracy :", best_svm_accuracy)

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

best_predictions = best_svm_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix - Best SVM")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nCLASSIFICATION REPORT\n")

print(classification_report(
    y_test,
    best_predictions,
    target_names=class_names
))

# =========================================================
# ROC CURVE & AUC
# =========================================================

print("\n" + "=" * 60)
print("ROC CURVE & AUC")
print("=" * 60)

# Binarize Label
y_test_bin = label_binarize(
    y_test,
    classes=np.arange(10)
)

# Predict Probability
y_score = best_svm_model.predict_proba(X_test)

# ROC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(10):

    fpr[i], tpr[i], _ = roc_curve(
        y_test_bin[:, i],
        y_score[:, i]
    )

    roc_auc[i] = auc(
        fpr[i],
        tpr[i]
    )

# Plot
plt.figure(figsize=(10, 8))

for i in range(10):

    plt.plot(
        fpr[i],
        tpr[i],
        label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})"
    )

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Best SVM")

plt.legend()

plt.grid()

plt.show()

# =========================================================
# PCA DECISION BOUNDARY
# =========================================================

print("\n" + "=" * 60)
print("DECISION BOUNDARY PCA")
print("=" * 60)

# PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_features)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train SVM PCA
svm_pca = SVC(kernel='rbf')

svm_pca.fit(X_train_pca, y_train_pca)

# Meshgrid
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1

y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.5),
    np.arange(y_min, y_max, 0.5)
)

Z = svm_pca.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 8))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3
)

scatter = plt.scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    c=y_test_pca,
    cmap='tab10'
)

plt.title("Decision Boundary SVM PCA 2D")

plt.xlabel("PCA 1")

plt.ylabel("PCA 2")

plt.colorbar(scatter)

plt.show()

# =========================================================
# LEARNING CURVE
# =========================================================

print("\n" + "=" * 60)
print("LEARNING CURVE")
print("=" * 60)

train_sizes, train_scores, test_scores = learning_curve(
    best_svm_model,
    X_features,
    y,
    cv=5,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = train_scores.mean(axis=1)

test_mean = test_scores.mean(axis=1)

# Plot
plt.figure(figsize=(10, 6))

plt.plot(
    train_sizes,
    train_mean,
    marker='o',
    label='Training Accuracy'
)

plt.plot(
    train_sizes,
    test_mean,
    marker='o',
    label='Validation Accuracy'
)

plt.title("Learning Curve")

plt.xlabel("Training Size")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.show()

# =========================================================
# STRATIFIED K-FOLD CROSS VALIDATION
# =========================================================

print("\n" + "=" * 60)
print("STRATIFIED K-FOLD")
print("=" * 60)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    best_svm_model,
    X_features,
    y,
    cv=skf,
    scoring='accuracy'
)

print("Cross Validation Scores :")

print(cv_scores)

print("\nAverage CV Accuracy :", cv_scores.mean())

# =========================================================
# GRID SEARCH CV
# =========================================================

print("\n" + "=" * 60)
print("GRID SEARCH CV")
print("=" * 60)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

print("Best Parameters :")

print(grid_search.best_params_)

print("\nBest Accuracy :")

print(grid_search.best_score_)

# =========================================================
# TABEL PERBANDINGAN
# =========================================================

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print("\nTOP KNN RESULT")

print(
    knn_df.sort_values(
        by='Accuracy',
        ascending=False
    ).head()
)

print("\nTOP SVM RESULT")

print(
    svm_df.sort_values(
        by='Accuracy',
        ascending=False
    ).head()
)

# =========================================================
# VISUALISASI TRAINING TIME
# =========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    ['Best KNN', 'Best SVM'],
    [
        knn_df['Train Time'].min(),
        svm_df['Train Time'].min()
    ]
)

plt.title("Training Time Comparison")

plt.ylabel("Seconds")

plt.show()

# =========================================================
# VISUALISASI INFERENCE TIME
# =========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    ['Best KNN', 'Best SVM'],
    [
        knn_df['Inference Time'].min(),
        svm_df['Inference Time'].min()
    ]
)

plt.title("Inference Time Comparison")

plt.ylabel("Seconds")

plt.show()

# =========================================================
# SAVE RESULT TABLE
# =========================================================

knn_df.to_csv("knn_results.csv", index=False)

svm_df.to_csv("svm_results.csv", index=False)

print("\nFile hasil berhasil disimpan:")
print("- knn_results.csv")
print("- svm_results.csv")

# =========================================================
# KESIMPULAN OTOMATIS
# =========================================================

print("\n" + "=" * 60)
print("KESIMPULAN")
print("=" * 60)

if best_svm_accuracy > best_knn_accuracy:

    print("""
SVM memberikan performa terbaik dibandingkan KNN
pada dataset Fashion-MNIST.

Kernel RBF mampu menangkap pola non-linear
dengan lebih baik sehingga menghasilkan akurasi
lebih tinggi.

KNN memiliki training time lebih cepat,
namun inference time cenderung lebih lambat
karena seluruh data training digunakan saat prediksi.
""")

else:

    print("""
KNN memberikan performa terbaik pada dataset ini.

Nilai K optimal menghasilkan generalisasi
yang baik terhadap data testing.
""")

print("\nPROGRAM SELESAI")