# ==========================================
# FINAL CNN + TRANSFER LEARNING CIFAR-10
# GOOGLE COLAB VERSION
# ==========================================

# ==========================================
# IMPORT LIBRARY
# ==========================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# ==========================================
# LOAD CIFAR-10
# ==========================================
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

class_names = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# NORMALIZE
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# ONE HOT
y_train_cat = keras.utils.to_categorical(y_train,10)
y_test_cat = keras.utils.to_categorical(y_test,10)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# ==========================================
# VISUALIZE DATASET
# ==========================================
plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis("off")

plt.suptitle("Sample CIFAR-10")
plt.show()

# ==========================================
# DATA AUGMENTATION
# ==========================================
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

# VISUALIZE AUGMENTATION
sample = X_train[0].reshape((1,32,32,3))

plt.figure(figsize=(10,5))

i = 0
for batch in datagen.flow(sample, batch_size=1):

    plt.subplot(2,5,i+1)
    plt.imshow(batch[0])
    plt.axis('off')

    i += 1
    if i == 10:
        break

plt.suptitle("Data Augmentation")
plt.show()

# ==========================================
# CNN FROM SCRATCH
# ==========================================
def build_cnn():

    model = keras.Sequential([

        layers.Conv2D(
            32,(3,3),
            activation='relu',
            padding='same',
            input_shape=(32,32,3)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(
            64,(3,3),
            activation='relu',
            padding='same'
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(
            128,(3,3),
            activation='relu',
            padding='same'
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),

        layers.Dense(256,activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(10,activation='softmax')
    ])

    return model

cnn_model = build_cnn()

cnn_model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    )
]

print("\nTRAINING CNN SCRATCH")

history_cnn = cnn_model.fit(
    datagen.flow(X_train,y_train_cat,batch_size=64),
    validation_data=(X_test,y_test_cat),
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# TRANSFER LEARNING
# ==========================================
print("\nTRAINING MOBILENETV2")

X_train_mobile = tf.image.resize(
    X_train,
    (96,96)
)

X_test_mobile = tf.image.resize(
    X_test,
    (96,96)
)

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96,96,3)
)

base_model.trainable = False

transfer_model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(10,activation='softmax')
])

transfer_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_transfer = transfer_model.fit(
    X_train_mobile,
    y_train_cat,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    verbose=1
)

# ==========================================
# EVALUATION FUNCTION
# ==========================================
def evaluate_model(model,X_test,y_test,y_name):

    pred_prob = model.predict(X_test)
    pred = np.argmax(pred_prob,axis=1)

    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(
        y_test,pred,average='weighted'
    )
    recall = recall_score(
        y_test,pred,average='weighted'
    )
    f1 = f1_score(
        y_test,pred,average='weighted'
    )

    print(f"\n===== {y_name} =====")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", f1)

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test,pred)

    plt.figure(figsize=(10,8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f'Confusion Matrix - {y_name}')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print(classification_report(
        y_test,
        pred,
        target_names=class_names
    ))

    # ROC AUC
    y_bin = label_binarize(
        y_test,
        classes=range(10)
    )

    plt.figure(figsize=(8,6))

    for i in range(10):

        fpr,tpr,_ = roc_curve(
            y_bin[:,i],
            pred_prob[:,i]
        )

        roc_auc = auc(fpr,tpr)

        plt.plot(
            fpr,
            tpr,
            label=f'{class_names[i]} AUC={roc_auc:.2f}'
        )

    plt.plot([0,1],[0,1],'k--')
    plt.title(f'ROC Curve - {y_name}')
    plt.legend()
    plt.show()

    return accuracy,precision,recall,f1

# ==========================================
# EVALUATION
# ==========================================
cnn_acc, cnn_prec, cnn_rec, cnn_f1 = evaluate_model(
    cnn_model,
    X_test,
    y_test.flatten(),
    "CNN Scratch"
)

mobile_acc, mobile_prec, mobile_rec, mobile_f1 = evaluate_model(
    transfer_model,
    X_test_mobile,
    y_test.flatten(),
    "MobileNetV2"
)

# ==========================================
# TRAINING CURVE
# ==========================================
def plot_history(history,title):

    fig,ax = plt.subplots(1,2,figsize=(12,5))

    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title(f'{title} Accuracy')
    ax[0].legend(['train','val'])

    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title(f'{title} Loss')
    ax[1].legend(['train','val'])

    plt.show()

plot_history(history_cnn,"CNN Scratch")
plot_history(history_transfer,"MobileNetV2")

# ==========================================
# FEATURE MAPS
# ==========================================
layer_outputs = [
    layer.output
    for layer in cnn_model.layers
    if 'conv2d' in layer.name
]

activation_model = keras.Model(
    inputs=cnn_model.inputs,
    outputs=layer_outputs
)

sample = X_test[0:1]

activations = activation_model.predict(sample)

plt.figure(figsize=(12,8))

for i in range(16):
    plt.subplot(4,4,i+1)

    plt.imshow(
        activations[0][0,:,:,i],
        cmap='viridis'
    )

    plt.axis('off')

plt.suptitle("Feature Maps")
plt.show()

# ==========================================
# TSNE VISUALIZATION
# ==========================================
feature_model = keras.Model(
    cnn_model.input,
    cnn_model.layers[-3].output
)

features = feature_model.predict(
    X_test[:1000]
)

tsne = TSNE(
    n_components=2,
    random_state=42
)

embedded = tsne.fit_transform(features)

plt.figure(figsize=(10,8))

scatter = plt.scatter(
    embedded[:,0],
    embedded[:,1],
    c=y_test[:1000].flatten()
)

plt.legend(
    handles=scatter.legend_elements()[0],
    labels=class_names,
    bbox_to_anchor=(1.05,1)
)

plt.title("t-SNE Feature Embedding")
plt.show()

# ==========================================
# COMPARISON TABLE
# ==========================================
comparison = pd.DataFrame({

    "Model":[
        "CNN Scratch",
        "MobileNetV2"
    ],

    "Accuracy":[
        cnn_acc,
        mobile_acc
    ],

    "Precision":[
        cnn_prec,
        mobile_prec
    ],

    "Recall":[
        cnn_rec,
        mobile_rec
    ],

    "F1-Score":[
        cnn_f1,
        mobile_f1
    ]
})

print("\nMODEL COMPARISON")
display(comparison)

print("\nSELESAI")