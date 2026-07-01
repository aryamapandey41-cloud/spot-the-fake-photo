import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array,
    ImageDataGenerator
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Load image paths
# -----------------------------

real_folder = "dataset/real"
screen_folder = "dataset/screen"

X = []
y = []

for file in os.listdir(real_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        X.append(os.path.join(real_folder, file))
        y.append(0)

for file in os.listdir(screen_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        X.append(os.path.join(screen_folder, file))
        y.append(1)

X = np.array(X)
y = np.array(y)

print("Total Images:", len(X))
print("Real:", np.sum(y == 0))
print("Screen:", np.sum(y == 1))

# -----------------------------
# Image Loader
# -----------------------------

def load_images(paths):

    images = []

    for path in paths:

        img = load_img(path, target_size=(224, 224))

        img = img_to_array(img)

        img = preprocess_input(img)

        images.append(img)

    return np.array(images)


# -----------------------------
# MobileNetV2 Model
# -----------------------------

def create_model():

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Fine-tune last 30 layers
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)

    x = Dense(128, activation="relu")(x)

    x = Dropout(0.3)(x)

    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# Data Augmentation
# -----------------------------

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

# -----------------------------
# Early Stopping
# -----------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# -----------------------------
# 5 Fold Cross Validation
# -----------------------------

kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = []

fold = 1

for train_index, test_index in kf.split(X, y):

    print("\n==============================")
    print("Fold", fold)
    print("==============================")

    X_train = load_images(X[train_index])
    X_test = load_images(X[test_index])

    y_train = y[train_index]
    y_test = y[test_index]

    model = create_model()

    model.fit(
        datagen.flow(X_train, y_train, batch_size=8),
        epochs=15,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    predictions = model.predict(X_test)

    predictions = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    print("Fold Accuracy:", accuracy)

    scores.append(accuracy)

    fold += 1

# -----------------------------
# Final Results
# -----------------------------

print("\n==============================")
print("Cross Validation Results")
print("==============================")

for i, score in enumerate(scores):
    print(f"Fold {i+1}: {score*100:.2f}%")

print("\nMean Accuracy: {:.2f}%".format(np.mean(scores) * 100))
print("Standard Deviation: {:.2f}%".format(np.std(scores) * 100))

print("\nTraining Complete!")