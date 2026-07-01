import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Settings
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
DATASET_DIR = "dataset"

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=42
)

# Load validation data
val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=42
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Classes: {train_generator.class_indices}")

# Build MobileNetV2 model
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model first
base_model.trainable = False

# Add our classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel built successfully!")
print(f"Total parameters: {model.count_params():,}")

# Train
print("\nTraining...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    verbose=1
)

# Results
val_acc = max(history.history['val_accuracy'])
print(f"\nBest validation accuracy: {val_acc*100:.1f}%")

# Save model
# Save model
model.save("model_mobilenet.h5")
print("Model saved as model_mobilenet.h5")

# Fine-tuning phase — unfreeze last 30 layers
print("\nFine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train again
history2 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    verbose=1
)

val_acc2 = max(history2.history['val_accuracy'])
print(f"\nBest fine-tuned validation accuracy: {val_acc2*100:.1f}%")

# Save final model
model.save("model_mobilenet_finetuned.h5")
print("Fine-tuned model saved!")