import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Load model
model = tf.keras.models.load_model("model_mobilenet_finetuned.h5")

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    score = model.predict(arr, verbose=0)[0][0]
    return float(score)

# Test on real images
print("Testing REAL images (expect score close to 0.00):")
real_correct = 0
real_folder = "test_data/real"
for fname in os.listdir(real_folder):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        score = predict(os.path.join(real_folder, fname))
        result = "✅" if score < 0.5 else "❌"
        print(f"  {result} {fname}: {score:.2f}")
        if score < 0.5:
            real_correct += 1

# Test on screen images
print("\nTesting SCREEN images (expect score close to 1.00):")
screen_correct = 0
screen_folder = "test_data/screen"
for fname in os.listdir(screen_folder):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        score = predict(os.path.join(screen_folder, fname))
        result = "✅" if score >= 0.5 else "❌"
        print(f"  {result} {fname}: {score:.2f}")
        if score >= 0.5:
            screen_correct += 1

# Final accuracy
total = real_correct + screen_correct
total_images = len([f for f in os.listdir(real_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]) + \
               len([f for f in os.listdir(screen_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])

print(f"\n{'='*40}")
print(f"Real:   {real_correct}/{len(os.listdir(real_folder))} correct")
print(f"Screen: {screen_correct}/{len(os.listdir(screen_folder))} correct")
print(f"Overall Accuracy: {total}/{total_images} = {total/total_images*100:.1f}%")