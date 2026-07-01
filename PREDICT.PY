import sys
import time
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model once
model = tf.keras.models.load_model("model_mobilenet_finetuned.h5")

def predict(image_path: str) -> float:
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # add batch dimension

    # Predict
    score = model.predict(arr, verbose=0)[0][0]
    return float(score)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Measure latency
    start = time.time()
    score = predict(image_path)
    end = time.time()

    latency_ms = (end - start) * 1000

    print(f"{score:.2f}")
    print(f"(Latency: {latency_ms:.1f} ms)")