import numpy as np
from PIL import Image
import cv2

def extract_features(image_path):
    # Load image
    img_pil = Image.open(image_path).convert("RGB")
    img_pil = img_pil.resize((512, 512))
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)

    # Feature 1: FFT Spikiness
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    ring = (dist > 40) & (dist < 150)
    ring_vals = magnitude[ring]
    fft_spikiness = ring_vals.max() / (ring_vals.mean() + 1e-6)

    # Feature 2: Blur Score (Laplacian variance - lower = blurrier)
    blur_score = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()

    # Feature 3: Color channel ratio (R/B - screens tend to be cooler/bluer)
    r = img[:, :, 0].mean()
    b = img[:, :, 2].mean()
    color_ratio = r / (b + 1e-6)

    # Feature 4: Edge density
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edge_density = edges.mean()

    # Feature 5: Brightness std dev
    brightness_std = gray.std()

    return [fft_spikiness, blur_score, color_ratio, edge_density, brightness_std]


# Quick test
import os

real_folder = "dataset/real"
screen_folder = "dataset/screen"

print("REAL photos:")
for fname in os.listdir(real_folder)[:3]:
    if fname.lower().endswith(('.jpg', '.jpeg')):
        feats = extract_features(os.path.join(real_folder, fname))
        print(f"  {fname}: {[round(f,2) for f in feats]}")

print("\nSCREEN photos:")
for fname in os.listdir(screen_folder)[:3]:
    if fname.lower().endswith(('.jpg', '.jpeg')):
        feats = extract_features(os.path.join(screen_folder, fname))
        print(f"  {fname}: {[round(f,2) for f in feats]}")