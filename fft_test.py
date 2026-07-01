import numpy as np
from PIL import Image
import os

def fft_spikiness_score(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((512, 512))
    arr = np.array(img, dtype=np.float64)

    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)  # log scale - standard for FFT visualization

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Look at a ring of mid-frequencies
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    ring_mask = (dist > 40) & (dist < 150)

    ring_values = magnitude[ring_mask]

    # High spikiness = sharp peaks (likely screen/moiré)
    spikiness = ring_values.max() / (ring_values.mean() + 1e-6)
    return spikiness


# Quick test on a few images from each folder
real_folder = "dataset/real"
screen_folder = "dataset/screen"

print("REAL photos:")
for fname in os.listdir(real_folder)[:5]:
    score = fft_spikiness_score(os.path.join(real_folder, fname))
    print(f"  {fname}: {score:.2f}")

print("\nSCREEN photos:")
for fname in os.listdir(screen_folder)[:5]:
    score = fft_spikiness_score(os.path.join(screen_folder, fname))
    print(f"  {fname}: {score:.2f}")