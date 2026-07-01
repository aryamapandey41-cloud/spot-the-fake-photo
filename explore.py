import matplotlib.pyplot as plt
from PIL import Image
import os

real_folder = "dataset/real"
screen_folder = "dataset/screen"

real_images = os.listdir(real_folder)[:3]
screen_images = os.listdir(screen_folder)[:3]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, fname in enumerate(real_images):
    img = Image.open(os.path.join(real_folder, fname))
    axes[0, i].imshow(img)
    axes[0, i].set_title("REAL")
    axes[0, i].axis("off")

for i, fname in enumerate(screen_images):
    img = Image.open(os.path.join(screen_folder, fname))
    axes[1, i].imshow(img)
    axes[1, i].set_title("SCREEN")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("comparison.png")
print("Saved comparison.png - open it to view!")