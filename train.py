import numpy as np
from PIL import Image
import cv2
import os
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def extract_features(image_path):
    try:
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

        # Feature 2: Blur Score
        blur_score = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()

        # Feature 3: Color ratio R/B
        r = img[:, :, 0].mean()
        b = img[:, :, 2].mean()
        color_ratio = r / (b + 1e-6)

        # Feature 4: Edge density
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edge_density = edges.mean()

        # Feature 5: Brightness std dev
        brightness_std = gray.std()

        # Feature 6: Saturation mean & std
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        saturation = img_hsv[:, :, 1]
        sat_mean = saturation.mean()
        sat_std = saturation.std()

        # Feature 7: Center vs edge brightness ratio
        center = gray[128:384, 128:384].mean()
        edge_region = np.concatenate([
            gray[:64, :].flatten(),
            gray[-64:, :].flatten(),
            gray[:, :64].flatten(),
            gray[:, -64:].flatten()
        ])
        edge_brightness = edge_region.mean()
        center_edge_ratio = center / (edge_brightness + 1e-6)

        # Feature 8: DCT energy in high frequency blocks
        gray_uint8 = gray.astype(np.uint8)
        dct_scores = []
        for i in range(0, 512, 64):
            for j in range(0, 512, 64):
                block = gray_uint8[i:i+64, j:j+64].astype(np.float32)
                dct = cv2.dct(block)
                high_freq = np.abs(dct[32:, 32:]).mean()
                low_freq = np.abs(dct[:32, :32]).mean()
                dct_scores.append(high_freq / (low_freq + 1e-6))
        dct_ratio = np.mean(dct_scores)

        # Feature 9: Noise level estimate
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.abs(gray - blur).mean()

        return [
            fft_spikiness, blur_score, color_ratio,
            edge_density, brightness_std, sat_mean,
            sat_std, center_edge_ratio, dct_ratio, noise
        ]
    except:
        return None

# Load all data
X, y = [], []

real_folder = "dataset/real"
screen_folder = "dataset/screen"

print("Loading real photos...")
for fname in os.listdir(real_folder):
    if fname.lower().endswith(('.jpg', '.jpeg')):
        feats = extract_features(os.path.join(real_folder, fname))
        if feats:
            X.append(feats)
            y.append(0)  # 0 = real

print("Loading screen photos...")
for fname in os.listdir(screen_folder):
    if fname.lower().endswith(('.jpg', '.jpeg')):
        feats = extract_features(os.path.join(screen_folder, fname))
        if feats:
            X.append(feats)
            y.append(1)  # 1 = screen

X = np.array(X)
y = np.array(y)
print(f"\nDataset: {X.shape[0]} images, {X.shape[1]} features")
print(f"Real: {sum(y==0)}, Screen: {sum(y==1)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train with cross-validation
clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
print(f"\nCross-validation accuracy: {scores}")
print(f"Mean accuracy: {scores.mean()*100:.1f}%")
print(f"Std: {scores.std()*100:.1f}%")


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Find misclassified images
all_files = []
for fname in os.listdir(real_folder):
    if fname.lower().endswith(('.jpg', '.jpeg')):
        all_files.append((os.path.join(real_folder, fname), 0))
for fname in os.listdir(screen_folder):
    if fname.lower().endswith(('.jpg', '.jpeg')):
        all_files.append((os.path.join(screen_folder, fname), 1))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
misclassified = []

for train_idx, test_idx in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    for i, (pred, true) in enumerate(zip(preds, y_test)):
        if pred != true:
            fname, label = all_files[test_idx[i]]
            misclassified.append((fname, true, pred))

print(f"\nMisclassified images ({len(misclassified)} total):")
for fname, true, pred in misclassified:
    true_label = "REAL" if true == 0 else "SCREEN"
    pred_label = "REAL" if pred == 0 else "SCREEN"
    print(f"  {os.path.basename(fname)}: TRUE={true_label}, PREDICTED={pred_label}")