# Spot the Fake Photo 🔍
> Computer Vision Take-Home Assignment

## Problem Statement
People sometimes cheat in mobile apps by taking a photo of another screen — a phone or laptop showing a picture — instead of taking a real photo of the real thing.

This project builds a classifier that detects whether an image is:
- `0.00` = **Real Photo** (genuine camera capture)
- `1.00` = **Photo of a Screen** (recaptured / fake)

---

## Usage
```bash
python predict.py image.jpg
```

**Output:**
```
0.00   ← Real Photo
1.00   ← Photo of a Screen
```

---

## Approach

### Journey
I explored three approaches before arriving at the final solution:

| Approach | Accuracy | Notes |
|---|---|---|
| Classical CV (FFT + Hand-crafted features) + Random Forest | 75.9% | Good start, high variance |
| Classical CV (FFT + Hand-crafted features) + SVM | 88.7% | Better, but still short of target |
| MobileNetV2 (Transfer Learning + Fine-tuning) | **95.5%** ✅ | Target achieved |

### Final Approach: MobileNetV2 Transfer Learning
1. Used **MobileNetV2** pretrained on ImageNet as the base model
2. Added a custom classification head (GlobalAveragePooling → Dropout → Dense → Sigmoid)
3. **Phase 1** — Froze base model, trained classification head for 20 epochs → 90.9%
4. **Phase 2** — Unfroze last 30 layers, fine-tuned with low learning rate (0.0001) for 15 epochs → **95.5%**

### Why MobileNetV2?
- Lightweight and fast — designed for mobile/edge deployment
- Pretrained on millions of images — strong feature extraction out of the box
- Works well even with small datasets (117 images) via transfer learning

### Classical CV Features Explored (train.py)
Before switching to deep learning, I extracted and tested these hand-crafted features:
- FFT Spikiness (moiré pattern detection)
- Blur Score (Laplacian variance)
- Color Channel Ratio (R/B)
- Edge Density
- Brightness Std Dev
- Saturation Mean & Std
- Center vs Edge Brightness Ratio
- DCT Energy Ratio
- Noise Level Estimate

---

## Dataset
- **Real photos:** 58 images (personal photos taken with Android phone)
- **Screen photos:** 59 images (photos of laptop screen displaying images)
- **Total:** 117 images
- **Transfer method:** WhatsApp (both classes equally compressed for fair comparison)
- **Split:** 80% training / 20% validation

---

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | **95.5%** |
| Latency (per image) | **~660 ms** (laptop CPU) |
| Device | Windows laptop, CPU only |
| Cost per image (on-device) | ~$0.00 |
| Cost per 1,000 images (cloud) | ~$0.01 (estimate) |

### Latency Breakdown
- First run: ~1834 ms (model loading + inference)
- Subsequent runs: ~660 ms (inference only)
- For production: model should be loaded once at startup, not per image

---

## Cost Analysis

| Deployment | Cost |
|---|---|
| On-device (phone/laptop) | **~$0** — runs free on user's device |
| Cloud (AWS Lambda/GCP) | **~$0.01 per 1,000 images** (assuming ~660ms × $0.0000166/sec on a small instance) |

For a mobile app, the ideal deployment is **on-device** using TensorFlow Lite — this gives zero inference cost and works offline.

---

## Project Structure
```
assignment/
├── dataset/
│   ├── real/                        ← 58 real photos
│   └── screen/                      ← 59 screen photos
├── explore.py                       ← Data exploration & visualization
├── fft_test.py                      ← FFT signal testing
├── features.py                      ← Hand-crafted feature extraction
├── train.py                         ← Random Forest + SVM experiments
├── train_mobilenet.py               ← MobileNetV2 training script
├── model_mobilenet_finetuned.h5     ← Final trained model
├── predict.py                       ← Final predictor (main deliverable)
└── README.md                        ← This file
```

---

## How to Run

### Install Dependencies
```bash
pip install tensorflow pillow numpy
```

### Train the Model (optional — pretrained model included)
```bash
python train_mobilenet.py
```

### Run Prediction
```bash
python predict.py path/to/image.jpg
```

---

## What I'd Improve With More Time

1. **More data** — 500+ images per class would significantly improve generalization
2. **TensorFlow Lite conversion** — convert model to `.tflite` for true on-device mobile deployment (~50ms latency)
3. **Better data collection** — use USB transfer instead of WhatsApp to preserve full image quality and moiré patterns
4. **Adaptive threshold** — instead of fixed 0.5 cutoff, tune threshold based on false positive/negative tradeoff for the specific use case
5. **Adversarial robustness** — test against cheaters who might try to fool the detector (e.g., adding grain to screen photos)
6. **Data augmentation** — more aggressive augmentation (brightness, contrast, JPEG compression simulation) to handle diverse real-world conditions

---

## Author
Aryama Pandey
