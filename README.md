# Spot the Fake Photo — Recapture Detection

Given one image, decide whether it's a **real photo** or a **photo of a screen/printout** ("recapture").

```bash
python predict.py image.jpg
# -> 0.93   (0 = real photo, 1 = photo of a screen)
```

---

## How I approached it

I collected ~100 photos on my phone — real objects/scenes vs. photos of a screen or printout showing an image — split into `dataset/real/` and `dataset/screen/`, with variety in lighting, angle, and screen type.

I then compared two approaches rather than committing to one blind:

**1. Traditional ML with hand-engineered features**
Extracted signals that are known tells of a recapture — color statistics, edge features, FFT (frequency-domain) features, and texture features — and fed them into classical classifiers:
- Random Forest
- Support Vector Machine (SVM)

**2. Deep learning via transfer learning**
Fine-tuned **MobileNetV2** (ImageNet-pretrained), unfreezing the top layers, with data augmentation and early stopping. Chosen specifically because it's small and fast enough to eventually run on a phone.

Both were evaluated with 5-fold cross-validation so the accuracy number isn't a lucky split.

---

## Results (honest numbers)

| Model | Evaluation | Accuracy |
|---|---|---|
| Random Forest | 5-fold CV | __ % |
| SVM | 5-fold CV | __ % |
| MobileNetV2 (fine-tuned) | 5-fold CV | **94%+** |

MobileNetV2 came out ahead, so that's the model behind `predict.py`.

---

## Latency & cost (required)

- **Latency:** __ ms per image, measured on **[device — e.g. laptop CPU]**
- **Cost per image:**
  - On-device: **$0** — inference runs locally on the user's phone
  - Cloud (if server-side): ~$__ per 1,000 images, assuming [instance type / throughput assumptions]

---

## What I'd improve with more time

- Grow the dataset beyond ~100 images and add more screen types (different phone models, monitors, e-ink, printouts under varying light) — the biggest lever for accuracy and robustness
- Quantize MobileNetV2 (int8, TFLite/Core ML export) to shrink it further for on-device deployment
- Track a moving cutoff threshold as more real fraud data comes in, rather than a fixed one — tuned on a precision-recall curve so false-accusations (blocking a genuine user) are weighted appropriately against missed recaptures
- Periodically refresh the training set as cheaters adapt (new screens, better printouts) — treat this as an ongoing arms race, not a one-time model

---

## Project structure

```
dataset/
    real/
    screen/
predict.py              # one-line predictor
train.py                 # trains Random Forest / SVM on hand-engineered features
train_mobilenet.py       # fine-tunes MobileNetV2
train_mobilenet_cv.py    # MobileNetV2 with 5-fold cross-validation
features.py              # feature extraction (color, edge, FFT, texture)
explore.py
fft_test.py
test_model.py
model_mobilenet.h5
model_mobilenet_finetuned.h5
comparison.png
README.md
```

---

## How to run

```bash
# install dependencies
pip install tensorflow scikit-learn opencv-python numpy matplotlib pillow

# train traditional ML models
python train.py

# train MobileNetV2
python train_mobilenet.py

# train MobileNetV2 with 5-fold cross-validation
python train_mobilenet_cv.py

# predict on a single image
python predict.py image.jpg

# evaluate on the test set
python test_model.py
```

---

## Author
**Aryama**
SalesCode AI — Computer Vision & Machine Learning Take-Home
