# Diabetic Retinopathy Detection (Binary Classification)

## 1. Model Overview

**Model Name:** DrishtiNet-DR  
**Task Type:** Binary classification (DR vs No-DR)  
**Framework:** PyTorch  
**Framework Version:** torch==2.0.1  

The model predicts whether a retinal fundus image shows **diabetic retinopathy (DR)** or **no diabetic retinopathy**.

---

## 2. Input / Output Specification

### Input

Accepted input formats:

- `(H, W)`
- `(H, W, 1)`
- `(H, W, 3)`
- `(N, H, W, 3)`
- PIL Image
- NumPy array
- pandas DataFrame (with `image_vector` column)

**Dtype:** `uint8` or `float32`

The inference pipeline automatically:
- Converts grayscale to RGB
- Converts float inputs to uint8 safely
- Handles missing batch dimensions
- Resizes images internally to `224 × 224 × 3`
- Applies ImageNet normalization

---

### Output

- **Shape:** `(N, 1)`
- **Type:** `bool`
- **Meaning:**
  - `True`  → Diabetic Retinopathy detected
  - `False` → No Diabetic Retinopathy detected

Internally, predictions are produced using a sigmoid-activated probability and a calibrated decision threshold.

---

## 3. How to Replicate Predictions (Mandatory)

### Step 1: Extract submission

```bash
tar -xvzf model.tar.gz
cd drishti_submission
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run inference

```bash
from code import inference
from PIL import Image

img = Image.open("raw_data/test_images/sample.png")
pred = inference.predict([img])

print(pred)
```

## 4. Training Summary

- **Dataset Used:**
Diabetic Retinopathy fundus image dataset

- **Label Mapping:**
| Label | Meaning |
| ----- | ------- |
| 0     | No DR   |
| 1     | DR      |

- **Loss Function:** Binary Cross-Entropy with Logits
- **Optimizer:** AdamW
- **Backbone:** EfficientNet-B0 (ImageNet pretrained)


## 5. Team Details

**Team Number:** b93b15
**Team Name:** Drishti-AI CareGrid