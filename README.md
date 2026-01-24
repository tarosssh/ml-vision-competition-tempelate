# Vision Classification Competition Template

## Overview
Reusable PyTorch template for image-based classification challenges.
Supports:
- Multi-seed training
- Calibration (temperature + threshold)
- Plug-and-play inference
- CPU-safe deployment

## Folder Structure
code/ → inference + model
training/ → training & evaluation
save/ → trained weights
data/ → dataset

## Training
```bash
python training/train.py
```

## Evaluation
```bash
python training/evaluate.py
```
## Inference
```bash
import inference
preds = inference.predict(batch)
```

**Notes**
- code/inference.py follows automated evaluation contracts
- No manual preprocessing required


