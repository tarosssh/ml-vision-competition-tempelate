# code/inference.py
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from model import build_model

# =======================
# PATHS (JUDGE SAFE)
# =======================
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights.pth")

# =======================
# GLOBALS (LAZY LOADED)
# =======================
_MODEL = None
_TRANSFORM = None
DEVICE = torch.device("cpu")

IMG_SIZE = 224
ORIG_SIZE = 512
TEMPERATURE = 1.03
THRESHOLD = 0.63

# =======================
# LOADERS
# =======================
def _build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def _load_model():
    global _MODEL, _TRANSFORM

    if _MODEL is not None:
        return

    _MODEL = build_model()
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)

    # handle packed ensemble OR single checkpoint
    if isinstance(state, dict) and "seed42" in state:
        _MODEL.load_state_dict(state["seed42"])
    else:
        _MODEL.load_state_dict(state)

    _MODEL.to(DEVICE)
    _MODEL.eval()
    _TRANSFORM = _build_transform()

# =======================
# INPUT HANDLING
# =======================
def _decode_image_vector(vec):
    arr = np.fromstring(vec, sep=",", dtype=np.uint8)
    arr = arr.reshape(ORIG_SIZE, ORIG_SIZE, 3)
    return Image.fromarray(arr, mode="RGB")

def _prepare_batch(batch):
    images = []

    # Case 1: pandas DataFrame (official judge format)
    if isinstance(batch, pd.DataFrame):
        for _, row in batch.iterrows():
            img = _decode_image_vector(row["image_vector"])
            images.append(_TRANSFORM(img))
        return torch.stack(images)

    # Case 2: numpy batch (N,H,W,C)
    if isinstance(batch, np.ndarray) and batch.ndim == 4:
        batch = list(batch)

    # Case 3: single item
    if not isinstance(batch, (list, tuple)):
        batch = [batch]

    for item in batch:
        if isinstance(item, Image.Image):
            img = item.convert("RGB")
        else:
            item = np.asarray(item)
            if item.dtype != np.uint8:
                item = (item * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(item).convert("RGB")

        images.append(_TRANSFORM(img))

    return torch.stack(images)

# =======================
# REQUIRED ENTRY POINT
# =======================
def predict(batch):
    """
    Input:
        batch: list | numpy.ndarray | pandas.DataFrame
    Output:
        numpy.ndarray of shape (N,) with values {0,1}
    """
    _load_model()
    inputs = _prepare_batch(batch).to(DEVICE)

    with torch.no_grad():
        logits = _MODEL(inputs) / TEMPERATURE
        probs = torch.sigmoid(logits)

        # collapse binary-2 → binary-1 safely
        if probs.ndim == 2 and probs.shape[1] == 2:
            probs = probs[:, 1]
        else:
            probs = probs.squeeze(1)

    # 🔒 CRITICAL: return INT (not bool)
    return (probs > THRESHOLD).long().cpu().numpy()
