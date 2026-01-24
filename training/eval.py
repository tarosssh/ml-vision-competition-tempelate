# evaluate.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from dataset import DRDataset
from model import DRModel
from utils import get_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

# Dataset
test_ds = DRDataset(
    root_dir="data/val",  # 👈 use val instead of test
    transform=get_transforms(train=False)
)

test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

# Model
model = DRModel(num_thresholds=2).to(DEVICE)
model.load_state_dict(torch.load("save/best_model.pth", map_location=DEVICE))
model.eval()

preds, targets = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        probs = torch.sigmoid(logits).mean(dim=1)

        preds.extend((probs > 0.5).cpu().numpy())
        targets.extend(y[:, 1].numpy())

print("\n📊 Classification Report:")
print(classification_report(targets, preds, digits=4))

print("🧩 Confusion Matrix:")
print(confusion_matrix(targets, preds))
