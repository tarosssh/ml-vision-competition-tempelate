import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from dataset import DRDataset
from model import DRModel
from utils import get_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = DRModel(num_thresholds=2).to(DEVICE)
model.load_state_dict(torch.load("save/best_model.pth"))
model.eval()

val_ds = DRDataset(
    root_dir="data/val",
    transform=get_transforms(train=False)
)

probs, targets = [], []

with torch.no_grad():
    for x, y in val_ds:
        x = x.unsqueeze(0).to(DEVICE)
        logit = model(x)
        prob = torch.sigmoid(logit).mean().item()

        probs.append(prob)
        targets.append(y[1].item())

probs = np.array(probs)
targets = np.array(targets)

# ROC
fpr, tpr, _ = roc_curve(targets, probs)
roc_auc = auc(fpr, tpr)

# PR
precision, recall, _ = precision_recall_curve(targets, probs)
pr_auc = auc(recall, precision)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(recall, precision, label=f"AUC={pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve")
plt.legend()

plt.tight_layout()
plt.show()
