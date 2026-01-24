# threshold_search.py
import torch
import numpy as np
from dataset import DRDataset
from model import DRModel
from utils import get_transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model = DRModel(num_thresholds=2).to(DEVICE)
    model.load_state_dict(torch.load("save/best_model.pth", map_location=DEVICE))
    model.eval()

    temp = torch.load("save/temperature.pt")["temperature"]

    val_ds = DRDataset(
        root_dir="data/val",
        transform=get_transforms(train=False)
    )
    loader = DataLoader(val_ds, batch_size=16, num_workers=0)

    probs_all, targets_all = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x) / temp
            probs = torch.sigmoid(logits).mean(dim=1)

            probs_all.extend(probs.cpu().numpy())
            targets_all.extend(y[:, 1].cpu().numpy())

    probs_all = np.array(probs_all)
    targets_all = np.array(targets_all)

    best_f1, best_t = 0, 0.5

    for t in np.linspace(0.3, 0.9, 121):
        preds = (probs_all >= t).astype(int)
        f1 = f1_score(targets_all, preds)

        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"✅ Best threshold: {best_t:.3f} | Val F1: {best_f1:.4f}")

    np.save("save/best_threshold.npy", best_t)


if __name__ == "__main__":
    main()
