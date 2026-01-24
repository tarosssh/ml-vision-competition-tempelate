import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import DRDataset
from model import DRModel
from utils import get_transforms, load_model_from_checkpoint


def evaluate(shuffle_labels=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Frozen preprocessing (NOT manual, reused from utils)
    val_transform = get_transforms(train=False)

    val_dataset = DRDataset(
        root_dir="data/val",
        transform=val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    # ✅ Model loading (rule-compliant)
    model = DRModel()
    model = load_model_from_checkpoint(model)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if shuffle_labels:
        print("⚠️ Shuffling labels for sanity check")
        rng = np.random.default_rng(42)
        rng.shuffle(all_labels)

    auc = roc_auc_score(all_labels, all_preds)
    print(f"✅ Validation AUC: {auc:.6f}")

    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle-labels", action="store_true")
    args = parser.parse_args()

    evaluate(shuffle_labels=args.shuffle_labels)
