# train.py
import os
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

from dataset import DRDataset
from model import DRModel
from utils import get_transforms, set_seed

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 8
BATCH_SIZE = 8
LR = 1e-4
HARD_MINING_START = 4   # epoch index (1-based)
POS_WEIGHT = 0.7       # penalize false positives
SAVE_DIR = "save"

set_seed(42)
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------
# Data
# -----------------------
train_ds = DRDataset(
    root_dir="data/train",
    transform=get_transforms(train=True)
)

val_ds = DRDataset(
    root_dir="data/val",
    transform=get_transforms(train=False)
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -----------------------
# Model
# -----------------------
model = DRModel(num_thresholds=2).to(DEVICE)

pos_weight = torch.tensor([POS_WEIGHT]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

best_f1 = 0.0

# -----------------------
# Training Loop
# -----------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # -----------------------
    # Validation
    # -----------------------
    model.eval()
    preds, targets, probs_all = [], [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            probs = torch.sigmoid(logits).mean(dim=1)

            preds.extend((probs > 0.5).cpu().numpy())
            probs_all.extend(probs.cpu().numpy())
            targets.extend(y[:, 1].cpu().numpy())

    f1 = f1_score(targets, preds)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Loss: {total_loss/len(train_loader):.4f} | "
        f"Val F1: {f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
        print("💾 Best model saved")

    # -----------------------
# Hard Negative Mining (ONCE)
# -----------------------
if epoch == HARD_MINING_START:
    gc.collect()
    torch.cuda.empty_cache()

    hard_indices = []
    idx = 0

    for x, y in val_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.no_grad():
            probs = torch.sigmoid(model(x)).mean(dim=1)

        for p, t in zip(probs.cpu(), y[:, 1].cpu()):
            if p > 0.5 and t == 0:  # false positive
                hard_indices.append(idx)
            idx += 1

    MAX_HARD = 32
    hard_indices = hard_indices[:MAX_HARD]

    if len(hard_indices) > 0:
        print(f"🔥 Hard mining on {len(hard_indices)} samples")

        hard_ds = Subset(val_ds, hard_indices)
        hard_loader = DataLoader(
            hard_ds,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )

        model.train()
        for x, y in hard_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    gc.collect()
    torch.cuda.empty_cache()

print("🏁 Training complete | Best Val F1:", best_f1)
