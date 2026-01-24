# visualize_cam.py
import torch
import cv2
import numpy as np
from PIL import Image
import os

from model import DRModel
from gradcam import GradCAM
from gradcam_pp import GradCAMPlusPlus
from utils import get_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# CAM ENTROPY
# --------------------------------------------------
def cam_entropy(cam):
    cam = cam - cam.min()
    cam = cam / (cam.sum() + 1e-8)
    cam = cam.flatten()
    return -np.sum(cam * np.log(cam + 1e-8))


# --------------------------------------------------
# Find pseudo-healthy image (leakage-safe)
# --------------------------------------------------
def find_healthy_image(val_dir, model):
    transform = get_transforms(train=False)

    for fname in os.listdir(val_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img = Image.open(os.path.join(val_dir, fname)).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prob = torch.sigmoid(model(input_tensor))[0, 0].item()

        if prob < 0.4:
            return img, fname, prob

    return None, None, None


# --------------------------------------------------
# Vessel Mask
# --------------------------------------------------
def vessel_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(gray)

    vessels = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )
    return cv2.medianBlur(vessels, 5)


# --------------------------------------------------
# Simple augmentation
# --------------------------------------------------
def simple_augment(img):
    out = img.copy()
    if np.random.rand() > 0.5:
        out = cv2.flip(out, 1)
    beta = np.random.randint(-10, 10)
    return cv2.convertScaleAbs(out, alpha=1.0, beta=beta)


# --------------------------------------------------
# Generate CAM
# --------------------------------------------------
def generate_cam(image_pil, cam_generator, model):
    transform = get_transforms(train=False)
    input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(input_tensor))[0, 0].item()
        pred_class = int(prob > 0.5)

    cam = cam_generator.generate(input_tensor)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    return cam, pred_class, prob


# --------------------------------------------------
# Load model
# --------------------------------------------------
model = DRModel()
model.load_state_dict(torch.load("save/best_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

target_layer = model.model.features[-3]
cam_gc = GradCAM(model, target_layer)
cam_gcpp = GradCAMPlusPlus(model, target_layer)


# --------------------------------------------------
# MAIN IMAGE
# --------------------------------------------------
img_path = r"C:\Users\Dell\OneDrive\Desktop\drishti_model\data\val\dr\0c7e82daf5a0.png"
image = Image.open(img_path).convert("RGB")
orig = np.array(image.resize((224, 224)))

cam_gcpp_map, pred_class, prob = generate_cam(image, cam_gcpp, model)
display_conf = min(prob, 0.99)

cam_gcpp_map[vessel_mask(orig) > 0] = 0
lesion_mask = cam_gcpp_map >= np.percentile(cam_gcpp_map, 80)
lesion_percent = 100.0 * lesion_mask.sum() / cam_gcpp_map.size

entropy = cam_entropy(cam_gcpp_map)
uncertainty = 0.5 * entropy

print(f"CAM Entropy (disease): {entropy:.4f}")
print(f"Uncertainty ± {uncertainty:.3f}")

heatmap = cv2.applyColorMap(np.uint8(255 * cam_gcpp_map), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

cv2.putText(overlay, f"Class: {pred_class}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(overlay, f"Conf: {display_conf:.2f}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(overlay, f"Lesion: {lesion_percent:.2f}%", (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.putText(overlay, f"H={entropy:.3f} ±{uncertainty:.3f}",
            (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

cv2.imwrite("gradcam_side_by_side.png",
             np.hstack([orig, heatmap, overlay]))


# --------------------------------------------------
# HEALTHY CAM
# --------------------------------------------------
val_dir = r"C:\Users\Dell\OneDrive\Desktop\drishti_model\data\val\dr"
healthy_img, healthy_name, healthy_prob = find_healthy_image(val_dir, model)

if healthy_img:
    healthy_cam, _, _ = generate_cam(healthy_img, cam_gcpp, model)
    healthy_entropy = cam_entropy(healthy_cam)
    print(f"Healthy CAM Entropy: {healthy_entropy:.4f}")

    healthy_orig = np.array(healthy_img.resize((224, 224)))
    healthy_overlay = cv2.addWeighted(
        healthy_orig, 0.7,
        cv2.applyColorMap(np.uint8(255 * healthy_cam), cv2.COLORMAP_JET),
        0.3, 0
    )
    cv2.imwrite("healthy_cam.png", healthy_overlay)


# --------------------------------------------------
# CAM STABILITY
# --------------------------------------------------
aug_np = simple_augment(orig)
aug_img = Image.fromarray(aug_np)

cam_aug, _, _ = generate_cam(aug_img, cam_gcpp, model)
entropy_aug = cam_entropy(cam_aug)

print(f"Augmented CAM Entropy: {entropy_aug:.4f}")
print(f"Entropy Drop: {abs(entropy - entropy_aug):.4f}")

cv2.imwrite("cam_stability.png", np.hstack([
    cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0),
    cv2.addWeighted(aug_np, 0.6,
                    cv2.applyColorMap(np.uint8(255 * cam_aug), cv2.COLORMAP_JET),
                    0.4, 0)
]))


# --------------------------------------------------
# GRAD-CAM vs GRAD-CAM++
# --------------------------------------------------
cam_gc_map, _, _ = generate_cam(image, cam_gc, model)
cam_gc_map[vessel_mask(orig) > 0] = 0

overlay_gc = cv2.addWeighted(
    orig, 0.6,
    cv2.applyColorMap(np.uint8(255 * cam_gc_map), cv2.COLORMAP_JET),
    0.4, 0
)

overlay_gcpp = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

cv2.putText(overlay_gc, "Grad-CAM", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(overlay_gcpp, "Grad-CAM++", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

cv2.imwrite("cam_vs_campp.png",
            np.hstack([orig, overlay_gc, overlay_gcpp]))

print("✅ Saved all CAM visualizations")
