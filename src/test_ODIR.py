import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Local project imports
from model import build_model
from transforms import get_val_transform

# Optional: GradCAM (if you want to use it later in this file)
try:
    from gradcam import GradCAM, overlay_cam_on_image
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False
    print("gradcam.py not found - GradCAM part will be skipped.")


# ============================================================
#  Config
# ============================================================

IN_ROOT = "data/ODIR_clean"          # expects DR/ and Normal/ inside
OUT_ROOT = "data/ODIR_preproc"       # will create soft/ and strong/ here

IMG_SIZE = 224
CLASSES = ["DR", "Normal"]           # IMPORTANT: must match training mapping { 'DR':0, 'Normal':1 }

WEIGHTS_PATH = "best_model_stage2.pth"   # APTOS-trained model


# ============================================================
#  Device helper
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
#  Preprocessing utilities
# ============================================================

def remove_black_borders(img_np, threshold=5):
    """
    Remove black borders by cropping to the bounding box of non-dark pixels.
    img_np: H x W x 3 (uint8)
    """
    gray = img_np.mean(axis=2)
    mask = gray > threshold

    if not mask.any():
        # if everything is dark, just return original
        return img_np

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img_np[y0:y1, x0:x1]
    return cropped


def circular_crop(img_np):
    """
    Apply a central circular mask, zero out everything outside.
    """
    h, w = img_np.shape[:2]
    cx, cy = w // 2, h // 2
    r = min(cx, cy)

    Y, X = np.ogrid[:h, :w]
    dist = (X - cx) ** 2 + (Y - cy) ** 2
    mask = dist <= r * r

    out = img_np.copy()
    out[~mask] = 0
    return out


def preprocess_soft(pil_img):
    """
    Soft, safer preprocessing:
      - remove black borders
      - resize to IMG_SIZE
      - light autocontrast
    """
    img_np = np.array(pil_img)

    # remove black borders
    img_np = remove_black_borders(img_np)

    # back to PIL
    img = Image.fromarray(img_np)

    # resize
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    # light auto-contrast
    img = ImageOps.autocontrast(img, cutoff=1)  # cutoff=1%

    return img


def preprocess_strong(pil_img):
    """
    Strong APTOS-like preprocessing:
      - remove black borders
      - circular crop
      - resize to IMG_SIZE
      - stronger autocontrast
      - slight sharpening
    """
    img_np = np.array(pil_img)

    # remove black borders
    img_np = remove_black_borders(img_np)

    # circular crop
    img_np = circular_crop(img_np)

    # back to PIL
    img = Image.fromarray(img_np)

    # resize
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    # stronger autocontrast
    img = ImageOps.autocontrast(img, cutoff=2)

    # slight sharpening
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    return img


def ensure_preproc_dirs():
    """
    Create:
      data/ODIR_preproc/soft/DR
      data/ODIR_preproc/soft/Normal
      data/ODIR_preproc/strong/DR
      data/ODIR_preproc/strong/Normal
    """
    for style in ["soft", "strong"]:
        for cls in CLASSES:
            out_dir = os.path.join(OUT_ROOT, style, cls)
            os.makedirs(out_dir, exist_ok=True)


def preprocess_odir():
    """
    Run both soft and strong preprocessing on all images in data/ODIR_clean.
    """
    ensure_preproc_dirs()

    for cls in CLASSES:
        in_dir = os.path.join(IN_ROOT, cls)
        if not os.path.isdir(in_dir):
            print(f"WARNING: {in_dir} does not exist, skipping.")
            continue

        files = sorted([
            f for f in os.listdir(in_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

        print(f"[Preprocess] {cls}: {len(files)} images")

        for fname in tqdm(files):
            in_path = os.path.join(in_dir, fname)
            try:
                img = Image.open(in_path).convert("RGB")
            except Exception as e:
                print(f"⚠️ Skipping corrupted image: {in_path} ({e})")
                continue

            # soft
            img_soft = preprocess_soft(img)
            out_soft = os.path.join(OUT_ROOT, "soft", cls, fname)
            img_soft.save(out_soft, quality=95)

            # strong
            img_strong = preprocess_strong(img)
            out_strong = os.path.join(OUT_ROOT, "strong", cls, fname)
            img_strong.save(out_strong, quality=95)

    print("Preprocessing complete.")
    print("Soft folder : data/ODIR_preproc/soft/DR, Normal")
    print("Strong folder: data/ODIR_preproc/strong/DR, Normal")


# ============================================================
#  Dataset for preprocessed ODIR
# ============================================================

class ODIRPreprocDataset(Dataset):
    """
    Expects:
        root_dir/DR/*.jpg
        root_dir/Normal/*.jpg
    Applies only the standard val_transform (normalize etc).
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        self.classes = CLASSES[:]  # enforce consistent order
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        print(f"Loading from: {root_dir}")
        print("Class to idx:", self.class_to_idx)

        exts = (".jpg", ".jpeg", ".png", ".bmp")
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                print(f"WARNING: {cls_path} missing, skipping class {cls}")
                continue
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(exts):
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls]))

        print(f"Total images in {root_dir}: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================
#  Metrics plotting
# ============================================================

def plot_confusion_matrix(cm, classes, save_path=None):
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("ODIR Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print("Saved confusion matrix →", save_path)
    plt.close()


def plot_roc(y_true, y_score, title_suffix="", save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ODIR ROC Curve {title_suffix}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print("Saved ROC curve →", save_path)
    plt.close()

    return roc_auc


# ============================================================
#  Evaluation on one style (soft or strong)
# ============================================================

def evaluate_style(style):
    """
    style: 'soft' or 'strong'
    Loads images from data/ODIR_preproc/{style}/DR, Normal
    Evaluates APTOS-trained model on them.
    """
    device = get_device()
    print(f"\n=== EVALUATING STYLE: {style.upper()} on device {device} ===")

    root_dir = os.path.join(OUT_ROOT, style)
    transform = get_val_transform()
    dataset = ODIRPreprocDataset(root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # load model
    model = build_model(num_classes=2, freeze_backbone=False)
    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_dr_scores = []   # probability for class 0 (DR)

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"ODIR {style}"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)
            dr_scores = probs[:, 0]   # class index 0 = DR

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_dr_scores.extend(dr_scores.cpu().numpy())

    target_names = CLASSES
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    cm = confusion_matrix(all_labels, all_preds)
    cm_path = f"plots/odir_confusion_matrix_{style}.png"
    plot_confusion_matrix(cm, target_names, save_path=cm_path)

    roc_path = f"plots/odir_roc_curve_{style}.png"
    auc_val = plot_roc(all_labels, all_dr_scores, title_suffix=f"({style})", save_path=roc_path)

    print(f"[{style}] Final AUC: {auc_val:.3f}")

    return auc_val


# ============================================================
#  (Optional) Grad-CAM on a few samples per style
# ============================================================

def run_gradcam_samples(style, num_samples=3):
    if not HAS_GRADCAM:
        print("GradCAM not available, skipping.")
        return

    device = get_device()
    print(f"\n=== Grad-CAM on style: {style} ===")

    root_dir = os.path.join(OUT_ROOT, style)
    transform = get_val_transform()
    dataset = ODIRPreprocDataset(root_dir, transform=transform)

    # load model
    model = build_model(num_classes=2, freeze_backbone=False)
    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # pick some indices
    idxs = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    target_layer = model.layer4
    cam = GradCAM(model, target_layer)

    for i, idx in enumerate(idxs):
        path, label = dataset.samples[idx]
        pil_img = Image.open(path).convert("RGB")

        img_t = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_t)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))

        cam_map = cam.generate(img_t, class_idx=pred_idx)
        overlay = overlay_cam_on_image(pil_img, cam_map, alpha=0.4, colormap_name="jet")

        out_path = f"plots/gradcam_{style}_{i}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        overlay.save(out_path)
        print(f"Saved Grad-CAM for {style} sample {i} → {out_path}")
        print(f"  True label: {CLASSES[label]}, Pred: {CLASSES[pred_idx]}, Probs: {probs}")


# ============================================================
#  Main
# ============================================================

def main():
    # 1. Preprocess (soft + strong)
    preprocess_odir()

    # 2. Evaluate soft
    auc_soft = evaluate_style("soft")

    # 3. Evaluate strong
    auc_strong = evaluate_style("strong")

    print("\n=== Summary AUCs ===")
    print(f"Soft   : {auc_soft:.3f}")
    print(f"Strong : {auc_strong:.3f}")

    # 4. Grad-CAM examples
    if HAS_GRADCAM:
        run_gradcam_samples("soft", num_samples=3)
        run_gradcam_samples("strong", num_samples=3)


if __name__ == "__main__":
    main()
