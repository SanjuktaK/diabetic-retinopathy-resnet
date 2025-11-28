import argparse
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn.functional as F

# ----------------------------
# Imports from this project
# ----------------------------
try:
    from .model import build_model
    from .transforms import get_val_transform
except ImportError:
    # fallback if run as: python src/gradcam.py
    from model import build_model
    from transforms import get_val_transform


# ----------------------------
# Device helper
# ----------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------
# Grad-CAM implementation
# ----------------------------
class GradCAM:
    """
    Simple GradCAM for ResNet-like models.
    Hooks into the given target_layer (e.g., model.layer4).
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # forward hook: save activations
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # backward hook: save gradients
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self.bwd_handle = self.target_layer.register_backward_hook(backward_hook)

    def __del__(self):
        # clean up hooks (not strictly necessary, but nice)
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: (1, C, H, W)
        class_idx: which class to backprop for.
                   If None, uses predicted class.
        returns: CAM heatmap (H, W), normalized to [0, 1]
        """
        # forward pass
        output = self.model(input_tensor)  # (1, num_classes)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # make one-hot
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0

        # backward pass
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        # [N, C, H, W]
        grads = self.gradients  # dY/dA
        activations = self.activations

        # global average pool the gradients: [C]
        weights = grads.mean(dim=(2, 3))[0]  # (C,)

        # weighted sum of activations
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]

        cam = F.relu(cam)

        # normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        cam_np = cam.detach().cpu().numpy()
        return cam_np


# ----------------------------
# Visualization utilities
# ----------------------------
def overlay_cam_on_image(img_pil, cam, alpha=0.4, colormap_name="jet"):
    """
    img_pil: original PIL image (RGB)
    cam: numpy array (H, W) with values [0, 1]
    returns: PIL.Image with heatmap overlay
    """
    img = np.array(img_pil).astype(np.float32)

    # resize cam to image size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8))
    cam_img = cam_img.resize((img.shape[1], img.shape[0]), Image.BILINEAR)
    cam_np = np.array(cam_img).astype(np.float32) / 255.0

    # apply colormap
    cmap = cm.get_cmap(colormap_name)
    cam_color = cmap(cam_np)[..., :3]  # drop alpha, shape (H, W, 3)
    cam_color = (cam_color * 255.0).astype(np.float32)

    # overlay
    overlay = alpha * cam_color + (1 - alpha) * img
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay)

    return overlay_pil


def show_and_save_gradcam(original_img, heatmap_img, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_img)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved Grad-CAM visualization to: {output_path}")


# ----------------------------
# Main CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for DR classifier")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input fundus image (PNG/JPG).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best_model_stage2.pth",
        help="Path to model weights (.pth).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of output classes of the model (e.g., 2 for DR vs Normal, 5 for severity grading).",
    )
    parser.add_argument(
        "--class-idx",
        type=int,
        default=None,
        help="Class index to visualize. If None, uses model's predicted class.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/gradcam_output.png",
        help="Path to save Grad-CAM visualization.",
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # ----------------------------
    # Load model
    # ----------------------------
    model = build_model(num_classes=args.num_classes, freeze_backbone=False)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded weights from: {args.weights}")

    # For ResNet-like model, target last conv block (layer4)
    target_layer = model.layer4
    cam_extractor = GradCAM(model, target_layer)

    # ----------------------------
    # Load and preprocess image
    # ----------------------------
    img_pil = Image.open(args.image).convert("RGB")
    transform = get_val_transform()
    img_tensor = transform(img_pil).unsqueeze(0).to(device)  # (1, C, H, W)

    # ----------------------------
    # Generate Grad-CAM
    # ----------------------------
    cam = cam_extractor.generate(img_tensor, class_idx=args.class_idx)

    overlay = overlay_cam_on_image(img_pil, cam, alpha=0.4, colormap_name="jet")
    show_and_save_gradcam(img_pil, overlay, args.output)


if __name__ == "__main__":
    main()
