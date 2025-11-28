import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import os

from dataset import load_datasets
from model import build_model

# ---------------------------------------------
#  SEEDING for reproducibility
# ---------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(42)


def train_stage2():

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    # ------- data -------
    train_ds, val_ds, class_to_idx = load_datasets()
    print("Class mapping:", class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

    # ------- class weights -------
    labels = [y for _, y in train_ds]
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    print("Class weights:", weights)

    # ------- model: full fine-tuning -------
    model = build_model(num_classes=2, freeze_backbone=False).to(device)

    # load Stage 1 checkpoint (frozen backbone model)
    state_dict = torch.load("best_model_stage1.pth", map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded weights from best_model_stage1.pth")

    # loss + optimizer (smaller LR for fine-tuning)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # tracking
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0

    num_epochs = 10   # you can set 5–10; 10 is safe

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        print(f"[Stage 2] Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}")

        # ---------- validation ----------
        model.eval()
        correct = 0
        total = 0
        running_val_loss = 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        val_acc = correct / total
        val_accuracies.append(val_acc)

        print(f"[Stage 2] Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_stage2.pth")
            print("Saved new best model (Stage 2).")

    print("Stage 2 training complete.")

    # ------------- plots -------------
    os.makedirs("plots", exist_ok=True)

    # Loss
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss (Stage 2 - Fine-tuning)")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/aptos_stage2_loss.png")
    plt.close()

    # Val accuracy
    plt.figure(figsize=(8,6))
    plt.plot(val_accuracies, label="Validation Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy (Stage 2)")
    plt.grid(True)
    plt.savefig("plots/aptos_stage2_acc.png")
    plt.close()

    print("Stage 2 plots saved in: plots/aptos_stage2_loss.png and plots/aptos_stage2_acc.png")


if __name__ == "__main__":
    train_stage2()
