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


def train():

    # device --------
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    # load datasets -----
    train_ds, val_ds, class_to_idx = load_datasets()
    print("Class mapping:", class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

    # compute class weights -----
    labels = [y for _, y in train_ds]
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    print("Class weights:", weights)

    # model -----------------
    model = build_model(num_classes=2, freeze_backbone=True).to(device)

    # loss + optimizer -------
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # tracking lists for graphs
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0

    for epoch in range(10):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}")

        # validation ----------
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

        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_stage1.pth")
            print("Saved new best model (Stage 1).")

    print("Training complete.")

    # ---------------------------------------------
    # PLOT GRAPHS
    # ---------------------------------------------
    os.makedirs("plots", exist_ok=True)

    # Train vs Val Loss
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss (Stage 1 - Transfer Learning)")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/aptos_stage1_loss.png")
    plt.close()

    # Validation Accuracy
    plt.figure(figsize=(8,6))
    plt.plot(val_accuracies, label="Validation Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy (Stage 1)")
    plt.grid(True)
    plt.savefig("plots/aptos_stage1_acc.png")
    plt.close()

    print("Plots saved in:  plots/aptos_stage1_loss.png  and  plots/aptos_stage1_acc.png")


if __name__ == "__main__":
    train()
