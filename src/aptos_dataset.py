import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class APTOSDataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

        # Normal -> 1, DR -> 0
        self.label_map = {"Normal": 1, "DR": 0}
        self.df["label"] = self.df["label"].map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filename = row["filename"]
        label = int(row["label"])

        subfolder = "Normal" if label == 1 else "DR"
        img_path = os.path.join(self.img_root, subfolder, filename)

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
