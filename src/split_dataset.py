import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

META_PATH = "data/ODIR_clean/metadata.csv"   # your uploaded file
IMG_BASE = "data/ODIR_cropped"         # use cropped images
OUT_BASE = "data/ODIR_split"

def make_dirs():
    for split in ["train", "val"]:
        for cls in ["Normal", "DR"]:
            os.makedirs(os.path.join(OUT_BASE, split, cls), exist_ok=True)

def main():
    df = pd.read_csv(META_PATH)

    # Only Normal and DR
    df = df[df["label"].isin(["Normal", "DR"])]

    # Stratified split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    make_dirs()

    # Copy images
    def copy_files(sub_df, split):
        for _, row in sub_df.iterrows():
            filename = row["filename"]
            label = row["label"]

            src = os.path.join(IMG_BASE, label, filename)
            dst = os.path.join(OUT_BASE, split, label, filename)

            if os.path.exists(src):
                shutil.copy(src, dst)

    print("Copying train samples...")
    copy_files(train_df, "train")

    print("Copying val samples...")
    copy_files(val_df, "val")

    print("Done!")
    print(f"Train images: {len(train_df)}")
    print(f"Val images:   {len(val_df)}")

if __name__ == "__main__":
    main()
