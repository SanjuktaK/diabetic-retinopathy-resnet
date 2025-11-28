import os
import cv2
import pandas as pd
from tqdm import tqdm

# YOUR APTOS PATH
RAW_DIR = "data/aptos2019-blindness-detection"
IMG_DIR = os.path.join(RAW_DIR, "train_images")
CSV_PATH = os.path.join(RAW_DIR, "train.csv")

# OUTPUT
OUT_BASE = "data/APTOS_cropped"
NORMAL_DIR = os.path.join(OUT_BASE, "Normal")
DR_DIR = os.path.join(OUT_BASE, "DR")

os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(DR_DIR, exist_ok=True)

def crop_circle(img):
    """
    Crop circular fundus region similar to ODIR cropping.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    return img[y:y+h, x:x+w]

def main():
    df = pd.read_csv(CSV_PATH)

    # Convert 0–4 → binary
    df["label"] = df["diagnosis"].apply(lambda x: "Normal" if x == 0 else "DR")

    meta = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing APTOS"):
        id_code = row["id_code"]
        label = row["label"]

        filename = id_code + ".png"
        img_path = os.path.join(IMG_DIR, filename)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        cropped = crop_circle(img)

        # resize to match ODIR preprocessing
        cropped = cv2.resize(cropped, (256, 256))

        out_dir = NORMAL_DIR if label == "Normal" else DR_DIR
        out_path = os.path.join(out_dir, filename)

        cv2.imwrite(out_path, cropped)
        meta.append([filename, label])

    # Save meta
    pd.DataFrame(meta, columns=["filename", "label"]).to_csv(
        os.path.join(OUT_BASE, "metadata.csv"), index=False
    )

    print("APTOS preprocessing complete!")
    print(f"Total images: {len(meta)}")

if __name__ == "__main__":
    main()
