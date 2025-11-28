import os
import cv2
import numpy as np
from tqdm import tqdm

INPUT_DIR = "data/ODIR_clean"
OUTPUT_DIR = "data/ODIR_cropped"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Normal"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "DR"), exist_ok=True)

def crop_circle(img):
    """Crop the circular fundus area."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # threshold to find bright circle
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img  # fallback: return original image

    # largest contour = retina
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # add small padding
    pad = int(0.03 * max(w, h))
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = w + 2 * pad
    h = h + 2 * pad

    cropped = img[y:y + h, x:x + w]

    return cropped


def process_class(cls):
    src_dir = os.path.join(INPUT_DIR, cls)
    dst_dir = os.path.join(OUTPUT_DIR, cls)

    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for fname in tqdm(files, desc=f"Cropping {cls}"):
        img_path = os.path.join(src_dir, fname)
        img = cv2.imread(img_path)

        if img is None:
            continue

        cropped = crop_circle(img)

        # resize to 256x256
        cropped = cv2.resize(cropped, (256, 256))

        out_path = os.path.join(dst_dir, fname)
        cv2.imwrite(out_path, cropped)


process_class("Normal")
process_class("DR")

print("ROI cropping complete!")
