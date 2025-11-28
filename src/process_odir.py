import os
import pandas as pd
import shutil

# Paths
BASE = "data/ODIR-5K"
IMG_DIR = os.path.join(BASE, "Training Images")
df = pd.read_excel(os.path.join(BASE, "data.xlsx"))

# Output folders
OUT = "data/ODIR_clean"
NORMAL_DIR = os.path.join(OUT, "Normal")
DR_DIR = os.path.join(OUT, "DR")

os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(DR_DIR, exist_ok=True)

# --- Helper functions ----------------------------------------------------

def is_clean_normal(row):
    """
    Strict NORMAL criteria:
    1. N == 1
    2. All other diseases == 0
    3. BOTH eyes must contain 'normal fundus'
    4. BOTH eyes must NOT contain 'lens dust' or 'low image quality'
    """
    # Binary label rule
    if row["N"] != 1:
        return False
    for disease in ["D", "G", "C", "A", "H", "M", "O"]:
        if row[disease] != 0:
            return False

    left_kw = str(row["Left-Diagnostic Keywords"]).lower()
    right_kw = str(row["Right-Diagnostic Keywords"]).lower()

    # Must say "normal fundus" in BOTH eyes
    if "normal fundus" not in left_kw:
        return False
    if "normal fundus" not in right_kw:
        return False

    # Reject dust & low-quality images
    bad_terms = ["lens dust", "low image quality"]

    if any(b in left_kw for b in bad_terms):
        return False
    if any(b in right_kw for b in bad_terms):
        return False

    return True


def keyword_is_dr(text):
    """Check if diagnostic keyword contains diabetic retinopathy terms."""
    if not isinstance(text, str):
        return False
    text = text.lower()
    dr_terms = [
        "retinopathy",
        "npdr",
        "pdr",
        "non proliferative",
        "proliferative"
    ]
    return any(t in text for t in dr_terms)


def is_clean_dr(row):
    """
    Strict DR criteria:
    1. D == 1
    2. BOTH eyes contain retinopathy keywords
    """
    if row["D"] != 1:
        return False

    left_kw = row["Left-Diagnostic Keywords"]
    right_kw = row["Right-Diagnostic Keywords"]

    return keyword_is_dr(left_kw) and keyword_is_dr(right_kw)


# --- Main extraction -----------------------------------------------------

records = []
saved_normal = 0
saved_dr = 0

for _, row in df.iterrows():
    left_img = row["Left-Fundus"]
    right_img = row["Right-Fundus"]

    # ------- NORMAL CASE -------
    if is_clean_normal(row):
        for img in [left_img, right_img]:
            src = os.path.join(IMG_DIR, img)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(NORMAL_DIR, img))
                records.append([img, "Normal"])
                saved_normal += 1
        continue

    # ------- DR CASE -------
    if is_clean_dr(row):
        for img in [left_img, right_img]:
            src = os.path.join(IMG_DIR, img)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(DR_DIR, img))
                records.append([img, "DR"])
                saved_dr += 1
        continue

# Save metadata
meta_df = pd.DataFrame(records, columns=["filename", "label"])
meta_df.to_csv(os.path.join(OUT, "metadata.csv"), index=False)

print("Processing complete!")
print("Normal images:", saved_normal)
print("DR images:", saved_dr)
