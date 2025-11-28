import pandas as pd
from sklearn.model_selection import train_test_split

meta_path = "data/APTOS_cropped/metadata.csv"

df = pd.read_csv(meta_path)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_df.to_csv("data/APTOS_cropped/train_split.csv", index=False)
val_df.to_csv("data/APTOS_cropped/val_split.csv", index=False)

print("Done.")
print("Train:", len(train_df))
print("Val:", len(val_df))
