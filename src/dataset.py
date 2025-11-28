from aptos_dataset import APTOSDataset

try:
    from .transforms import get_train_transform, get_val_transform
except ImportError:
    from transforms import get_train_transform, get_val_transform

def load_datasets():
    train_csv = "data/APTOS_cropped/train_split.csv"
    val_csv   = "data/APTOS_cropped/val_split.csv"
    img_root  = "data/APTOS_cropped"

    train_ds = APTOSDataset(train_csv, img_root, transform=get_train_transform())
    val_ds   = APTOSDataset(val_csv,   img_root, transform=get_val_transform())

    return train_ds, val_ds, {"DR":0, "Normal":1}
