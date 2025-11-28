# Diabetic Retinopathy Classification using ResNet50 (APTOS → ODIR Cross-Domain Study)

This project builds an end-to-end deep learning pipeline for **diabetic retinopathy (DR) detection** using fundus images.  
The model is trained on the **APTOS 2019 Blindness Detection** dataset and evaluated on the **ODIR-5K** dataset to study **cross-domain generalization**, a core challenge in medical AI.

The project includes:

- APTOS preprocessing and training  
- ODIR cleaning + soft & strong preprocessing  
- Transfer learning with ResNet50  
- Fine-tuning  
- Cross-dataset evaluation  
- Grad-CAM interpretability  
- Confusion matrix, ROC, AUC  
- Training/validation curves  

This work sets the foundation for a future comparative study using EfficientNet and ConvNeXt.

---

## 📁 Project Structure

```
diabetic-retinopathy-resnet/
│
├── src/
│   ├── train.py                 # Train ResNet50 on APTOS
│   ├── train_finetune.py        # Finetuning model 
│   ├── model.py                 # Model builder (ResNet50)
│   ├── transforms.py            # Train/val augmentations
│   ├── dataset.py               # Load APTOS datasets
│   ├── preprocess_aptos.py      # Aptos preprocessing
│   ├── aptos_dataset.py         # CSV-based APTOS loader
│   ├── preprocess_odir.py       # ODIR soft & strong preprocessing
│   ├── test_ODIR.py             # Evaluate APTOS model on ODIR
│   ├── gradcam.py               # Grad-CAM utils
└── README.md
```

---

## 🔧 Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Supports **Apple Silicon (MPS)** acceleration.

---

## 🧹 APTOS Preprocessing

APTOS images undergo:

- border removal  
- circular crop (optional)  
- resize to 224×224  
- normalization  
- train/val CSV split  

Located under:

```
data/APTOS_cropped/
train_split.csv
val_split.csv
```

---

## 🏋🏻 Training on APTOS

Run:

```bash
python src/train.py
```

### Stage 1 — Transfer Learning  
- Freeze ResNet50 backbone  
- Train FC head  
- LR = 1e-4  
→ `best_model_stage1.pth`

### Stage 2 — Fine-Tuning  
- Unfreeze backbone  
- LR = 1e-5  
→ `best_model_stage2.pth`

Training curves saved in `plots/`.

---

## 🧼 ODIR Preprocessing (Soft & Strong)

ODIR images differ from APTOS, so preprocessing improves domain adaptation.

### Soft
- border removal  
- resize  
- light autocontrast  

### Strong
- border removal  
- circular crop  
- aggressive autocontrast  
- sharpening  

Run:

```bash
python src/test_ODIR.py
```

Creates:

```
data/ODIR_preproc/soft/{DR,Normal}
data/ODIR_preproc/strong/{DR,Normal}
```

Corrupted images are skipped automatically.

---

## 🧪 Cross-Domain Evaluation (APTOS → ODIR)

Run:

```bash
python src/test_ODIR.py
```

Outputs:
- classification report  
- confusion matrix  
- ROC curve + AUC  
- Grad-CAM heatmaps  

### Example baseline (before preprocessing improvements)

| Metric | Score |
|--------|-------|
| Accuracy | 0.63 |
| AUC | 0.593 |
| DR Recall | 0.33 |
| Normal Recall | 0.83 |

Shows strong **domain gap** between APTOS and ODIR.

---

## 🔥 Grad-CAM

Grad-CAM is generated for:
- APTOS validation images  
- ODIR soft-processed  
- ODIR strong-processed  
- Misclassified images  


Helps visualize whether the model attends to:
- optic disc  
- macula  
- vessels  
- microaneurysms  
- artifacts  

---

## ✔ Completed

- APTOS preprocessing  
- Clean ODIR dataset  
- Soft & strong preprocessing  
- ResNet50 training  
- Fine-tuning  
- Training curves  
- ODIR evaluation  
- Grad-CAM  
- Confusion matrix & ROC  

---

## 🚀 Next Steps

### Planned:
- EfficientNet-B0 training  
- ConvNeXt-Tiny training  
- Full 3-model comparison  
- Ensemble fusion  
- Publication-style report  

---

## 👩‍⚕️ Author

**Sanjukta Biswas**  
Biomedical Engineering  
Deep Learning for Medical Imaging  
