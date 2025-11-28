# Experiments Log for Diabetic Retinopathy Project

This file tracks all experiments, preprocessing steps, training results, notes, and future plans for your DR classification pipeline.

---

## **Dataset Summary**
- Source: ODIR-5K
- Classes (strict filtering): Normal, DR
- Normal images: 2004
- DR images: 1384
- Total cleaned images: 3388
- ROI-cropped and resized to 256×256

---

## **Experiment 1 — Baseline (Pretrained ResNet50, All Layers Trainable)**
**Training Details:**
- Model: ResNet50 (ImageNet pretrained)
- All layers unfrozen
- Loss: Weighted CrossEntropy
- LR: 1e-4
- Batch size: 16
- Epochs: 10
- Device: Apple MPS

**Results:**
| Epoch | Train Loss | Val Acc |
|-------|------------|---------|
| 1 | 0.5820 | 0.7445 |
| 2 | 0.4086 | 0.7903 |
| 3 | 0.2457 | 0.7459 |
| 4 | 0.1492 | 0.7770 |
| 5 | 0.1127 | 0.7917 |
| 6 | 0.0833 | 0.7755 |
| 7 | 0.0909 | 0.7858 |
| 8 | 0.0556 | 0.7474 |
| 9 | 0.0444 | 0.7814 |
| 10 | 0.0347 | **0.8080** |

**Best Validation Accuracy:** **80.80%**

**Observations:**
- Strong early learning.
- Good stability across epochs.
- Small fluctuations around 74–79% are normal for medical data.
- Model has not converged yet—loss still decreasing.
- Clear room for improvement via staged finetuning.

---

## **Next Planned Experiments**
### **Experiment 2 — Finetuning Phase 1 (Freeze All Backbone Layers)**
- Freeze all ResNet layers except FC.
- LR: 1e-3
- Epochs: 2
- Expected: classifier adaptation.

### **Experiment 3 — Finetuning Phase 2 (Unfreeze Layer4)**
- Only unfreeze last block.
- LR: 1e-4
- Epochs: 5
- Expected: significant gain: 85–90%.

### **Experiment 4 — Finetuning Phase 3 (Unfreeze Entire Network)**
- Unfreeze all layers.
- LR: 1e-5
- Epochs: 5–10
- Expected: 90–95% accuracy.

---

## **Future Directions**
- Grad-CAM and activation visualization.
- ROC-AUC and sensitivity/specificity analysis.
- Testing on external datasets (APTOS / Messidor).
- Model calibration.
- Deployable inference script.
- Optional: retina vessel segmentation pretext task.

---

*This file will continue to grow as you run more experiments.*

