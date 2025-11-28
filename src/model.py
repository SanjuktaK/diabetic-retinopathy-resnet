import torch
from torch import nn
from torchvision import models

def build_model(num_classes=2, freeze_backbone=False):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # freeze / unfreeze
    for name, param in model.named_parameters():
        if freeze_backbone:
            # freeze everything except fc
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True   # full fine-tune

    return model
