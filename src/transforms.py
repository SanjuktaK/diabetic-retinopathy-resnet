from torchvision import transforms

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),                      # standardize size
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)), # slight crop variability
        transforms.RandomRotation(15),                      # realistic rotation
        transforms.RandomHorizontalFlip(p=0.5),             # OK in fundus
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2,
                               saturation=0.1),             # slightly improve variability
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])