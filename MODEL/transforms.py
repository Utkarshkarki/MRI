import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size=224):
    """
    Returns albumentations transforms for training.
    Includes CLAHE, spatial augmentations, and normalization.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        # Medical specific: Apply CLAHE for contrast enhancement
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        # Spatial augmentations to simulate inter-patient positioning variance
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
        # Pixel-level augmentations to simulate noise and scanner variance
        A.GaussNoise(std_range=(0.012, 0.027), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(image_size=224):
    """
    Returns albumentations transforms for validation/inference.
    Only basic resize and normalization.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), # apply consistently for inference
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
