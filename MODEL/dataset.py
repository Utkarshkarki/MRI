import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None, classes=None):
        """
        Args:
            data_dir (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            classes (list, optional): List of class names. Defaults to glioma, meningioma, notumor, pituitary.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        if classes is None:
            self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        else:
            self.classes = classes
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.exists(cls_dir):
                print(f"Warning: Directory {cls_dir} does not exist.")
                continue
                
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read image using OpenCV (BGR to RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            # Assume albumentations is used: albumentations returns a dict
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label
