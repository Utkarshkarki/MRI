import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        self.image_paths = []
        self.labels = []
        
        for idx, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
                
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(idx)
                
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        label = self.labels[idx]
        return img, label

def get_stratified_loader(dataset, batch_size, is_train=True):
    if not is_train:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

    # PyTorch random_split returns a Subset, which hides the labels inside indices
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        subset_labels = [dataset.dataset.labels[i] for i in dataset.indices]
    else:
        subset_labels = dataset.labels

    # Calculate frequencies inside this specific subset
    class_counts = np.bincount(subset_labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    
    # Assign the calculated structural weight to each sample index
    sample_weights = class_weights[subset_labels]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True
    )
