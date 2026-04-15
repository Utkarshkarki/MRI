import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import copy
import numpy as np

from MODEL.dataset import BrainTumorDataset
from MODEL.transforms import get_train_transforms, get_val_transforms
from MODEL.models import MCDropoutResNet
from MODEL.utils import FocalLoss, calculate_clinical_metrics

def mixup_data(x, y, alpha=0.2, device='cpu'):
    """Returns mixed inputs, pairs of targets, and lambda for mixup."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_model(data_dir, num_epochs=20, batch_size=32, lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Training on device: {device}")
    
    # Setup Dataset
    full_dataset = BrainTumorDataset(data_dir, transform=get_train_transforms())
    
    # Splitting into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # Overwrite transform for validation dataset
    val_dataset.dataset.transform = get_val_transforms()
    
    workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    # Initialize Model, Loss (Hybrid SVM formulation), Optimizer
    model = MCDropoutResNet(num_classes=4).to(device)
    
    # Implementing the Hybrid CNN-SVM loss as outlined in the systemic review
    criterion = nn.MultiMarginLoss(p=1, margin=1.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
                
            running_loss = 0.0
            
            all_preds = []
            all_labels = []
            
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        # Apply MixUp augmentation for Synthetic Regularization
                        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2, device=device)
                        outputs = model(inputs)
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                        loss.backward()
                        optimizer.step()
                        _, preds = torch.max(outputs, 1)
                        # For tracking metrics, we treat original `labels` as ground truth approximation 
                        # just for training loop visualizations. True evaluation happens in exactly in val.
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloader.dataset)
            
            # Use sklearn for clinical evaluation
            clinical_metrics = calculate_clinical_metrics(torch.tensor(all_preds), torch.tensor(all_labels))
            sens = clinical_metrics['sensitivity']
            spec = clinical_metrics['specificity']
            f1 = clinical_metrics['f1_score']
            
            epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Sens: {sens:.4f} Spec: {spec:.4f} F1: {f1:.4f}")
            
            # Using F1-score to pick the best model due to its robustness to imbalance
            if phase == 'val' and f1 > best_f1:
                best_f1 = f1
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'best_model.pth')
                
    print(f"Training complete. Best Val F1-Score: {best_f1:.4f}")
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    train_model(data_dir='./Dataset', num_epochs=5, batch_size=32)
