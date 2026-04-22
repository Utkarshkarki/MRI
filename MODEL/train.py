import torch
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
import os
import copy
import yaml

from MODEL.dataset import BrainTumorDataset, get_stratified_loader
from MODEL.transforms import get_train_transforms, get_val_transforms
from MODEL.models import MCDropoutResNet
from MODEL.loss import CombinedLoss

class GPUOptimizedTrainer:
    def __init__(self, data_dir, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.full_dataset = BrainTumorDataset(data_dir, transform=get_train_transforms())
        train_size = int(0.8 * len(self.full_dataset))
        val_size = len(self.full_dataset) - train_size
        
        self.train_dataset = copy.deepcopy(self.full_dataset)
        self.val_dataset = copy.deepcopy(self.full_dataset)
        self.val_dataset.transform = get_val_transforms()
        
        gen = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(self.full_dataset), generator=gen).tolist()
        self.train_subset = torch.utils.data.Subset(self.train_dataset, indices[:train_size])
        self.val_subset = torch.utils.data.Subset(self.val_dataset, indices[train_size:])
        self.train_loader = get_stratified_loader(self.train_subset, self.config['training']['batch_size'], is_train=True)
        self.val_loader = get_stratified_loader(self.val_subset, self.config['training']['batch_size'], is_train=False)
        
        self.model = MCDropoutResNet(num_classes=self.config['model']['num_classes']).to(self.device)
        self.criterion = CombinedLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=float(self.config['training']['learning_rate']))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config['training']['num_epochs'])
        
        self.use_amp = self.config['gpu']['mixed_precision']
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        
        # Checkpoint Resume Logic natively baked into the constructor
        self.start_epoch = 0
        if os.path.exists('latest_checkpoint.pth'):
            print("\n[ACTIVE] Found previously saved state. Injecting checkpoint targets to resume training...")
            checkpoint = torch.load('latest_checkpoint.pth', map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"--> Continuing straight from Epoch {self.start_epoch + 1}")
        
    def train(self):
        print(f"Beginning Structural Training Sequence on Engine: {self.device}...")
        best_acc = 0.0
        best_wts = copy.deepcopy(self.model.state_dict())
        epochs = self.config['training']['num_epochs']
        
        for epoch in range(self.start_epoch, epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    loader = self.train_loader
                else:
                    self.model.eval()
                    loader = self.val_loader
                
                loss_rt = 0.0
                corrects = 0
                
                for inputs, labels in tqdm(loader, desc=f"{phase}"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                            
                        if phase == 'train':
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            
                    loss_rt += loss.item() * inputs.size(0)
                    corrects += torch.sum(preds == labels.data)
                    
                if phase == 'train':
                    self.scheduler.step()
                    
                ep_loss = loss_rt / len(loader.dataset)
                ep_acc = corrects.double() / len(loader.dataset)
                print(f"{phase.upper()} Loss: {ep_loss:.4f} Acc: {ep_acc:.4f}")
                
                if phase == 'val' and ep_acc > best_acc:
                    best_acc = ep_acc
                    best_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(best_wts, 'best_model.pth')
                    
            # Safe-save absolute structural layout at the immediate end of every epoch cycle
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict()
            }, 'latest_checkpoint.pth')
                    
        print(f"\nTraining Routine Terminated. Peak Validation Stability Hit: {best_acc:.4f}")

if __name__ == '__main__':
    trainer = GPUOptimizedTrainer(data_dir='./Dataset')
    trainer.train()
