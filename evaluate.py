import torch
import yaml
from torch.utils.data import random_split
from tqdm import tqdm
from sklearn.metrics import classification_report

from MODEL.dataset import BrainTumorDataset, get_stratified_loader
from MODEL.transforms import get_val_transforms
from MODEL.models import MCDropoutResNet
from MODEL.utils import calculate_clinical_metrics

def evaluate_model(data_dir="./Dataset", config_path="config.yaml", weights_path="best_model.pth"):
    print(f"\n--- Initializing Clinical Evaluation ---")
    
    # 1. Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 2. Setup Validation Dataset (Must match the exact seed used in train.py)
    full_dataset = BrainTumorDataset(data_dir, transform=get_val_transforms())
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    gen = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=gen)
    
    # Use stratified loader for validation (shuffle=False handled internally)
    val_loader = get_stratified_loader(val_dataset, config['training']['batch_size'], is_train=False)
    
    # 3. Load Model
    model = MCDropoutResNet(num_classes=config['model']['num_classes'])
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Successfully loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"\n[ERROR] '{weights_path}' not found! You must finish training first to evaluate.")
        return
        
    model = model.to(device)
    model.eval()
    
    # 4. Run Inference over validation set
    all_preds = []
    all_targets = []
    
    print("\nRunning offline predictions on validation set...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.tolist())
            
    # 5. Calculate Metrics
    # Convert lists to tensors for custom calculate_clinical_metrics
    preds_tensor = torch.tensor(all_preds)
    targets_tensor = torch.tensor(all_targets)
    
    custom_metrics = calculate_clinical_metrics(preds_tensor, targets_tensor)
    
    # 6. Print Report
    classes = full_dataset.classes
    print("\n" + "="*50)
    print(" 📊 DETAILED CLINICAL CLASSIFICATION REPORT")
    print("="*50)
    
    print(f"\n[Custom Math Metrics from utils.py]")
    print(f"Overall Sensitivity (Recall) : {custom_metrics['sensitivity']*100:.2f}%")
    print(f"Overall Specificity          : {custom_metrics['specificity']*100:.2f}%")
    print(f"Overall F1-Score (Macro)     : {custom_metrics['f1_score']*100:.2f}%")
    
    print("\n[Per-Class Scikit-Learn Breakdown]")
    report = classification_report(all_targets, all_preds, target_names=classes, digits=4)
    print(report)
    print("="*50 + "\n")

if __name__ == '__main__':
    evaluate_model()
