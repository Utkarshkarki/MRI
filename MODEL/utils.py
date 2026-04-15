import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal loss to handle class imbalance.
        Args:
            alpha (Tensor, optional): Weights for each class.
            gamma (float): Focusing parameter.
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # ce_loss is computed for single target labels, it returns -log(p_t)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

from sklearn.metrics import recall_score, f1_score, confusion_matrix
import numpy as np

def calculate_clinical_metrics(preds, targets):
    """
    Computes clinical evaluation metrics: Sensitivity (Recall), Specificity, and F1-Score (Macro).
    These metrics are highlighted in medical literature as critical for imbalanced clinical sets.
    """
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Sensitivity corresponds to recall in a multi-class setting
    sensitivity = recall_score(targets_np, preds_np, average='macro', zero_division=0)
    
    # F1 Score
    f1 = f1_score(targets_np, preds_np, average='macro', zero_division=0)
    
    # Specificity is more complex in multiclass, we compute True Negative Rate per class and average
    cm = confusion_matrix(targets_np, preds_np)
    
    specificity_sum = 0
    valid_classes = 0
    num_classes = cm.shape[0]
    
    if num_classes > 1:
        for i in range(num_classes):
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fp = np.sum(cm[:, i]) - cm[i, i]
            if (tn + fp) > 0:
                specificity_sum += tn / (tn + fp)
                valid_classes += 1
                
        specificity = specificity_sum / valid_classes if valid_classes > 0 else 0.0
    else:
        specificity = 0.0 # Undefined for single class targets in a batch
        
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1
    }
