import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class ConfidencePenalty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        probs = torch.softmax(logits, dim=1)
        max_prob = torch.max(probs, dim=1)[0]
        penalty = torch.clamp(max_prob - 0.95, min=0).mean()
        return penalty

class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=1.0, penalty_weight=0.1):
        super().__init__()
        self.focal = FocalLoss()
        self.penalty = ConfidencePenalty()
        self.focal_weight = focal_weight
        self.penalty_weight = penalty_weight

    def forward(self, logits, targets):
        return (self.focal_weight * self.focal(logits, targets)) + (self.penalty_weight * self.penalty(logits))
