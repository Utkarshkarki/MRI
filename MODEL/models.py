import torch
import torch.nn as nn
import torchvision.models as models

class MCDropoutResNet(nn.Module):
    def __init__(self, num_classes=4, num_heads=3, dropout_rate=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )
        
        self.classifiers = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(num_heads)
        ])
        
    def forward(self, x, return_features=False):
        features_raw = self.backbone(x)
        features_latent = self.feature_extractor(features_raw)
        
        head_outputs = [clf(features_latent) for clf in self.classifiers]
        logits = torch.stack(head_outputs, dim=0).mean(dim=0)
        
        if return_features:
            return logits, features_latent
        return logits
        
    def enable_mc_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
