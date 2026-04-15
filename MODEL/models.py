import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class MCDropoutResNet(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.5):
        """
        ResNet50 backbone modified for Monte Carlo Dropout (Uncertainty Estimation).
        """
        super(MCDropoutResNet, self).__init__()
        # Load pre-trained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Extract features (up to the avgpool)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Classifier replacement with MC Dropout (Hybrid CNN-SVM boundary representation)
        in_features = self.backbone.fc.in_features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            # Outputs raw margins for MultiMarginLoss (Linear SVM), not just softmax logits
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def enable_dropout(self):
        """
        Enable dropout layers during test time for MC-Dropout.
        """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
