import torch
import cv2
import numpy as np
import yaml

from MODEL.models import MCDropoutResNet
from MODEL.transforms import get_val_transforms
from MODEL.uncertainty import UncertaintyEstimator
from MODEL.explainability import GradCAM, build_cam_overlay

class InferenceEngine:
    def __init__(self, weights_path="best_model.pth", config_path="config.yaml"):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception:
            self.config = {'model': {'num_classes': 4}, 'uncertainty': {'mc_passes': 15, 'anomaly_threshold': 0.25}}
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        self.model = MCDropoutResNet(num_classes=self.config['model']['num_classes'])
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        except:
            print("Notice: Missing weights. Initializing blind engine for testing routing structure.")
            
        self.model = self.model.to(self.device).eval()
        
        self.transform = get_val_transforms()
        self.uncertainty_engine = UncertaintyEstimator(
            self.model, 
            num_passes=self.config['uncertainty']['mc_passes'], 
            device=self.device
        )
        
        self.grad_cam = GradCAM(self.model, self.model.backbone[-1])

    def predict(self, raw_rgb_image):
        augmented = self.transform(image=raw_rgb_image)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        uncertainty_output = self.uncertainty_engine.estimate(input_tensor)
        predicted_idx = uncertainty_output['prediction_idx']
        
        predicted_class_name = self.classes[predicted_idx]
        is_anomalous = uncertainty_output['uncertainty_score'] > self.config['uncertainty']['anomaly_threshold']
        
        self.model.eval()
        cam_weight_map, _ = self.grad_cam.generate(input_tensor, target_class=predicted_idx)
        cam_overlay = build_cam_overlay(cv2.resize(raw_rgb_image, (224, 224)), cam_weight_map)
        
        return {
            'diagnosis': predicted_class_name,
            'confidence': uncertainty_output['confidence'],
            'uncertainty': uncertainty_output['uncertainty_score'],
            'is_anomaly': is_anomalous,
            'probabilities': uncertainty_output['mean_probs'],
            'heatmap_overlay': cam_overlay
        }
