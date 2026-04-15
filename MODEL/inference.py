import torch
import numpy as np
import cv2
from MODEL.models import MCDropoutResNet
from MODEL.transforms import get_val_transforms
from MODEL.explainability import GradCAM, overlay_cam_on_image

class InferenceEngine:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.model = MCDropoutResNet(num_classes=4).to(self.device)
        import os
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        else:
            print(f"Warning: Model weights not found at {model_path}. Using uninitialized weights.")
            
        self.transform = get_val_transforms()
        # The target layer for ResNet50 is usually the last basic block of layer4
        self.grad_cam = GradCAM(self.model, self.model.features[-1])

    def predict_with_uncertainty(self, image_np, num_passes=30):
        """
        Runs Monte Carlo Dropout to estimate uncertainty.
        Args:
           image_np (np.array): RGB image
           num_passes (int): number of forward passes
        """
        # Preprocessing
        augmented = self.transform(image=image_np)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        self.model.train() # Enable dropout for MC Dropout
        self.model.enable_dropout()
        
        preds = []
        with torch.no_grad():
            for _ in range(num_passes):
                out = self.model(input_tensor)
                probs = torch.softmax(out, dim=1)
                preds.append(probs.cpu().numpy())
                
        preds = np.array(preds) # Shape: (passes, 1, classes)
        
        mean_probs = np.mean(preds, axis=0)[0]
        std_probs = np.std(preds, axis=0)[0]
        
        predicted_class_idx = np.argmax(mean_probs)
        predicted_class = self.classes[predicted_class_idx]
        confidence = mean_probs[predicted_class_idx]
        uncertainty = std_probs[predicted_class_idx] # Predictive uncertainty for the predicted class
        
        # Generate Grad-CAM for interpretation (using eval mode)
        self.model.eval()
        cam_mask, _ = self.grad_cam.generate(input_tensor, predicted_class_idx)
        
        # Resize original image to match tensor size for overlay
        # Tensor size is 224x224 based on our transforms
        img_resized = cv2.resize(image_np, (224, 224))
        cam_overlay = overlay_cam_on_image(img_resized, cam_mask)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'mean_probs': mean_probs,
            'cam_overlay': cam_overlay
        }
