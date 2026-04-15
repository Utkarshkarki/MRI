import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        logits = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        score = logits[0][target_class]
        self.model.zero_grad()
        score.backward()
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam.cpu().detach().numpy(), target_class

def build_cam_overlay(original_image_np, cam_weights, alpha=0.5):
    cam_weights_resized = cv2.resize(cam_weights, (original_image_np.shape[1], original_image_np.shape[0]))
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_weights_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image_np, 1 - alpha, cam_heatmap, alpha, 0)
    return overlay
