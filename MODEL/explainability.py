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
        
        # Hook into the target layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        out = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(out, dim=1).item()
            
        self.model.zero_grad()
        score = out[0, target_class]
        score.backward(retain_graph=True)
        
        # Pull the gradients and activations
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight activations by the gradients
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU to keep only features that have positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        if np.max(cam) != 0:
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            
        return cam, target_class

def overlay_cam_on_image(img, mask):
    """
    Overlays Grad-CAM mask on the original image.
    img: original image array (RGB) HxWxC
    mask: heatmap mask 0-1
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] # BGR to RGB
    
    img = np.float32(img) / 255
    cam_result = heatmap + img
    cam_result = cam_result / np.max(cam_result)
    
    return np.uint8(255 * cam_result)
