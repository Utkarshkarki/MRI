import torch

class UncertaintyEstimator:
    def __init__(self, model, num_passes=15, device='cpu'):
        self.model = model
        self.num_passes = num_passes
        self.device = device
    
    def estimate(self, tensor_image):
        self.model.train() 
        input_batch = tensor_image.repeat(self.num_passes, 1, 1, 1).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_batch)
            probs = torch.softmax(logits, dim=1)
        
        mean_probs = probs.mean(dim=0)
        std_probs = probs.std(dim=0)
        
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-9))
        confidence, predicted_idx = torch.max(mean_probs, dim=0)
        
        return {
            'prediction_idx': predicted_idx.item(),
            'confidence': confidence.item(),
            'uncertainty_score': entropy.item(),
            'mean_probs': mean_probs.cpu().numpy()
        }
