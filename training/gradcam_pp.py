# gradcam_pp.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """
        Returns normalized Grad-CAM++ map (224x224)
        """
        self.model.zero_grad()

        logits = self.model(input_tensor)
        score = logits[:, 0].sum()  # binary classifier → logit

        score.backward(retain_graph=True)

        grads = self.gradients       # [B, C, H, W]
        acts = self.activations

        eps = 1e-8

        grads2 = grads ** 2
        grads3 = grads ** 3

        alpha = grads2 / (
            2 * grads2 +
            torch.sum(acts * grads3, dim=(2, 3), keepdim=True) +
            eps
        )

        weights = torch.sum(alpha * F.relu(grads), dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1)

        cam = F.relu(cam)
        cam = cam[0].cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        cam /= (cam.max() + eps)

        return cam
