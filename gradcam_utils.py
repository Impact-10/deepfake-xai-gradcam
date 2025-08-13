import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """Grad-CAM implementation for CNN visualization."""

    def __init__(self, model, target_layer):
        """
        Initialize with model and the specific convolutional layer to target for Grad-CAM.
        Registers forward and backward hooks to capture activations and gradients.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the input tensor.
        If target_class is None, uses the class with highest prediction score.
        """
        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass to get gradients of target_class wrt activations
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)

        # Pool the gradients across the spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # shape: [batch, channels, 1, 1]

        # Weight the activations by corresponding pooled gradients
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # shape: [batch, 1, H, W]
        cam = F.relu(cam)  # Apply ReLU

        # Upsample CAM to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        # Normalize CAM
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.squeeze().cpu().numpy()

def show_cam_on_image(img, mask, use_rgb=False, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        img: Original image as numpy array (H, W, 3), float32 scaled [0,1].
        mask: Heatmap mask as numpy array (H, W), scaled [0,1].
        use_rgb: If True, convert heatmap to RGB after applying colormap.
        colormap: OpenCV colormap to apply.
    Returns:
        Overlayed image as uint8 numpy array (H, W, 3) scaled [0,255].
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    heatmap = np.float32(heatmap) / 255.0
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)  # Normalize between 0-1

    return np.uint8(255 * cam)
