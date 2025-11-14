import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import transforms
from utils import get_model

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # Find target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Target layer {self.target_layer_name} not found")
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        # Get device from input tensor
        device = input_tensor.device
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, 0].backward()
        
        # Generate CAM - ensure all tensors are on same device
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Move to same device if needed
        if gradients.device != device:
            gradients = gradients.to(device)
        if activations.device != device:
            activations = activations.to(device)
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps - create tensor on correct device
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU and normalize
        cam = F.relu(cam)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        return cam.detach().cpu().numpy()

def get_target_layer(model_name):
    """Get target layer name for Grad-CAM based on model architecture"""
    if model_name == 'xception':
        return 'backbone.conv4'  # Last convolutional layer before global pooling
    elif model_name == 'efficientnet':
        return 'backbone.conv_head'  # Head convolution layer
    elif model_name == 'resnet50':
        return 'backbone.layer4'  # Last residual block
    else:
        raise ValueError(f"Unknown model: {model_name}")

def load_and_preprocess_image(image_path):
    """Load and preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor, image

def create_heatmap_overlay(original_image, cam, alpha=0.6):
    """Create heatmap overlay on original image"""
    # Resize CAM to match original image size
    height, width = original_image.size[1], original_image.size[0]
    cam_resized = cv2.resize(cam, (width, height))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Convert PIL to numpy
    original_np = np.array(original_image)
    
    # Overlay
    overlay = heatmap * alpha + original_np * (1 - alpha)
    overlay = np.uint8(overlay)
    
    return overlay

def generate_gradcam_for_image(model_name, image_path, output_dir="outputs/gradcam"):
    """Generate Grad-CAM for a single image"""
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_model(model_name)
    checkpoint = torch.load(f'checkpoints/{model_name}_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    input_tensor, original_image = load_and_preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        # ImageFolder loads alphabetically: Fake=0, Real=1
        # So prob > 0.5 means Real, prob <= 0.5 means Fake
        prediction = "REAL" if prob > 0.5 else "FAKE"
    
    # Generate Grad-CAM
    target_layer = get_target_layer(model_name)
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor)
    
    # Create overlay
    overlay = create_heatmap_overlay(original_image, cam)
    
    # Save results
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\nPrediction: {prediction} ({prob:.3f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_{image_name}_gradcam.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Grad-CAM saved for {image_name} using {model_name}")
    return prediction, prob, f'{output_dir}/{model_name}_{image_name}_gradcam.png'

def main():
    """Generate Grad-CAM for user-specified image using all three models"""
    models = ['xception', 'efficientnet', 'resnet50']
    
    print("="*70)
    print("GRAD-CAM VISUALIZATION GENERATOR")
    print("="*70)
    
    # Get image path from user
    image_path = input("\nEnter path to image (e.g., dataset/Test/Real/real_5.jpg): ").strip()
    
    # Validate image path
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    print(f"\nProcessing image: {image_path}")
    print("-"*70)
    
    # Generate Grad-CAM for each model
    for model_name in models:
        checkpoint_path = f'checkpoints/{model_name}_best.pth'
        if os.path.exists(checkpoint_path):
            try:
                print(f"Generating Grad-CAM using {model_name.upper()}...")
                generate_gradcam_for_image(model_name, image_path)
            except Exception as e:
                print(f"Error generating Grad-CAM for {model_name}: {str(e)}")
        else:
            print(f"No checkpoint found for {model_name}")
    
    print("\n" + "="*70)
    print("GRAD-CAM GENERATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated 3 composite images in 'outputs/gradcam/' folder:")
    print(f"  - xception_*_gradcam.png")
    print(f"  - efficientnet_*_gradcam.png")
    print(f"  - resnet50_*_gradcam.png")
    print("\nEach image shows: Original | Heatmap | Overlay")

if __name__ == "__main__":
    main()