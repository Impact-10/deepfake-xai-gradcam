import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gradcam_utils import GradCAM, show_cam_on_image
import argparse
import os

def load_model(model_path=None):
    """
    Load pre-trained ResNet50 model modified for binary classification (real/fake).
    If a checkpoint path is provided, loads the saved weights.
    """
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary classification head

    if model_path is not None and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.eval()
    return model

def preprocess_image(image_path):
    """
    Load and preprocess an input image for ResNet50.
    Returns the tensor for model input and the resized image as numpy (scaled 0-1) for visualization.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # For visualization: convert to numpy float32 scaled [0,1]
    image_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    return tensor, image_np

def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Demo for Deepfake Detection')
    parser.add_argument('--image', type=str, default='data/sample.jpg',
                        help='Path to input image')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='gradcam_output.jpg',
                        help='Output image path')
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Processing image: {args.image}")
    input_tensor, original_image = preprocess_image(args.image)
    input_tensor = input_tensor.to(device)

    # Choose last convolutional block for Grad-CAM in ResNet50
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    print("Generating Grad-CAM heatmap...")
    grayscale_cam = gradcam(input_tensor)  # Returns numpy array HxW scaled [0,1]

    # Overlay heatmap on image
    visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

    # Save output image
    cv2.imwrite(args.output, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM visualization saved to: {args.output}")

    # Display using matplotlib
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(grayscale_cam, cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(visualization)
    plt.title('Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
