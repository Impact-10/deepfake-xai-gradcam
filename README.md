# Deepfake Detection with Grad-CAM Visualization

This project demonstrates how to use Grad-CAM (Gradient-weighted Class Activation Mapping) for visualizing deepfake detection models. Grad-CAM helps understand which regions of an image contribute most to the model's decision.

## Features

- Grad-CAM implementation for CNN visualization
- Support for various CNN architectures
- Easy-to-use interface for generating heatmaps
- Visualization tools for model interpretability

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

1. Place your test image in `data/sample.jpg`
2. Run the Grad-CAM demo:
   ```bash
   python gradcam_demo.py --image data/sample.jpg
   ```

### Advanced Usage

```bash
python gradcam_demo.py \
    --image path/to/image.jpg \
    --model path/to/model.pth \
    --output output.jpg
```

## Directory Structure

```
deepfake-xai-gradcam/
│
├── data/
│   ├── sample.jpg        # Your test image (face image or frame from a video)
│
├── gradcam_demo.py       # Main script
├── gradcam_utils.py      # Grad-CAM helper functions
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Dependencies

- torch: PyTorch deep learning framework
- torchvision: Computer vision utilities
- opencv-python: Image processing
- matplotlib: Plotting and visualization
- Pillow: Image handling

## How It Works

1. **Model Loading**: Loads a pre-trained deepfake detection model
2. **Image Preprocessing**: Resizes and normalizes the input image
3. **Grad-CAM Generation**: Computes gradient-based attention maps
4. **Visualization**: Overlays heatmap on original image for interpretability

## Example Output

The script will generate:
- Original image
- Grad-CAM heatmap showing important regions
- Overlay visualization combining both

## Notes

- Replace the placeholder model loading with your actual deepfake detection model
- Ensure test images contain clear face regions for best results
- Adjust model architecture and preprocessing as needed for your specific use case
