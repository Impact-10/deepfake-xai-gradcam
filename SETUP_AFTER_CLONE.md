# Setup Guide After Cloning

## What's Included vs What's Not

### ✅ Included in the Repository
- All Python scripts (train.py, evaluate.py, app.py, etc.)
- requirements.txt with all dependencies
- README with full documentation
- Empty directories (checkpoints/, dataset/, outputs/) with .gitkeep files

### ❌ NOT Included (You Need to Prepare)
- **Model checkpoints** (*.pth files) - Will be generated when you train
- **Dataset** (images) - You need to download/prepare separately
- **Training outputs** (results, visualizations) - Generated during training/evaluation

## Step-by-Step Setup After Cloning

### 1. Clone the Repository
```bash
git clone https://github.com/Impact-10/deepfake-xai-gradcam.git
cd deepfake-xai-gradcam
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
# OR
source .venv/bin/activate      # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify PyTorch/CUDA Setup
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
```

### 5. Prepare Dataset
You have two options:

#### Option A: Use Your Own Dataset
Create this exact structure:
```
dataset/
├── Train/
│   ├── Real/      # Put real face images here
│   └── Fake/      # Put fake/deepfake images here
├── val/
│   ├── Real/
│   └── Fake/
└── Test/
    ├── Real/
    └── Fake/
```

#### Option B: Download Public Datasets
- **FaceForensics++**: https://github.com/ondyari/FaceForensics
- **Celeb-DF**: http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html
- **DFDC**: https://ai.facebook.com/datasets/dfdc/

Then organize into the structure above.

**Requirements:**
- Images will be auto-resized to 224×224
- Formats: JPG, PNG
- Recommended: 1000+ images per class (Real/Fake) for good results

### 6. Train Models
Once dataset is ready:
```bash
# Train all three models (Xception → EfficientNet → ResNet50)
python train.py
```

This will:
- Automatically create checkpoints in `checkpoints/`
- Generate best model weights: `xception_best.pth`, `efficientnet_best.pth`, `resnet50_best.pth`
- Use smart epoch allocation and early stopping
- Resume from last checkpoint if training is interrupted

### 7. Evaluate Models (After Training)
```bash
# Evaluate all models on test set
python evaluate.py

# Compare models side-by-side
python compare.py
```

Results will be saved in `outputs/results/`

### 8. Generate Grad-CAM Visualizations
```bash
python gradcam.py
```

Visualizations saved in `outputs/gradcam/`

### 9. Run Streamlit App
```bash
streamlit run app.py
```

## Quick Troubleshooting

### "No module named 'torch'"
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: dataset/Train/Real"
You need to prepare the dataset first (see Step 5 above).

### "CUDA out of memory"
The training script uses smart batch sizing, but if you still face issues:
- Close other GPU applications
- The script will automatically reduce batch size for your GPU

### "No checkpoint found"
You need to train the models first:
```bash
python train.py
```

## What This Project Does

1. **Training**: Trains 3 CNN models (Xception, EfficientNet, ResNet50) for deepfake detection
2. **Evaluation**: Generates metrics (accuracy, precision, recall, F1, AUC) and visualizations
3. **Explainability**: Creates Grad-CAM heatmaps showing what regions the model focuses on
4. **Deployment**: Provides Streamlit app for easy inference on images/videos

## Expected Training Time (RTX 3050 Laptop GPU)
- Xception: ~1-2 hours (20 epochs)
- EfficientNet: ~45-90 min (15 epochs)
- ResNet50: ~1-1.5 hours (18 epochs)

Times vary based on dataset size and hardware.
