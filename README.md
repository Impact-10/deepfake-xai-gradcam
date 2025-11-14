# Deepfake Detection with Grad-CAM Explainability

## Overview
This project delivers an end-to-end pipeline for detecting deepfake imagery while explaining every prediction with Grad-CAM heatmaps. Three complementary convolutional backbones—Xception, EfficientNet-B0, and ResNet50—are trained, evaluated, and compared on curated real and fake facial datasets. The repository also packages a Streamlit dashboard for live demos and captures reusable evaluation artefacts for reporting.

## Key Features
- Multi-model training that sequentially fits Xception, EfficientNet-B0, and ResNet50 with smart epoch allocation and mixed-precision acceleration.
- Explainable AI tooling that produces Grad-CAM overlays from any checkpoint so stakeholders can inspect the facial regions that drive each prediction.
- Comprehensive evaluation pipeline generating confusion matrices, ROC curves, CSV summaries, and model comparison visualisations.
- Streamlit application for end-user image/video uploads, confidence reporting, and optional Grad-CAM visualization.
- GPU-friendly utilities: adaptive batch sizing, checkpoint resume, and CUDA tuning for RTX 3050-class devices.

## Repository Layout
```
app.py                      # Streamlit dashboard
train.py                    # Sequential training script (Xception → EfficientNet → ResNet50)
train_anti_overfitting.py   # Alternate training pipeline with stronger regularisation
evaluate.py                 # Test-set evaluation for a single model
compare.py                  # Aggregate comparison across checkpoints
gradcam.py                  # Grad-CAM generation utilities
utils.py                    # Data loaders, model builders, helpers
requirements.txt            # Python dependencies
dataset/                    # Train/val/test folders (Real vs Fake)
checkpoints/                # Saved best-model weights
outputs/                    # Metrics, comparison charts, Grad-CAM artefacts
```

## Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Impact-10/deepfake-xai-gradcam.git
   cd deepfake-xai-gradcam
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows Git Bash: source .venv/Scripts/activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Verify PyTorch detects your accelerator:
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"
   ```

## Dataset Preparation

**Important:** The dataset is **not included** in this repository due to size constraints. You need to prepare it before training.

### Option 1: Use Your Own Dataset
Organise your dataset exactly as follows:
```
dataset/
├── Train/
│   ├── Real/      # Real face images
│   └── Fake/      # Deepfake/manipulated images
├── val/
│   ├── Real/
│   └── Fake/
└── Test/
    ├── Real/
    └── Fake/
```

### Option 2: Download Public Datasets
You can use datasets like:
- **FaceForensics++** - https://github.com/ondyari/FaceForensics
- **Celeb-DF** - http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html
- **DFDC (Deepfake Detection Challenge)** - https://ai.facebook.com/datasets/dfdc/

After downloading, organize them into the structure above.

### Requirements
- Images are resized to 224×224 and normalised automatically
- Supported formats: JPG, PNG
- Balanced splits (equal Real/Fake) yield the most stable metrics
- Recommended: 1000+ images per class for training

## Training Workflow

**Note:** Model checkpoints are **not included** in the repository. You'll generate them by running the training scripts.

Run the full training sequence (Xception → EfficientNet → ResNet50):
```bash
python train.py
```
Highlights:
- Smart epoch allocation when `num_epochs` is omitted (`xception:20`, `efficientnet:15`, `resnet50:18`).
- Automatic mixed-precision (`torch.amp`) with `GradScaler`.
- Checkpoints saved to `checkpoints/{model}_best.pth` whenever validation accuracy improves; reruns resume from the last saved epoch unless the checkpoint is removed.
- Early stopping thresholds at 95%, 96%, and 97.5% validation accuracy.

For the regularisation-focused experiment:
```bash
python train_anti_overfitting.py
```

To train a single model programmatically:
```python
from train import train_model
train_model('efficientnet', num_epochs=12)
```

## Evaluation and Comparison
Generate metrics, confusion matrices, and ROC curves for each model:
```bash
python evaluate.py
```
Results are saved to `outputs/results/`. For consolidated comparisons (tables and plots):
```bash
python compare.py
```

## Grad-CAM Explainability
Create Grad-CAM overlays using the trained checkpoints:
```bash
python gradcam.py
```
The script samples one real and one fake image from `dataset/Test/` (if available) and writes visualisations to `outputs/gradcam/`.

Programmatic usage:
```python
from gradcam import generate_gradcam_for_image
prediction, probability, heatmap_path = generate_gradcam_for_image('xception', 'path/to/image.jpg')
```

## Streamlit Application
Launch the presentation-ready interface:
```bash
streamlit run app.py
```
Key options:
- Upload images or short videos for inference.
- Switch between Xception, EfficientNet, and ResNet50.
- Toggle Grad-CAM overlays for explainability.
- Review per-frame predictions and aggregated confidence.

## Results Snapshot
Representative metrics from the best checkpoints (see `outputs/results/evaluation_summary.csv` for exact values):

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Xception | ~0.978 | ~0.981 | ~0.975 | ~0.978 | ~0.992 |
| EfficientNet | ~0.969 | ~0.972 | ~0.966 | ~0.969 | ~0.988 |
| ResNet50 | ~0.964 | ~0.968 | ~0.960 | ~0.964 | ~0.985 |

Update the table if you regenerate checkpoints.

## Presentation Checklist
- [ ] Run `train.py` to demonstrate all three models and checkpoint resume behaviour.
- [ ] Walk through `outputs/results/` (confusion matrices, ROC curves, CSV summaries).
- [ ] Showcase Grad-CAM artefacts in `outputs/gradcam/`.
- [ ] Demo the Streamlit app with both image and video inputs.
- [ ] Reference `train_anti_overfitting.py` when discussing robustness improvements.

## License
This repository is provided for academic use within the project cohort. Add an explicit license file if broader distribution is planned.



