import streamlit as st
import torch
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import tempfile




# Import project modules
from utils import get_model
from gradcam import GradCAM, get_target_layer, create_heatmap_overlay
from torchvision import transforms

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection with Grad-CAM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for classic styling
st.markdown("""
<style>
    /* Main theme colors - Classic blue and white */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --success-color: #059669;
        --danger-color: #dc2626;
        --neutral-color: #6b7280;
        --background: #f8fafc;
    }
    
    /* Main container styling */
    .main {
        background-color: var(--background);
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary-color);
    }
    
    /* Title styling */
    .main-title {
        color: var(--primary-color);
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    .subtitle {
        color: var(--neutral-color);
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Badge styling */
    .badge-real {
        background-color: var(--success-color);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 1.2rem;
    }
    
    .badge-fake {
        background-color: var(--danger-color);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 1.2rem;
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section headers */
    .section-header {
        color: var(--primary-color);
        font-size: 1.8rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary-color);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: var(--secondary-color);
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid var(--secondary-color);
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def load_model_cached(model_name):
    """Load and cache model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(model_name)
        checkpoint_path = f'checkpoints/{model_name}_best.pth'
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            return model, device, checkpoint.get('val_acc', 0.0)
        else:
            st.error(f"Checkpoint not found: {checkpoint_path}")
            return None, device, 0.0
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, 0.0

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_image(model, image_tensor, device):
    """Make prediction on image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        # ImageFolder loads alphabetically: Fake=0, Real=1
        # So prob > 0.5 means Real, prob <= 0.5 means Fake
        prediction = "REAL" if prob > 0.5 else "FAKE"
        confidence = prob if prob > 0.5 else (1 - prob)
    return prediction, confidence, prob

def generate_gradcam_overlay(model, image_tensor, model_name, device, original_image):
    """Generate Grad-CAM overlay"""
    try:
        target_layer = get_target_layer(model_name)
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(image_tensor.to(device))
        overlay = create_heatmap_overlay(original_image, cam)
        return overlay, cam
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {str(e)}")
        return None, None

def extract_video_frames(video_path, num_frames=5):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return []
    
    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

def load_evaluation_results():
    """Load evaluation results from CSV"""
    results_path = 'outputs/results/evaluation_summary.csv'
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    return None

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-title">Deepfake Detection with Grad-CAM Explainability</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Multi-model deepfake classification with explainable AI visualizations</p>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Section:",
        ["Home & Prediction", "Model Comparison", "Grad-CAM Gallery", "About & Instructions"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Info")
    st.sidebar.info("This dashboard provides deepfake detection using three state-of-the-art models with explainable AI visualizations.")
    
    # Page routing
    if page == "Home & Prediction":
        prediction_page()
    elif page == "Model Comparison":
        comparison_page()
    elif page == "Grad-CAM Gallery":
        gradcam_gallery_page()
    else:
        about_page()

def prediction_page():
    """Image/Video Upload & Prediction Page"""
    st.markdown('<h2 class="section-header">Upload & Analyze</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Upload Media")
        
        upload_type = st.radio("Select input type:", ["Image", "Video"], horizontal=True)
        
        if upload_type == "Image":
            uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
        else:
            uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Settings")
        
        model_name = st.selectbox(
            "Select Model:",
            ["xception", "efficientnet", "resnet50"],
            format_func=lambda x: x.upper()
        )
        
        enable_gradcam = st.toggle("Enable Grad-CAM Overlay", value=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        if upload_type == "Image":
            process_image(uploaded_file, model_name, enable_gradcam)
        else:
            process_video(uploaded_file, model_name, enable_gradcam)

def process_image(uploaded_file, model_name, enable_gradcam):
    """Process uploaded image"""
    # Display original image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Original Image")
        st.image(image, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Predict button
    if st.button("Analyze Image", width='stretch'):
        with st.spinner('Analyzing image...'):
            # Load model
            model, device, val_acc = load_model_cached(model_name)
            
            if model is None:
                st.error("Failed to load model. Please check if checkpoint exists.")
                return
            
            # Preprocess and predict
            image_tensor = preprocess_image(image)
            prediction, confidence, prob = predict_image(model, image_tensor, device)
            
            # Display results
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Prediction Results")
                
                # Prediction badge
                if prediction == "REAL":
                    st.markdown(f'<div style="text-align: center;"><span class="badge-real">REAL</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="text-align: center;"><span class="badge-fake">FAKE</span></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Confidence metrics
                st.metric("Confidence", f"{confidence*100:.2f}%")
                st.metric("Model Used", model_name.upper())
                st.metric("Model Validation Accuracy", f"{val_acc*100:.2f}%" if val_acc > 0 else "N/A")
                
                # Confidence bar
                st.progress(confidence)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Grad-CAM visualization
            if enable_gradcam:
                st.markdown('<h3 class="section-header">Explainability Analysis</h3>', unsafe_allow_html=True)
                
                with st.spinner('Generating Grad-CAM visualization...'):
                    overlay, cam = generate_gradcam_overlay(model, image_tensor, model_name, device, image)
                    
                    if overlay is not None:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown("#### Original")
                            st.image(image, width='stretch')
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown("#### Heatmap")
                            fig, ax = plt.subplots(figsize=(6, 6))
                            im = ax.imshow(cam, cmap='jet')
                            ax.axis('off')
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            st.pyplot(fig)
                            plt.close()
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown("#### Overlay")
                            st.image(overlay, width='stretch')
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="info-box"><strong>Interpretation:</strong> Red/warm regions indicate areas the model focused on when making its decision. These are typically facial features, edges, or artifacts.</div>', unsafe_allow_html=True)

def process_video(uploaded_file, model_name, enable_gradcam):
    """Process uploaded video"""
    # Save video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    if st.button("Analyze Video", width='stretch'):
        with st.spinner('Extracting frames from video...'):
            frames = extract_video_frames(video_path, num_frames=5)
            
            if not frames:
                st.error("Failed to extract frames from video.")
                return
            
            st.success(f"Extracted {len(frames)} frames from video")
        
        # Load model
        model, device, val_acc = load_model_cached(model_name)
        
        if model is None:
            st.error("Failed to load model. Please check if checkpoint exists.")
            return
        
        # Analyze each frame
        predictions = []
        confidences = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, frame in enumerate(frames):
            status_text.text(f'Analyzing frame {idx + 1}/{len(frames)}...')
            
            image_tensor = preprocess_image(frame)
            prediction, confidence, prob = predict_image(model, image_tensor, device)
            
            predictions.append(prediction)
            confidences.append(confidence)
            
            progress_bar.progress((idx + 1) / len(frames))
        
        status_text.empty()
        progress_bar.empty()
        
        # Aggregate results
        fake_count = predictions.count("FAKE")
        real_count = predictions.count("REAL")
        avg_confidence = np.mean(confidences)
        
        overall_prediction = "FAKE" if fake_count > real_count else "REAL"
        
        # Display aggregate results
        st.markdown('<h3 class="section-header">Video Analysis Results</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Overall</div><div class="metric-value">{overall_prediction}</div></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Confidence</div><div class="metric-value">{avg_confidence*100:.1f}%</div></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Fake Frames</div><div class="metric-value">{fake_count}/{len(frames)}</div></div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Real Frames</div><div class="metric-value">{real_count}/{len(frames)}</div></div>', unsafe_allow_html=True)
        
        # Display per-frame results
        st.markdown('<h3 class="section-header">Frame-by-Frame Analysis</h3>', unsafe_allow_html=True)
        
        cols = st.columns(len(frames))
        
        for idx, (frame, pred, conf) in enumerate(zip(frames, predictions, confidences)):
            with cols[idx]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.image(frame, width='stretch')
                
                if pred == "REAL":
                    st.markdown(f'<div style="text-align: center; margin: 0.5rem 0;"><span class="badge-real" style="font-size: 0.9rem; padding: 0.3rem 1rem;">REAL</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="text-align: center; margin: 0.5rem 0;"><span class="badge-fake" style="font-size: 0.9rem; padding: 0.3rem 1rem;">FAKE</span></div>', unsafe_allow_html=True)
                
                st.caption(f"Confidence: {conf*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Grad-CAM for video frames
        if enable_gradcam:
            st.markdown('<h3 class="section-header">Grad-CAM Analysis (Sample Frames)</h3>', unsafe_allow_html=True)
            
            # Show Grad-CAM for first and last frame
            sample_indices = [0, -1] if len(frames) > 1 else [0]
            
            for sample_idx in sample_indices:
                frame = frames[sample_idx]
                image_tensor = preprocess_image(frame)
                
                with st.spinner(f'Generating Grad-CAM for frame {sample_idx + 1}...'):
                    overlay, cam = generate_gradcam_overlay(model, image_tensor, model_name, device, frame)
                    
                    if overlay is not None:
                        st.markdown(f"**Frame {sample_idx + 1}**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.image(frame, caption="Original", width='stretch')
                        
                        with col2:
                            fig, ax = plt.subplots(figsize=(4, 4))
                            im = ax.imshow(cam, cmap='jet')
                            ax.axis('off')
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            st.pyplot(fig)
                            plt.close()
                        
                        with col3:
                            st.image(overlay, caption="Overlay", width='stretch')
    
    # Clean up temp file
    try:
        os.unlink(video_path)
    except:
        pass

def comparison_page():
    """Model Comparison Page"""
    st.markdown('<h2 class="section-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    # Load evaluation results
    results_df = load_evaluation_results()
    
    if results_df is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Performance Metrics")
        
        # Format numeric columns as percentages for display
        display_df = results_df.copy()
        percentage_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        
        for col in percentage_cols:
            if col in display_df.columns:
                # Convert to percentage if in decimal format (0-1)
                if display_df[col].dtype != 'object' and display_df[col].max() <= 1.0:
                    display_df[col] = (display_df[col] * 100).round(2).astype(str) + '%'
                elif display_df[col].dtype != 'object':
                    display_df[col] = display_df[col].round(2).astype(str) + '%'
        
        # Display metrics table
        st.dataframe(display_df, width='stretch', hide_index=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Accuracy Comparison")
            
            # Bar chart for accuracy
            fig, ax = plt.subplots(figsize=(8, 5))
            models = results_df['Model']
            
            # Handle both string ('95.6%'), percentage (95.6), and decimal (0.956) formats
            if results_df['Accuracy'].dtype == 'object':
                accuracy = results_df['Accuracy'].str.rstrip('%').astype(float)
            else:
                accuracy = results_df['Accuracy'].astype(float)
            
            # Convert to percentage if values are in 0-1 range (decimal format)
            if accuracy.max() <= 1.0:
                accuracy = accuracy * 100
            
            bars = ax.bar(models, accuracy, color=['#1e3a8a', '#3b82f6', '#60a5fa'])
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_ylim(80, 100)  # Adjusted range for better visibility
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%',
                       ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### F1-Score & Precision")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            x = np.arange(len(models))
            width = 0.35
            
            # Handle both string and numeric formats
            if results_df['Precision'].dtype == 'object':
                precision = results_df['Precision'].str.rstrip('%').astype(float)
            else:
                precision = results_df['Precision'].astype(float)
            
            if results_df['F1-Score'].dtype == 'object':
                f1_score = results_df['F1-Score'].str.rstrip('%').astype(float)
            else:
                f1_score = results_df['F1-Score'].astype(float)
            
            # Convert to percentage if values are in 0-1 range (decimal format)
            if precision.max() <= 1.0:
                precision = precision * 100
            if f1_score.max() <= 1.0:
                f1_score = f1_score * 100
            
            ax.bar(x - width/2, precision, width, label='Precision', color='#059669')
            ax.bar(x + width/2, f1_score, width, label='F1-Score', color='#3b82f6')
            
            ax.set_ylabel('Score (%)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.set_ylim(80, 100)  # Adjusted range for better visibility
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Confusion matrices and ROC curves
        st.markdown('<h3 class="section-header">Detailed Analysis</h3>', unsafe_allow_html=True)
        
        selected_model = st.selectbox("Select model for detailed view:", 
                                      ["xception", "efficientnet", "resnet50"],
                                      format_func=lambda x: x.upper())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"#### {selected_model.upper()} - Confusion Matrix")
            
            cm_path = f'outputs/results/{selected_model}_confusion_matrix.png'
            if os.path.exists(cm_path):
                st.image(cm_path, width='stretch')
            else:
                st.warning("Confusion matrix not found. Run evaluation first.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"#### {selected_model.upper()} - ROC Curve")
            
            roc_path = f'outputs/results/{selected_model}_roc_curve.png'
            if os.path.exists(roc_path):
                st.image(roc_path, width='stretch')
            else:
                st.warning("ROC curve not found. Run evaluation first.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.warning("No evaluation results found. Please run `python evaluate.py` to generate performance metrics.")
        st.info("The evaluation script will analyze all trained models and generate comprehensive performance reports.")

def gradcam_gallery_page():
    """Grad-CAM Gallery Page"""
    st.markdown('<h2 class="section-header">Grad-CAM Visualization Gallery</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box"><strong>About Grad-CAM:</strong> Gradient-weighted Class Activation Mapping (Grad-CAM) highlights the important regions in an image that influenced the model\'s decision. Warmer colors (red/yellow) indicate higher importance.</div>', unsafe_allow_html=True)
    
    # Check for existing Grad-CAM outputs
    gradcam_dir = 'outputs/gradcam'
    
    if os.path.exists(gradcam_dir):
        gradcam_files = [f for f in os.listdir(gradcam_dir) if f.endswith('.png')]
        
        if gradcam_files:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### Available Visualizations ({len(gradcam_files)} files)")
            
            # Group by model
            models = ['xception', 'efficientnet', 'resnet50']
            
            for model in models:
                model_files = [f for f in gradcam_files if f.startswith(model)]
                
                if model_files:
                    st.markdown(f"#### {model.upper()}")
                    
                    cols = st.columns(min(3, len(model_files)))
                    
                    for idx, filename in enumerate(model_files[:3]):  # Show first 3
                        with cols[idx % 3]:
                            st.image(os.path.join(gradcam_dir, filename), 
                                   caption=filename.replace(f'{model}_', '').replace('_gradcam.png', ''),
                                   width='stretch')
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No Grad-CAM visualizations found in the gallery.")
            st.info("Generate visualizations by running `python gradcam.py` or using the prediction page with Grad-CAM enabled.")
    else:
        st.warning("Grad-CAM output directory not found.")
        st.info("Generate visualizations by running `python gradcam.py` or using the prediction page with Grad-CAM enabled.")
    
    # Generate new Grad-CAM section
    st.markdown('<h3 class="section-header">Generate New Visualization</h3>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("Upload an image to generate Grad-CAM visualization for all three models")
    
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'], key='gradcam_upload')
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        if st.button("Generate Grad-CAM for All Models"):
            st.markdown("### Original Image")
            st.image(image, width=300)
            
            st.markdown("### Grad-CAM Results")
            
            models = ['xception', 'efficientnet', 'resnet50']
            
            for model_name in models:
                with st.spinner(f'Generating Grad-CAM for {model_name.upper()}...'):
                    model, device, _ = load_model_cached(model_name)
                    
                    if model is not None:
                        image_tensor = preprocess_image(image)
                        overlay, cam = generate_gradcam_overlay(model, image_tensor, model_name, device, image)
                        
                        if overlay is not None:
                            st.markdown(f"#### {model_name.upper()}")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.image(image, caption="Original", width='stretch')
                            
                            with col2:
                                fig, ax = plt.subplots(figsize=(4, 4))
                                im = ax.imshow(cam, cmap='jet')
                                ax.axis('off')
                                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                                st.pyplot(fig)
                                plt.close()
                            
                            with col3:
                                st.image(overlay, caption="Overlay", width='stretch')
    
    st.markdown('</div>', unsafe_allow_html=True)

def about_page():
    """About & Instructions Page"""
    st.markdown('<h2 class="section-header">About This Project</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### Project Overview
    
    This dashboard provides an end-to-end solution for detecting deepfake images and videos using state-of-the-art deep learning models. 
    The system not only classifies media as real or fake but also explains its decisions through Grad-CAM visualizations.
    
    ### Architecture
    
    The project employs three complementary CNN architectures:
    
    - **Xception**: Depthwise separable convolutions for efficient feature extraction
    - **EfficientNet-B0**: Balanced accuracy and efficiency through compound scaling  
    - **ResNet50**: Deep residual learning with skip connections
    
    All models are trained on a curated dataset of real and deepfake facial imagery with:
    - Smart epoch allocation based on model complexity
    - Mixed-precision training for faster convergence
    - Early stopping to prevent overfitting
    - Checkpoint resume capability
    
    ### Explainable AI with Grad-CAM
    
    Grad-CAM (Gradient-weighted Class Activation Mapping) provides visual explanations by:
    1. Computing gradients of the prediction with respect to feature maps
    2. Weighting feature activations by these gradients
    3. Generating heatmaps showing important regions
    4. Overlaying heatmaps on original images
    
    **Interpretation Guide:**
    - Red/warm areas: High importance for the decision
    - Blue/cool areas: Low importance
    - Focus typically appears on facial features, edges, or artifacts
    
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="section-header">User Guide</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        #### Prediction Page
        
        1. **Upload**: Select an image (JPG/PNG) or video (MP4)
        2. **Configure**: Choose your model and enable/disable Grad-CAM
        3. **Analyze**: Click the analyze button to process
        4. **Review**: Check prediction, confidence, and visualizations
        
        **For Images:**
        - Instant prediction with confidence score
        - Optional Grad-CAM overlay showing decision factors
        
        **For Videos:**
        - Frame-by-frame analysis (5 frames sampled)
        - Aggregate prediction across all frames
        - Optional Grad-CAM for sample frames
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        #### Model Comparison
        
        View comprehensive performance metrics:
        - Accuracy, Precision, Recall, F1-Score, AUC
        - Visual comparisons across all models
        - Confusion matrices showing classification details
        - ROC curves for threshold analysis
        
        #### Grad-CAM Gallery
        
        - Browse previously generated visualizations
        - Compare Grad-CAM outputs across models
        - Generate new visualizations for uploaded images
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### Tips for Best Results
    
    - **Image Quality**: Use clear, well-lit images with visible faces
    - **Video Length**: Shorter videos (< 30 seconds) process faster
    - **Model Selection**: Xception typically offers best accuracy, EfficientNet is fastest
    - **Grad-CAM**: Enable for explainability, but adds processing time
    - **Interpretation**: Look for focus on eyes, mouth, and facial boundaries in Grad-CAM
    
    ### Technical Details
    
    - **Input Size**: All images resized to 224×224 pixels
    - **Normalization**: ImageNet mean and std deviation
    - **Output**: Binary classification (Real=0, Fake=1)
    - **Threshold**: 0.5 probability cutoff
    - **Device**: Automatic GPU/CPU detection
    
    ### Repository & Resources
    
    - Training script: `python train.py`
    - Evaluation: `python evaluate.py`
    - Grad-CAM generation: `python gradcam.py`
    - Model comparison: `python compare.py`
    
    For more details, refer to the [README.md](https://github.com/Impact-10/deepfake-xai-gradcam) in the repository.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
