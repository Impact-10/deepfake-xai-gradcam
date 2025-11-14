import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
import time
import pandas as pd
from utils import DeepfakeDataLoader, get_model, calculate_metrics

def evaluate_model(model_name):
    """Evaluate a single model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_model(model_name)
    checkpoint_path = f'checkpoints/{model_name}_best.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found for {model_name}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load test data
    dataloader = DeepfakeDataLoader()
    _, _, test_loader = dataloader.get_dataloaders()
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []
    test_loss = 0.0
    
    criterion = nn.BCEWithLogitsLoss()
    
    inference_times = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time / inputs.size(0))  # Per sample
            
            loss = criterion(outputs, labels.float().unsqueeze(1))
            test_loss += loss.item()
            
            # Convert logits to predictions and probabilities
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy().flatten())
    
    test_loss /= len(test_loader)
    avg_inference_time = np.mean(inference_times)
    
    # Calculate metrics
    accuracy, precision, recall, f1, auc_score = calculate_metrics(all_labels, all_preds, all_probs)
    
    results = {
        'model': model_name,
        'test_loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'avg_inference_time': avg_inference_time,
        'y_true': all_labels,
        'y_pred': all_preds,
        'y_prob': all_probs
    }
    
    print(f"\n{model_name.upper()} Test Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC: {auc_score:.4f}")
    print(f"  Avg Inference Time: {avg_inference_time*1000:.2f}ms per sample")
    
    return results

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name, save_path):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name.upper()}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Evaluate all models"""
    models = ['xception', 'efficientnet', 'resnet50']
    results = []
    
    # Create output directory
    os.makedirs('outputs/results', exist_ok=True)
    
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    for model_name in models:
        result = evaluate_model(model_name)
        if result is not None:
            results.append(result)
            
            # Plot confusion matrix
            plot_confusion_matrix(
                result['y_true'], 
                result['y_pred'], 
                model_name,
                f'outputs/results/{model_name}_confusion_matrix.png'
            )
            
            # Plot ROC curve
            plot_roc_curve(
                result['y_true'], 
                result['y_prob'], 
                model_name,
                f'outputs/results/{model_name}_roc_curve.png'
            )
    
    # Create summary table
    summary_data = []
    for result in results:
        summary_data.append({
            'Model': result['model'].upper(),
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1']:.4f}",
            'AUC': f"{result['auc']:.4f}",
            'Inference Time (ms)': f"{result['avg_inference_time']*1000:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('outputs/results/evaluation_summary.csv', index=False)
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    
    print(f"\nResults saved to outputs/results/")

if __name__ == "__main__":
    main()