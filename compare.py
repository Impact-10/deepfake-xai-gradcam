import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from evaluate import evaluate_model
from utils import get_model


def load_training_results():
    """Load training results from checkpoints."""
    models = ['xception', 'efficientnet', 'resnet50']
    training_results = []

    for model_name in models:
        checkpoint_path = f'checkpoints/{model_name}_best.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            training_results.append(
                {
                    'model': model_name,
                    'best_val_acc': checkpoint.get('val_acc', 0.0),
                    'epoch': checkpoint.get('epoch', 0)
                }
            )

    return training_results


def create_comparison_plots(comparison_df):
    """Create comparison plots."""
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Metrics comparison bar plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']

    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3

        ax = axes[row, col]
        bars = ax.bar(comparison_df['Model'], comparison_df[metric].astype(float))
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')

        plt.setp(ax.get_xticklabels(), rotation=45)

    # 2. Inference time comparison
    ax = axes[1, 2]
    inference_times = comparison_df['Inference Time (ms)'].astype(float)
    bars = ax.bar(comparison_df['Model'], inference_times)
    ax.set_title('Inference Time Comparison')
    ax.set_ylabel('Time (ms)')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')

    plt.setp(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig('outputs/results/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Radar chart for comprehensive comparison
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    metrics_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, (_, row) in enumerate(comparison_df.iterrows()):
        values = [float(row[metric]) for metric in metrics_radar]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_radar)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax.grid(True)

    plt.savefig('outputs/results/radar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_detailed_comparison_table():
    """Create detailed comparison with additional metrics."""
    models = ['xception', 'efficientnet', 'resnet50']

    print("Generating detailed comparison...")

    # Get evaluation results for all models
    detailed_results = []
    for model_name in models:
        result = evaluate_model(model_name)
        if result is not None:
            detailed_results.append(
                {
                    'Model': model_name.upper(),
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1-Score': result['f1'],
                    'AUC': result['auc'],
                    'Test Loss': result['test_loss'],
                    'Inference Time (ms)': result['avg_inference_time'] * 1000,
                    'Parameters': get_model_parameters(model_name)
                }
            )

    return pd.DataFrame(detailed_results)


def get_model_parameters(model_name):
    """Get number of parameters for each model."""
    model = get_model(model_name)
    total_params = sum(p.numel() for p in model.parameters())
    return f"{total_params:,}"


def main():
    """Create comprehensive model comparison."""
    os.makedirs('outputs/results', exist_ok=True)

    print("=" * 60)
    print("MODEL COMPARISON ANALYSIS")
    print("=" * 60)

    # Load evaluation results
    if os.path.exists('outputs/results/evaluation_summary.csv'):
        comparison_df = pd.read_csv('outputs/results/evaluation_summary.csv')
    else:
        print("Running evaluation first...")
        from evaluate import main as evaluate_main
        evaluate_main()
        comparison_df = pd.read_csv('outputs/results/evaluation_summary.csv')

    # Create detailed comparison
    detailed_df = create_detailed_comparison_table()
    detailed_df.to_csv('outputs/results/detailed_comparison.csv', index=False)

    # Create comparison plots
    create_comparison_plots(comparison_df)

    # Print rankings
    print("\n" + "=" * 60)
    print("MODEL RANKINGS")
    print("=" * 60)

    # Rank by different metrics
    metrics_to_rank = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']

    for metric in metrics_to_rank:
        if metric in comparison_df.columns:
            ranked = comparison_df.sort_values(metric, ascending=False)
            print(f"\n{metric} Ranking:")
            for i, (_, row) in enumerate(ranked.iterrows(), 1):
                print(f"  {i}. {row['Model']}: {row[metric]}")

    # Overall ranking (average of normalized scores)
    print(f"\nOverall Performance Ranking:")
    scoring_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']

    # Normalize scores
    normalized_df = comparison_df.copy()
    for metric in scoring_metrics:
        if metric in normalized_df.columns:
            values = normalized_df[metric].astype(float)
            normalized_df[f'{metric}_norm'] = values / values.max()

    # Calculate average score
    norm_cols = [f'{metric}_norm' for metric in scoring_metrics if f'{metric}_norm' in normalized_df.columns]
    normalized_df['Overall_Score'] = normalized_df[norm_cols].mean(axis=1)

    overall_ranking = normalized_df.sort_values('Overall_Score', ascending=False)
    for i, (_, row) in enumerate(overall_ranking.iterrows(), 1):
        print(f"  {i}. {row['Model']}: {row['Overall_Score']:.4f}")

    # Best model for different use cases
    print(f"\n" + "=" * 40)
    print("RECOMMENDATIONS")
    print("=" * 40)

    best_accuracy = comparison_df.loc[comparison_df['Accuracy'].astype(float).idxmax(), 'Model']
    best_speed = comparison_df.loc[comparison_df['Inference Time (ms)'].astype(float).idxmin(), 'Model']
    best_precision = comparison_df.loc[comparison_df['Precision'].astype(float).idxmax(), 'Model']

    print(f"Best Overall Accuracy: {best_accuracy}")
    print(f"Fastest Inference: {best_speed}")
    print(f"Best Precision (fewer false positives): {best_precision}")

    print(f"\nComparison results saved to outputs/results/")


if __name__ == "__main__":
    main()