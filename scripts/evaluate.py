import os
import argparse
import numpy as np
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_data(processed_dir):
    splits = ['test']
    data = {}
    for split in splits:
        X = np.load(os.path.join(processed_dir, split, 'X.npy'))
        y = np.load(os.path.join(processed_dir, split, 'y.npy'))
        data[split] = (X, y)
        print(f"Loaded {split} set: {X.shape[0]} samples.")
    
    # Load label encoder
    with open(os.path.join(processed_dir, 'label_encoder.json'), 'r') as f:
        label_encoder = json.load(f)
    class_names = label_encoder['classes']
    return data, class_names

def evaluate_model(model_path, data, class_names):
    model = load_model(model_path)
    X_test, y_test = data['test']
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print(f"\nClassification Report for {os.path.basename(model_path)}:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f'Confusion Matrix - {os.path.basename(model_path)}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_plot_path = model_path.replace('.h5', '_confusion_matrix.png')
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_plot_path}")
    
    # Return metrics for comparison
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return {
        'model': os.path.basename(model_path),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained CNN models on test set.')
    parser.add_argument('--processed_dir', type=str, default='data/processed/', help='Path to processed data')
    parser.add_argument('--models_dir', type=str, default='models/', help='Path to trained models')
    parser.add_argument('--output_dir', type=str, default='models/evaluation_reports/', help='Path to save evaluation reports')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    data, class_names = load_data(args.processed_dir)
    
    # Get list of model files
    model_files = [f for f in os.listdir(args.models_dir) if f.endswith('.h5')]
    
    # Evaluate each model
    metrics = []
    for model_file in model_files:
        model_path = os.path.join(args.models_dir, model_file)
        metric = evaluate_model(model_path, data, class_names)
        metrics.append(metric)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_csv_path = os.path.join(args.output_dir, 'models_performance.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"All models' performance metrics saved to {metrics_csv_path}")
    
    # Plot comparison
    plt.figure(figsize=(10,6))
    sns.barplot(x='model', y='precision', data=metrics_df, label='Precision')
    sns.barplot(x='model', y='recall', data=metrics_df, label='Recall', alpha=0.6)
    sns.barplot(x='model', y='f1_score', data=metrics_df, label='F1-Score', alpha=0.4)
    plt.ylabel('Score')
    plt.title('Models Performance Comparison')
    plt.legend()
    comparison_plot_path = os.path.join(args.output_dir, 'models_performance_comparison.png')
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"Models performance comparison plot saved to {comparison_plot_path}")

if __name__ == '__main__':
    main()
