import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# Load test data
X_test = np.load("data/processed/X_images.npy")
y_test = np.load("data/processed/y_labels.npy")

# Class names
with open("data/processed/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
class_names = label_encoder.classes_

def evaluate_model(model_path, model_name):
    print(f"Evaluating {model_name}")
    model = load_model(model_path)

    # Predict on test data
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Evaluate each model
model_paths = {
    "ResNet-50": "models/resnet50_model.h5",
    "EfficientNetB0": "models/efficientnetb0_model.h5",
    "MobileNetV2": "models/mobilenetv2_model.h5",
    "DenseNet121": "models/densenet121_model.h5",
    "InceptionV3": "models/inceptionv3_model.h5",
    "Vision Transformer (ViT)": "models/vit_model.h5",
    "Fine-tuned MobileNetV2": "models/fine_tuned_mobilenetv2.h5",
    "Fine-tuned MobileNetV2 (Focal)": "models/fine_tuned_mobilenetv2_focal.h5",
    "Final DenseNet121": "models/final_densenet121_model.h5",
}

for model_name, model_path in model_paths.items():
    evaluate_model(model_path, model_name)
