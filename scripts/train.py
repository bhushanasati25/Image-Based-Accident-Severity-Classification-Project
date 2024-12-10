import os
import argparse
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121, ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(processed_dir):
    splits = ['train', 'val', 'test']
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

def get_class_weights(y_train, num_classes):
    counter = Counter(y_train)
    total = sum(counter.values())
    class_weights = {}
    for cls in range(num_classes):
        if cls in counter:
            class_weights[cls] = total / (num_classes * counter[cls])
        else:
            class_weights[cls] = 1.0
    print("Class weights:", class_weights)
    return class_weights

def create_generators(data, batch_size, train_aug=True):
    if train_aug:
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
    else:
        train_datagen = ImageDataGenerator()

    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow(data['train'][0], data['train'][1], batch_size=batch_size, shuffle=True)
    val_gen = val_datagen.flow(data['val'][0], data['val'][1], batch_size=batch_size, shuffle=False)
    test_gen = test_datagen.flow(data['test'][0], data['test'][1], batch_size=batch_size, shuffle=False)
    
    return train_gen, val_gen, test_gen

def build_model(base_model_fn, input_shape, num_classes):
    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True  # Fine-tune all layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(model, train_gen, val_gen, class_weights, learning_rate, epochs, model_save_path):
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights
    )
    
    # Save model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return history

def plot_history(history, model_name, save_dir):
    # Plot training & validation accuracy and loss
    plt.figure(figsize=(12,5))
    
    # Loss
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{model_name}_training_history.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Train multiple CNN models for object classification.')
    parser.add_argument('--processed_dir', type=str, default='data/processed/', help='Path to processed data')
    parser.add_argument('--models_dir', type=str, default='models/', help='Path to save trained models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    
    args = parser.parse_args()
    
    # Create models directory if not exists
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Load data
    data, class_names = load_data(args.processed_dir)
    num_classes = len(class_names)
    
    # Get class weights
    class_weights = get_class_weights(data['train'][1], num_classes)
    
    # Create data generators
    train_gen, val_gen, test_gen = create_generators(data, args.batch_size, train_aug=True)
    
    # Define models to train
    models_to_train = {
        'DenseNet121': DenseNet121,
        'ResNet50': ResNet50,
        'EfficientNetB0': EfficientNetB0,
        'MobileNetV2': MobileNetV2
    }
    
    input_shape = (224, 224, 3)
    
    for model_name, base_model_fn in models_to_train.items():
        print(f"\n===== Training {model_name} =====")
        model = build_model(base_model_fn, input_shape, num_classes)
        history = train_model(
            model,
            train_gen,
            val_gen,
            class_weights,
            args.learning_rate,
            args.epochs,
            os.path.join(args.models_dir, f"{model_name.lower()}_model.h5")
        )
        # Plot and save training history
        plot_history(history, model_name, args.models_dir)

if __name__ == '__main__':
    main()
