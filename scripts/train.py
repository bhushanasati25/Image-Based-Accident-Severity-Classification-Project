import numpy as np
import os
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pickle

def load_data():
    X = np.load('data/processed/X_images.npy')
    y = np.load('data/processed/y_labels.npy')
    with open('data/processed/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    class_names = label_encoder.classes_
    return X, y, class_names

def train_model(model_name, X_train, y_train, X_val, y_val, class_names):
    if model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("Model not supported.")

    base_model.trainable = False  # Freeze the base model

    # Add custom layers on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define data generators
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )

    val_datagen = ImageDataGenerator()

    # Train the model
    model.fit(
        train_datagen.flow(X_train, y_train, batch_size=32),
        validation_data=val_datagen.flow(X_val, y_val),
        epochs=10
    )

    # Save the model
    os.makedirs('models/', exist_ok=True)
    model.save(f'models/{model_name.lower()}_model.h5')
    print(f"{model_name} model trained and saved.")

def main():
    X, y, class_names = load_data()

    # Split data into training, validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Train models
    for model_name in ['ResNet50', 'EfficientNetB0', 'MobileNetV2']:
        train_model(model_name, X_train, y_train, X_val, y_val, class_names)

if __name__ == "__main__":
    main()
