import os
import argparse
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import json
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract_kitti(api, dataset_name, download_path):
    if not os.path.exists(download_path):
        os.makedirs(download_path, exist_ok=True)
        print("Downloading KITTI dataset...")
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already exists. Skipping download.")

def map_label(label_path, class_mapping):
    mapped_classes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split()
            obj_class = data[0]
            mapped_class = class_mapping.get(obj_class)
            if mapped_class is not None:
                mapped_classes.append(mapped_class)
    return mapped_classes

def preprocess_data(raw_image_dir, raw_label_dir, processed_dir, target_size=(224,224)):
    class_mapping = {
        'Pedestrian': 'Human',
        'Person_sitting': 'Human',
        'Cyclist': 'Human',
        'Car': 'Vehicle',
        'Truck': 'Vehicle',
        'Van': 'Vehicle',
        'Tram': 'Vehicle',
        'Misc': None,
        'DontCare': None
    }

    images = []
    labels = []
    image_files = [f for f in os.listdir(raw_image_dir) if f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(raw_image_dir, image_file)
        label_path = os.path.join(raw_label_dir, image_file.replace('.png', '.txt'))
        mapped_classes = map_label(label_path, class_mapping)
        if mapped_classes:
            label = mapped_classes[0]  # Taking the first valid object
            image = cv2.imread(image_path)
            if image is None:
                continue  # Skip if image is not readable
            image_resized = cv2.resize(image, target_size)
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            images.append(image_rgb)
            labels.append(label)
    
    X = np.array(images)
    y = np.array(labels)
    X = X / 255.0  # Normalize to [0,1]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_)
    print("Reduced Categories:", class_names)

    # Split into train, validation, test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Create processed_dir/train, processed_dir/val, processed_dir/test
    splits = {'train': (X_train, y_train),
              'val': (X_val, y_val),
              'test': (X_test, y_test)}

    for split, (X_split, y_split) in splits.items():
        split_dir = os.path.join(processed_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        np.save(os.path.join(split_dir, 'X.npy'), X_split)
        np.save(os.path.join(split_dir, 'y.npy'), y_split)
        print(f"Saved {split} set: {X_split.shape[0]} samples.")
    
    # Save label encoder
    with open(os.path.join(processed_dir, 'label_encoder.json'), 'w') as f:
        json.dump({'classes': class_names}, f)
    
    print("Preprocessing completed.")

def main():
    parser = argparse.ArgumentParser(description='Preprocess KITTI dataset for object classification.')
    parser.add_argument('--dataset_name', type=str, default='garymk/kitti-3d-object-detection-dataset', help='Kaggle dataset name')
    parser.add_argument('--download_path', type=str, default='data/raw/', help='Path to download raw data')
    parser.add_argument('--processed_dir', type=str, default='data/processed/', help='Path to save processed data')
    parser.add_argument('--api_credentials', type=str, default='kaggle.json', help='Path to kaggle.json')
    
    args = parser.parse_args()

    # Setup Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download and extract dataset
    download_and_extract_kitti(api, args.dataset_name, args.download_path)
    
    # Define raw image and label directories
    raw_image_dir = os.path.join(args.download_path, 'training', 'image_2')
    raw_label_dir = os.path.join(args.download_path, 'training', 'label_2')
    
    # Preprocess data
    preprocess_data(raw_image_dir, raw_label_dir, args.processed_dir)

if __name__ == '__main__':
    main()
