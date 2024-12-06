import os
import numpy as np
import pandas as pd
import cv2
import zipfile
from sklearn.preprocessing import LabelEncoder
import pickle

def download_dataset():
    # Check if kaggle.json exists
    if not os.path.exists('kaggle.json'):
        raise FileNotFoundError("kaggle.json not found. Please place it in the project root directory.")

    # Move kaggle.json to the correct location
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
        os.system('cp kaggle.json ~/.kaggle/')
        os.system('chmod 600 ~/.kaggle/kaggle.json')

    # Download dataset
    dataset_name = 'garymk/kitti-3d-object-detection-dataset'
    dataset_path = 'data/raw/'
    os.makedirs(dataset_path, exist_ok=True)
    os.system(f'kaggle datasets download -d {dataset_name} -p {dataset_path}')

    # Unzip dataset
    zip_path = os.path.join(dataset_path, 'kitti-3d-object-detection-dataset.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    print("Dataset downloaded and extracted.")

def load_and_preprocess_images():
    # Define paths
    image_dir = os.path.join('data/raw', 'training/image_2/')
    label_dir = os.path.join('data/raw', 'training/label_2/')

    # Get image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    # Class mapping
    class_mapping = {
        'Pedestrian': 'Pedestrian',
        'Person_sitting': 'Pedestrian',
        'Cyclist': 'Cyclist',
        'Car': 'Car',
        'Truck': 'Large_Vehicle',
        'Van': 'Large_Vehicle',
        'Tram': 'Large_Vehicle',
        'Misc': 'Miscellaneous',
        'DontCare': None  # Exclude 'DontCare' labels
    }

    images = []
    labels = []

    def map_label(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            mapped_classes = []
            for line in lines:
                data = line.strip().split()
                obj_class = data[0]
                mapped_class = class_mapping.get(obj_class)
                if mapped_class is not None:
                    mapped_classes.append(mapped_class)
            return mapped_classes

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace('.png', '.txt'))

        # Read and preprocess the image
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (224, 224))  # Resize to 224x224
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        images.append(image_rgb)

        # Read and map labels
        mapped_classes = map_label(label_path)
        if mapped_classes:
            # For simplicity, we use the first object's class
            labels.append(mapped_classes[0])
        else:
            labels.append('Miscellaneous')  # Default to 'Miscellaneous' if no labels

    # Convert lists to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Normalize images
    X = X / 255.0

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_

    # Save processed data
    os.makedirs('data/processed/', exist_ok=True)
    np.save('data/processed/X_images.npy', X)
    np.save('data/processed/y_labels.npy', y_encoded)
    with open('data/processed/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    print("Data preprocessing complete.")

if __name__ == "__main__":
    download_dataset()
    load_and_preprocess_images()
