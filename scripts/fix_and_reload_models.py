import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import h5py

# Path to the directory containing your models
MODELS_DIR = "/Users/ihack-pc/Desktop/KDD Project/Multiclass Object Classification in Autonomous Driving using the KITTI 3D Object Detection Dataset/models/"
UPDATED_MODELS_DIR = "models/updated/"

# Ensure the updated models directory exists
os.makedirs(UPDATED_MODELS_DIR, exist_ok=True)

# List of model files to process
model_files = [
    "resnet50_model.h5",
    "efficientnetb0_model.h5",
    "mobilenetv2_model.h5",
    "densenet121_model.h5",
    "inceptionv3_model.h5",
    "vit_model.h5",
    "fine_tuned_mobilenetv2.h5",
    "fine_tuned_mobilenetv2_focal.h5",
    "final_densenet121_model.h5"
]

def fix_and_save_model(model_path, updated_model_path):
    """
    Load the model, handle deserialization issues, and save the updated model.
    """
    try:
        print(f"Processing model: {model_path}")
        # Load the model with custom object handling for BatchNormalization
        model = load_model(
            model_path,
            custom_objects={'BatchNormalization': tf.keras.layers.BatchNormalization}
        )

        # Save the model in HDF5 format
        model.save(updated_model_path)
        print(f"Model updated and saved to: {updated_model_path}")

        # Convert and save in SavedModel format for compatibility
        saved_model_dir = updated_model_path.replace(".h5", "_saved_model")
        model.save(saved_model_dir, save_format='tf')
        print(f"SavedModel format saved to: {saved_model_dir}")

    except Exception as e:
        print(f"Failed to process model {model_path}: {e}")
        # Debugging the issue by loading layer configurations
        with h5py.File(model_path, 'r') as f:
            if 'model_config' in f.attrs:
                model_config = f.attrs['model_config']
                print(f"Model config for {model_path}:")
                print(model_config)
            else:
                print(f"No model_config found for {model_path}")

# Iterate through each model file
for model_file in model_files:
    model_path = os.path.join(MODELS_DIR, model_file)
    updated_model_path = os.path.join(UPDATED_MODELS_DIR, model_file)
    fix_and_save_model(model_path, updated_model_path)

print("All models processed.")
