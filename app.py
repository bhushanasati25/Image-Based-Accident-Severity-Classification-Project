import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Define the TensorFlow Lite model path
TFLITE_MODEL_PATH = os.path.join('models', '/Users/ihack-pc/Desktop/KDD Project/Multiclass Object Classification in Autonomous Driving using the KITTI 3D Object Detection Dataset/densenet121_model (1).h5')

# Load the TensorFlow Lite model
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Preprocess the uploaded image
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert('RGB')  # Ensure the image is in RGB mode
    image = image.resize(target_size)  # Resize the image
    image_array = np.array(image).astype(np.float32) / 255.0  # Normalize the image
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Predict the class of the uploaded image
def predict_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ensure the image matches the expected input shape
    if image.shape != tuple(input_details[0]['shape']):
        raise ValueError(f"Expected input shape {input_details[0]['shape']}, but got {image.shape}")

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions

def main():
    st.title("Multiclass Object Classification")
    st.write("Upload an image to classify using the TensorFlow Lite model.")

    # Upload the image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Preprocess the image
            processed_image = preprocess_image(image)

            # Load the TFLite model
            interpreter = load_tflite_model(TFLITE_MODEL_PATH)

            # Perform prediction
            with st.spinner('Classifying...'):
                predictions = predict_image(interpreter, processed_image)

            # Define class names
            class_names = ['Car', 'Cyclist', 'Large_Vehicle', 'Miscellaneous', 'Pedestrian']

            # Display predictions
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            st.write(f"**Predicted Class:** {class_names[predicted_class]}")
            st.write(f"**Confidence:** {confidence:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
