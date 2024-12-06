import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Multiclass Object Classification in Autonomous Driving using the KITTI 3D Object Detection Dataset",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
CATEGORIES = ["Human", "Vehicle"]  # Ensure this order matches model's output
MODEL_PATH = "models/fine_tuned_densenet121_saved_model"  # TensorFlow Serving model path
CONFIDENCE_THRESHOLD = 0.5  # Adjust based on desired sensitivity

# Load model using TFSMLayer
@st.cache_resource
def load_model():
    return TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

model = load_model()

# Preprocess image
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the uploaded image to match the model's expected input.
    """
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Display top-k predictions
def display_top_k(predictions, categories, k=2):
    """
    Display the top-k predictions with their confidence scores.
    """
    st.markdown("### Top Predictions:")
    top_k_indices = np.argsort(predictions)[::-1][:k]
    for i in top_k_indices:
        st.write(f"- **{categories[i]}**: {predictions[i]:.2%}")

# Sidebar
st.sidebar.title("About This App")
st.sidebar.write(
    """
    This app uses a fine-tuned DenseNet121 model to classify images into two categories:
    - **Human**: Pedestrian, Person_sitting, Cyclist
    - **Vehicle**: Car, Truck, Van, Tram
    """
)
st.sidebar.info("Powered by TensorFlow and Streamlit.")

# Main layout
st.title("Multiclass Object Classification in Autonomous Driving using the KITTI 3D Object Detection Dataset")
st.write("Upload an image to classify it as either 'Human' or 'Vehicle'.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"], label_visibility="visible"
)

if uploaded_file:
    try:
        # Display uploaded image
        st.subheader("Uploaded Image")
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # Preprocess and predict
        st.subheader("Prediction")
        with st.spinner("Analyzing the image..."):
            processed_image = preprocess_image(uploaded_image)
            output = model(processed_image)  # Get model output

            # Handle different output formats
            if isinstance(output, tf.Tensor):
                predictions = output.numpy()[0]
            elif isinstance(output, dict):
                # Assume the first key corresponds to the output layer
                output_key = list(output.keys())[0]
                predictions = output[output_key].numpy()[0]
            else:
                st.error("Unexpected model output format.")
                st.stop()

            # Debug: Display model output shape and values (optional)
            # st.write(f"Model Output Shape: {predictions.shape}")
            # st.write(f"Model Predictions: {predictions}")

            # Determine the predicted class
            predicted_index = np.argmax(predictions)
            predicted_class = CATEGORIES[predicted_index]
            confidence_score = predictions[predicted_index]

        # Display top-k predictions
        display_top_k(predictions, CATEGORIES, k=2)

        # Display the most confident result with threshold check
        if confidence_score < CONFIDENCE_THRESHOLD:
            st.warning(
                "Model is not confident about the top prediction. It may not belong to the known categories."
            )
            st.markdown("### Confidence Scores (All Categories):")
            for category, score in zip(CATEGORIES, predictions):
                st.write(f"- **{category}**: {score:.2%}")
        else:
            st.markdown(
                f"""
                ### Predicted Class: **{predicted_class}**
                **Confidence Score:** {confidence_score:.2%}
                """
            )

        # Debugging information (optional)
        debug_mode = st.sidebar.checkbox("Enable Debug Mode")
        if debug_mode:
            st.write("### Debug Information")
            st.write(f"Raw Predictions: {predictions}")
            st.write("Confidence Scores for Each Category:")
            for category, score in zip(CATEGORIES, predictions):
                st.write(f"- {category}: {score:.2%}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
else:
    st.info("Please upload an image to see predictions.")

# Footer
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        font-size: 12px;
        margin-top: 20px;
    }
    </style>
    <div class="footer">
        Developed By RED Coder !! ❤️ 
    </div>
    """,
    unsafe_allow_html=True,
)
