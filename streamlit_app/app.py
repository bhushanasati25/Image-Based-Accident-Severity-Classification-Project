import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Image Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Updated categories (just two classes)
CATEGORIES = ["Human", "Vehicle"]
MODEL_PATH = "models/fine_tuned_densenet121_saved_model"
CONFIDENCE_THRESHOLD = 0.5  # Adjust if needed

# Load model using TFSMLayer
@st.cache_resource
def load_model():
    return TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

model = load_model()

def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def display_top_k(predictions, categories, k=2):
    st.markdown("### Top Predictions:")
    top_k_indices = np.argsort(predictions)[::-1][:k]
    for i in top_k_indices:
        st.write(f"- **{categories[i]}**: {predictions[i]:.2%}")

st.sidebar.title("About This App")
st.sidebar.write(
    """
    This app classifies images into two categories:
    - **Human**: Pedestrian, Person_sitting, Cyclist
    - **Vehicle**: Car, Truck, Van, Tram
    """
)
st.sidebar.info("Powered by TensorFlow and Streamlit.")

st.title("🚗 Image Classification with DenseNet121")
st.write("Upload an image to classify it as 'Human' or 'Vehicle'.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_file:
    st.subheader("Uploaded Image")
    uploaded_image = Image.open(uploaded_file).convert("RGB")
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    st.subheader("Prediction")
    with st.spinner("Analyzing the image..."):
        processed_image = preprocess_image(uploaded_image)
        output = model(processed_image)
        # Check if output is a direct tensor or a dict
        if isinstance(output, tf.Tensor):
            predictions = output.numpy()[0]
        else:
            # If output is a dict, get the first key
            output_key = list(output.keys())[0]
            predictions = output[output_key].numpy()[0]

        predicted_index = np.argmax(predictions)
        predicted_class = CATEGORIES[predicted_index]
        confidence_score = predictions[predicted_index]

    # Display top-k predictions
    display_top_k(predictions, CATEGORIES, k=2)

    # Check confidence threshold
    if confidence_score < CONFIDENCE_THRESHOLD:
        st.warning("Model is not fully confident about the prediction.")
    else:
        st.markdown(
            f"""
            ### Predicted Class: **{predicted_class}**
            **Confidence Score:** {confidence_score:.2%}
            """
        )

    # Debug mode
    debug_mode = st.sidebar.checkbox("Enable Debug Mode")
    if debug_mode:
        st.write("### Debug Information")
        st.write("Raw Predictions:", predictions)
        st.write("Confidence Scores for Each Category:")
        for category, score in zip(CATEGORIES, predictions):
            st.write(f"- {category}: {score:.2%}")

    st.markdown("---")
    st.write("### Feedback")
    feedback = st.text_area("Do you agree with the prediction?")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

else:
    st.info("Please upload an image to see predictions.")

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
        Developed By RED Coders !!
    </div>
    """,
    unsafe_allow_html=True,
)
