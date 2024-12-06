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

# Set up categories and model path
CATEGORIES = ["Car", "Cyclist", "Large Vehicle", "Miscellaneous", "Pedestrian"]
CONFIDENCE_THRESHOLD = 0.3
MODEL_PATH = "models/kdd_project_densenet121_clean_model"

# Load model
@st.cache_resource
def load_model():
    return TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

model = load_model()

# Function to preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize the image
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Sidebar information
st.sidebar.title("About This App")
st.sidebar.write(
    """
    This app uses a pre-trained DenseNet121 model to classify images into one of the following categories:
    - **Car**
    - **Cyclist**
    - **Large Vehicle**
    - **Miscellaneous**
    - **Pedestrian**
    
    If the model's confidence is low, it will indicate uncertainty. The model was not trained on all possible categories, such as bikes.
    """
)
st.sidebar.info("Powered by TensorFlow and Streamlit.")

# Main layout
st.title("🚗 Image Classification with DenseNet121")
st.write("Upload an image to classify it into one of the categories.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"], label_visibility="visible"
)

# Main content
if uploaded_file:
    st.subheader("Uploaded Image")
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    st.subheader("Prediction")
    with st.spinner("Analyzing the image..."):
        processed_image = preprocess_image(uploaded_image)
        predictions = model(processed_image)["output_layer"].numpy()[0]
        predicted_index = np.argmax(predictions)
        predicted_class = CATEGORIES[predicted_index]
        confidence_score = predictions[predicted_index]

    # Handle low confidence
    if confidence_score < CONFIDENCE_THRESHOLD:
        st.warning(
            "Model is not confident about the prediction. The image might not belong to the known categories."
        )
    else:
        # Display results
        st.markdown(
            f"""
            ### Predicted Class: **{predicted_class}**
            **Confidence Score:** {confidence_score:.2%}
            """
        )

    # Provide feedback
    st.markdown("---")
    st.write("### Feedback")
    feedback = st.text_area(
        "Do you agree with the prediction? If not, what category do you think this image belongs to?"
    )
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

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
        Built with ❤️ using TensorFlow and Streamlit.
    </div>
    """,
    unsafe_allow_html=True,
)


