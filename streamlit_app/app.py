import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Multiclass Object Classification in Autonomous Driving",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit default menu and footer for a cleaner look
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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

# Create Tabs for Navigation
tabs = st.tabs(["🏠 Home", "ℹ️ About", "🔍 Model Status", "📸 Prediction", "👥 Team"])

# Home Tab
with tabs[0]:
    st.title("🚗 Multiclass Object Classification in Autonomous Driving")
    st.markdown(
        """
        Welcome to the **Multiclass Object Classification** app tailored for autonomous driving systems. 
        This application leverages a fine-tuned DenseNet121 model to accurately classify objects in images 
        as either **Human** or **Vehicle**. Upload an image to get started and see how our model performs!
        """
    )
    st.image(
        "https://images.unsplash.com/photo-1518770660439-4636190af475?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80",
        caption="Autonomous Driving",
        use_container_width=True,  # Updated parameter
    )

# About Tab
with tabs[1]:
    st.title("📖 About This App")
    st.markdown(
        """
        ### Purpose
        This application is designed to assist in the development of autonomous driving systems by providing 
        accurate object classification. By distinguishing between humans and various vehicles, the system can 
        make informed decisions to enhance safety and efficiency on the roads.

        ### Features
        - **Image Classification**: Classify uploaded images into 'Human' or 'Vehicle' categories.
        - **Model Integration**: Utilizes a fine-tuned DenseNet121 model served via TensorFlow Serving.
        - **User-Friendly Interface**: Intuitive design for easy navigation and interaction.

        ### Technologies Used
        - **Streamlit**: For building the interactive web interface.
        - **TensorFlow**: For model development and serving.
        - **Pillow (PIL)**: For image processing.
        - **NumPy**: For numerical operations.
        """
    )

# Model Status Tab
with tabs[2]:
    st.title("🔍 Model Status")
    try:
        if model:
            st.success("✅ Model loaded successfully!")
            st.info(f"**Model Path:** {MODEL_PATH}")
            st.info(f"**Categories:** {', '.join(CATEGORIES)}")
            st.info(f"**Confidence Threshold:** {CONFIDENCE_THRESHOLD * 100:.0f}%")
        else:
            st.error("❌ Model failed to load.")
    except Exception as e:
        st.error(f"❌ An error occurred while loading the model: {e}")

# Prediction Tab
with tabs[3]:
    st.title("📸 Prediction")
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
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)  # Updated parameter

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

                # Determine the predicted class
                predicted_index = np.argmax(predictions)
                predicted_class = CATEGORIES[predicted_index]
                confidence_score = predictions[predicted_index]

            # Display top-k predictions
            display_top_k(predictions, CATEGORIES, k=2)

            # Display the most confident result with threshold check
            if confidence_score < CONFIDENCE_THRESHOLD:
                st.warning(
                    "⚠️ Model is not confident about the top prediction. It may not belong to the known categories."
                )
                st.markdown("### Confidence Scores (All Categories):")
                for category, score in zip(CATEGORIES, predictions):
                    st.write(f"- **{category}**: {score:.2%}")
            else:
                st.markdown(
                    f"""
                    ### 🏆 Predicted Class: **{predicted_class}**
                    **Confidence Score:** {confidence_score:.2%}
                    """
                )

            # Debugging information (optional)
            debug_mode = st.checkbox("🔍 Enable Debug Mode")
            if debug_mode:
                st.write("### 🛠️ Debug Information")
                st.write(f"**Raw Predictions:** {predictions}")
                st.write("**Confidence Scores for Each Category:**")
                for category, score in zip(CATEGORIES, predictions):
                    st.write(f"- {category}: {score:.2%}")

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")
    else:
        st.info("ℹ️ Please upload an image to see predictions.")

# Team Tab
with tabs[4]:
    st.title("👥 Meet the Team")
    st.markdown("### Our Dedicated Team Members")

    # Define team members' information
    team_members = [
        {
            "name": "Bhushan Asati",
            "role": "Data Scientist",
            "image": "assets/Bhushan.jpg",
            "linkedin": "https://www.linkedin.com/in/bhushanasati25/",
            "github": "https://github.com/bhushanasati25",
        },
        {
            "name": "Rujuta Dabke",
            "role": "Lead Developer",
            "image": "https://via.placeholder.com/150",
            "linkedin": "https://www.linkedin.com/in/rujuta-dabke/",
            "github": "https://github.com/RujutaDabke",
        },
        {
            "name": "Suyash Madhavi",
            "role": "Data Scientist",
            "image": "assets/Suyash.jpeg",
            "linkedin": "https://www.linkedin.com/in/suyash-madhavi-24260922a",
            "github": "https://github.com/SUYASH-a17",
        },
        {
            "name": "Anirudha Sharma",
            "role": "UI/UX Designer",
            "image": "https://via.placeholder.com/150",
            "linkedin": "https://www.linkedin.com/in/davidbrown",
            "github": "https://github.com/davidbrown",
        },
    ]

    # Display team members in a grid
    cols = st.columns(4)
    for idx, member in enumerate(team_members):
        with cols[idx % 4]:
            st.image(member["image"], width=150, use_container_width=True)  # Updated parameter
            st.markdown(f"**{member['name']}**")
            st.write(member["role"])
            st.markdown(
                f"""
                [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white)]({member['linkedin']})
                [![GitHub](https://img.shields.io/badge/GitHub-black?logo=github&logoColor=white)]({member['github']})
                """
            )

# Footer (appears on all tabs)
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
        Developed By <strong>RED Coders</strong> !! ❤️ 
    </div>
    """,
    unsafe_allow_html=True,
)
