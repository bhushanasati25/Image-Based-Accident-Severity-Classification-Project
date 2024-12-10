import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import time
import base64
from io import BytesIO

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

# Function to add Google Fonts
def add_google_font():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body, div, span, applet, object, iframe,
            h1, h2, h3, h4, h5, h6, p, blockquote, pre,
            a, abbr, acronym, address, big, cite, code,
            del, dfn, em, img, ins, kbd, q, s, samp,
            small, strike, strong, sub, sup, tt, var,
            b, u, i, center,
            dl, dt, dd, ol, ul, li,
            fieldset, form, label, legend,
            table, caption, tbody, tfoot, thead, tr, th, td,
            article, aside, canvas, details, embed, 
            figure, figcaption, footer, header, hgroup, 
            menu, nav, output, ruby, section, summary,
            time, mark, audio, video {
                font-family: 'Roboto', sans-serif;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

add_google_font()

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

# Utility function to convert image to base64
def image_to_base64(img):
    """
    Convert PIL Image to base64 encoded string.
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    return img_base64

# Display top-k predictions using Plotly
def display_top_k(predictions, categories, k=2):
    """
    Display the top-k predictions with their confidence scores using Plotly.
    """
    top_k_indices = np.argsort(predictions)[::-1][:k]
    top_categories = [categories[i] for i in top_k_indices]
    top_scores = [float(predictions[i]) for i in top_k_indices]

    fig = go.Figure(go.Bar(
        x=top_scores,
        y=top_categories,
        orientation='h',
        marker=dict(color=['#1abc9c', '#3498db']),
        text=[f"{score:.2%}" for score in top_scores],
        textposition='auto'
    ))

    fig.update_layout(
        title="Top Predictions",
        xaxis=dict(title="Confidence Score", range=[0, 1]),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=100, r=50, t=50, b=50),
        template='plotly_white',
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

# Function to display the predicted class in a styled card
def display_predicted_class(predicted_class, confidence_score):
    """
    Display the predicted class and confidence score in a styled card.
    """
    # Define colors based on the predicted class
    if predicted_class.lower() == "human":
        color = "#1abc9c"  # Greenish
        icon = "👤"         # User icon
    else:
        color = "#3498db"  # Bluish
        icon = "🚗"         # Car icon

    # Create a styled HTML block
    html_content = f"""
    <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #ffffff, {color});
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        margin-top: 20px;
        width: 300px;
        margin-left: auto;
        margin-right: auto;
    ">
        <div style="
            font-size: 50px;
            margin-right: 20px;
        ">
            {icon}
        </div>
        <div style="
            text-align: left;
        ">
            <h2 style="
                margin: 0;
                font-family: 'Roboto', sans-serif;
                color: #2c3e50;
            ">Predicted Class</h2>
            <h1 style="
                margin: 5px 0;
                font-family: 'Roboto', sans-serif;
                color: {color};
            ">{predicted_class}</h1>
            <p style="
                margin: 0;
                font-size: 18px;
                color: #34495e;
            ">Confidence Score: {confidence_score:.2%}</p>
        </div>
    </div>
    """

    # Render the HTML content in Streamlit
    st.markdown(html_content, unsafe_allow_html=True)

# Function to display a team member's information
def display_team_member(name, role, image_path, linkedin, github):
    """
    Display a single team member's information in a styled card.
    """
    # Attempt to open the team member's image
    try:
        img = Image.open(image_path)
        # Resize the image to ensure uniformity
        img = img.resize((200, 200))
    except FileNotFoundError:
        st.error(f"❌ Image not found: {image_path}")
        return

    # Define custom CSS for circular images and card styling
    st.markdown(
        f"""
        <style>
        .team-card {{
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 20px;
            text-align: center;
            transition: transform 0.3s;
            width: 250px;
            margin: auto;
        }}
        .team-card:hover {{
            transform: scale(1.05);
        }}
        .team-image {{
            border-radius: 50%;
            width: 150px;
            height: 150px;
            object-fit: cover;
            margin-bottom: 15px;
        }}
        .team-name {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .team-role {{
            font-size: 16px;
            color: #7f8c8d;
            margin-bottom: 15px;
        }}
        .social-icons a {{
            margin: 0 10px;
            text-decoration: none;
        }}
        .social-icons img {{
            width: 30px;
            height: 30px;
        }}
        </style>
        <div class="team-card">
            <img src="data:image/png;base64,{image_to_base64(img)}" alt="{name}" class="team-image">
            <div class="team-name">{name}</div>
            <div class="team-role">{role}</div>
            <div class="social-icons">
                <a href="{linkedin}" target="_blank">
                    <img src="https://img.icons8.com/color/48/000000/linkedin.png" alt="LinkedIn">
                </a>
                <a href="{github}" target="_blank">
                    <img src="https://img.icons8.com/color/48/000000/github.png" alt="GitHub">
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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
        "assets/autonomous_driving.jpg",  # Ensure this image exists in the assets folder
        caption="Autonomous Driving",
        use_container_width=True,
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
        # Display uploaded image
        st.subheader("🖼️ Uploaded Image")
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # Custom CSS for the Predict button
        st.markdown(
            """
            <style>
            div.stButton > button:first-child {
                background-color: #1abc9c;
                color: white;
                height: 3em;
                width: 15em;
                border-radius:10px;
                border:1px solid #1abc9c;
                font-size:16px;
            }
            div.stButton > button:first-child:hover {
                background-color: #16a085;
                border:1px solid #16a085;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # "Predict" Button
        if st.button("🧠 Predict"):
            try:
                # Initialize progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Simulate a loading process
                for percent_complete in range(100):
                    time.sleep(0.02)  # Adjust the sleep time for faster or slower progress
                    progress_bar.progress(percent_complete + 1)
                    status_text.text(f"Processing... {percent_complete + 1}%")

                status_text.text("Processing... 100%")
                progress_bar.empty()
                status_text.empty()

                # Preprocess and predict after loading
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
                confidence_score = float(predictions[predicted_index])  # Convert to Python float

                # Display Top-k Predictions with Plotly Bar Chart
                display_top_k(predictions, CATEGORIES, k=2)

                # Highlight the Predicted Class using the stylish display
                display_predicted_class(predicted_class, confidence_score)

                # Display an Interactive Pie Chart for Confidence Scores
                st.markdown("### 📊 Confidence Distribution:")
                fig_pie = go.Figure(data=[go.Pie(labels=CATEGORIES, values=[float(score) for score in predictions],
                                                 hole=.3, marker=dict(colors=['#1abc9c', '#3498db']))])

                fig_pie.update_layout(
                    title_text='Confidence Distribution',
                    annotations=[dict(text='Confidence', x=0.5, y=0.5, font_size=20, showarrow=False)],
                    template='plotly_white'
                )

                st.plotly_chart(fig_pie, use_container_width=True)

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
    st.header("Meet Our Team")
    # Removed the subheader "Our Dedicated Professionals"

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
            "role": "Data Scientist",
            "image": "assets/Rujuta.jpg", 
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
            "role": "Data Scientist",
            "image": "assets/Anirudha.jpg",  
            "linkedin": "https://www.linkedin.com/in/anirudhasharma/",
            "github": "https://github.com/anirudhasharma",
        },
    ]

    # Define the number of columns based on desired layout
    num_columns = len(team_members)  # Set to the number of team members to display in one row
    cols = st.columns(num_columns)

    for idx, member in enumerate(team_members):
        col = cols[idx % num_columns]
        with col:
            display_team_member(
                name=member["name"],
                role=member["role"],
                image_path=member["image"],
                linkedin=member["linkedin"],
                github=member["github"]
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
        Developed By <strong>RED Coders</strong> !!
    </div>
    """,
    unsafe_allow_html=True,
)


