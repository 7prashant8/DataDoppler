import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model('model.h5')  # Ensure model.h5 is in the same directory

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((500, 500))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize image
    return img

# Streamlit page configuration
st.set_page_config(page_title="Data Doppler", page_icon="🌀", layout="centered")

# Sidebar Navigation
st.sidebar.title("Data Doppler")
page = st.sidebar.selectbox("Navigation", ["Home", "Dataset", "Contact"])

# Home Page
if page == "Home":
    st.image("logo.png", width=100)  # Logo image
    st.title("Data Doppler")
    st.write("""
        Welcome to Data Doppler! This web application is designed to classify images using a Convolutional Neural Network (CNN) model.
        **Tech Stack:** Streamlit, TensorFlow, Python
        **Technology:** CNN (Convolutional Neural Network)
    """)

# Dataset Page
elif page == "Dataset":
    st.header("Upload and Classify an Image")
    num_images = st.number_input("How many images to generate?", min_value=1, max_value=10, step=1)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        if st.button("Generate and Classify"):
            processed_img = preprocess_image(img)
            predictions = model.predict(processed_img)
            class_index = np.argmax(predictions[0])

            # Display the generated images
            st.write(f"Predicted Class: {class_index}")
            st.write(f"Confidence: {np.max(predictions[0]) * 100:.2f}%")

# Contact Page
elif page == "Contact":
    st.header("Contact Us")
    st.write("**Address:** GHRCE College, Nagpur")
    st.write("**Mobile Number:** 7263049920")
    st.write("**Team Members:** Prashant, Shreerang, Mahek, Yashaswi")
