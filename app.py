import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
model = load_model('my_model.h5')  # Ensure the model file is named correctly and in the same directory

# Set up the navigation menu
st.sidebar.title("Data Doppler")
page = st.sidebar.selectbox("Navigate", ["Home", "Dataset", "Contact"])

# Define a function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize to [0, 1] range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Home Page
if page == "Home":
    st.image("logo.png", width=300)  # Load a large logo image
    st.title("Data Doppler")
    st.header("Exploring Rare Disease Image Classification")
    st.write(
        """
        Welcome to **Data Doppler**, a machine learning-based tool for identifying and generating rare disease images using Convolutional Neural Networks (CNNs).
        Developed by a team of students at GHRCE College, Nagpur, this project leverages state-of-the-art deep learning and image processing technology.
        
        **Tech Stack:**
        - **TensorFlow and Keras** for deep learning model building.
        - **Streamlit** for seamless web application integration.
        - **Python** for efficient backend scripting and preprocessing.
        
        We invite you to explore this tool, which can classify disease images with high accuracy and generate augmented versions upon request!
        """
    )

# Dataset Page
elif page == "Dataset":
    st.title("Dataset")
    st.write("Upload an image to classify and generate augmented samples:")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    num_images = st.number_input("How many similar images would you like to generate?", min_value=1, max_value=10, value=3, step=1)

    if uploaded_file is not None:
        # Preprocess and predict
        processed_img = preprocess_image(uploaded_file)
        st.image(processed_img[0], caption="Uploaded Image", use_column_width=True)

        # Model prediction
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        st.write(f"Predicted Class: {'Benign' if predicted_class == 0 else 'Malignant'}")

        # Augmentations using ImageDataGenerator
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        data_gen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        st.write(f"Generating {num_images} augmented images:")
        generated_images = []
        for i in range(num_images):
            img_iterator = data_gen.flow(processed_img, batch_size=1)
            augmented_img = next(img_iterator)[0]
            generated_images.append(augmented_img)

        st.image(generated_images, width=100, caption=[f"Augmented {i+1}" for i in range(num_images)])

# Contact Page
elif page == "Contact":
    st.title("Contact Us")
    st.write(
        """
        **Address**: GHRCE College, Nagpur  
        **Phone**: 7263049920  
        **Team Members**:  
        - Prashant  
        - Shreerang  
        - Mahek  
        - Yashaswi
        """
    )

