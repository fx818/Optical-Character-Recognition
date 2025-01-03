import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model("customModel.h5")

# Define the Streamlit app
st.title("Optical Character Recognition")
st.write("It can only recognize single character at a time so please provide a single character image.")

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

import pathlib
import numpy as np

train_dir = "kaggle/input/standard-ocr-dataset/data/training_data"

data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))



# Let the user choose the input method
input_method = st.radio("Choose input method", ("Upload Image", "Use Camera"))

# Initialize an image variable
image = None

if input_method == "Upload Image":
    # File uploader for images
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    

elif input_method == "Use Camera":
    # Camera input for capturing an image
    uploaded_image = st.camera_input("Take a picture")
    


if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # Preprocess the image
    st.write("Processing the image...")
    image_array = preprocess_image(uploaded_image)

    # Make prediction
    with st.spinner("Classifying the image..."):
        prediction = model.predict(image_array)

    import numpy as np
    np.max(prediction)
    idx = np.argmax(prediction)
    print(class_names[idx])

    # Display the prediction
    st.title(f"Output: {class_names[idx]}")
