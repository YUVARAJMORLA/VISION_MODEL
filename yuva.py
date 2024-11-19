import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Load the VGG16 model
model2 = load_model(r"vgg16.h5", compile=False)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=r"model.tflite")
interpreter.allocate_tensors()

# Function to run inference with the TFLite model
def predict_tflite(img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run the inference
    interpreter.invoke()

    # Get the result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data, axis=1)

# Function for preprocessing the image
def preprocess_image(file_path):
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.float32(img_array)  # Convert to float32 for TFLite model
    return img_array

# Create the "upload" folder if it does not exist
UPLOAD_FOLDER = "upload"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Index mapping
index = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

# Streamlit UI
st.title("Eye Disease Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload an eye image for prediction", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to the "upload" folder
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved to {file_path}")

    # Display the uploaded image
    st.image(file_path, caption="Uploaded Image", use_column_width=False)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Prediction buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Predict with EfficientNet (TFLite)"):
            pred = predict_tflite(img_array)
            st.write("Type of Eye Disease using EfficientNet (TFLite): ", index[pred[0]])

    with col2:
        if st.button("Predict with VGG16"):
            pred = np.argmax(model2.predict(img_array), axis=1)
            st.write("Type of Eye Disease using VGG16: ", index[pred[0]])
