import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Fruit Classification",
    page_icon="🍎",
    layout="centered"
)


# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "models/fruit_mobilenetv3.keras",
        compile=False
    )
    return model


# --------------------------------------------------
# Load Class Names
# --------------------------------------------------

@st.cache_data
def load_classes():
    with open("models/class_names.json", "r") as file:
        classes = json.load(file)

    return classes


# --------------------------------------------------
# Load Model and Classes
# --------------------------------------------------

model = load_model()
class_names = load_classes()


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("🍎 Fruit Classification")

st.write(
    "Upload a fruit image and MobileNetV3 "
    "will predict the fruit."
)

st.divider()


# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded Fruit Image",
        width=400
    )

    # Resize image
    image_resized = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(
        image_resized,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Make prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    # Get predicted class index
    predicted_index = np.argmax(prediction[0])

    # Get predicted class
    if isinstance(class_names, dict):
        predicted_class = class_names[str(predicted_index)]
    else:
        predicted_class = class_names[predicted_index]

    # Get confidence
    confidence = prediction[0][predicted_index] * 100

    # Display prediction
    st.success(
        f"🍓 Prediction: {predicted_class}"
    )

    # Display confidence
    st.info(
        f"🎯 Confidence: {confidence:.2f}%"
    )