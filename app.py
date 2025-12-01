import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------------------------------
# Load your trained model
# -------------------------------------------------------
model = tf.keras.models.load_model("brain_tumor_model_v2.keras")

# Class names (edit them based on your dataset)
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI", use_column_width=True)

    # Preprocess
    img = img.convert("RGB")               # ensure 3 channels
    img = img.resize((224, 224))           # resize
    img_array = np.array(img) / 255.0      # normalize
    img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)

    # Prediction
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    pred_class = class_names[pred_index]
    confidence = np.max(prediction) * 100

    # Output
    st.subheader("🔍 Prediction Result")
    st.write(f"**Tumor Type:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
