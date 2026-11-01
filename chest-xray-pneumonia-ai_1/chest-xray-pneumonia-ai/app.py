"""Streamlit web app: upload a chest X-ray, get a prediction + Grad-CAM heatmap.

Run with:   streamlit run app.py
"""
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from gradcam import make_gradcam_heatmap, overlay_heatmap

MODEL_PATH = Path("models/pneumonia_model.keras")
IMG_SIZE = 224
THRESHOLD = 0.5

st.set_page_config(page_title="Chest X-ray Pneumonia Classifier", page_icon="🩻")
st.title("Chest X-ray Pneumonia Classifier")
st.caption("EfficientNetB0 (transfer learning) trained on PneumoniaMNIST, with Grad-CAM explanations.")
st.warning(
    "Research and education demo only. This is NOT a medical device and must not be "
    "used for diagnosis or treatment decisions."
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        f"Model file not found at `{MODEL_PATH}`. Train the model with the notebook "
        "first and put `pneumonia_model.keras` in the `models/` folder."
    )
    st.stop()

model = load_model()

uploaded = st.file_uploader("Upload a chest X-ray (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    # Same preprocessing as training: grayscale -> 224x224 -> 3 channels, values 0..255
    gray = Image.open(uploaded).convert("L").resize((IMG_SIZE, IMG_SIZE))
    gray_arr = np.array(gray)
    batch = np.stack([gray_arr] * 3, axis=-1).astype("float32")[np.newaxis]

    prob = float(model.predict(batch, verbose=0)[0, 0])
    label = "Pneumonia" if prob >= THRESHOLD else "Normal"

    st.subheader(f"Prediction: {label}")
    st.metric("Pneumonia probability", f"{prob:.1%}")
    st.progress(prob)

    heatmap = make_gradcam_heatmap(batch, model)
    overlay = overlay_heatmap(gray_arr, heatmap)

    col1, col2 = st.columns(2)
    col1.image(gray_arr, caption="Input X-ray", use_container_width=True)
    col2.image(overlay, caption="Grad-CAM: regions that raised the pneumonia score", use_container_width=True)

    with st.expander("How to read this"):
        st.write(
            "Red/yellow areas pushed the model towards 'Pneumonia'. "
            "If the hot spots sit outside the lungs (text labels, borders, devices), "
            "the model may be relying on shortcuts instead of real anatomy."
        )
