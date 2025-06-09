import streamlit as st
from PIL import Image
import numpy as np
import cv2
import base64
import os
from AppManager import AppManager  # Adjust import path as needed

# Initialize AppManager once
app_manager = AppManager()

# Must be first Streamlit command
st.set_page_config(page_title="Concentration Predictor", layout="centered")

# Background image setup
def set_bg(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        header {{
            background-color: rgba(0, 0, 0, 0) !important;
        }}
        [data-testid="stToolbar"] {{
            background-color: rgba(0, 0, 0, 0) !important;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.7) !important;
            border-radius: 15px;
            padding: 2rem;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .stTitle {{
            color: white !important;
        }}
        .stMarkdown {{
            color: white !important;
        }}
        .uploadedFile {{
            background-color: rgba(0, 0, 0, 0.3) !important;
            border-radius: 10px;
        }}
        .stFileUploader > div {{
            background-color: rgba(255, 255, 255, 0.1) !important;
            border: 2px dashed rgba(255, 255, 255, 0.3) !important;
            border-radius: 10px;
            color: white !important;
        }}
        .stFileUploader label {{
            color: white !important;
        }}
        .stAlert {{
            background-color: rgba(0, 0, 0, 0.5) !important;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background
set_bg("Team.png")

# UI layout
st.title("📷 Concentration Prediction from Image")

uploaded_file = st.file_uploader("Upload an image of the tube", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    predicted_value = app_manager.predict_obj(image)

    if predicted_value is not None:
        st.success(f"Predicted Concentration: **{predicted_value:.4f}**")
    else:
        st.error("Prediction failed. Please try a different image.")
