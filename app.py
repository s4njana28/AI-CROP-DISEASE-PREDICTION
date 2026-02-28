"""
app.py
Run: streamlit run app.py
"""
import streamlit as st
from PIL import Image
from src.predict import predict_disease, generate_gradcam

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="AI Crop Disease Detection", layout="wide")

st.title("🌱 AI-Based Crop Disease Classification")
st.write("Upload a leaf image to detect crop disease.")

# ---------------------------
# Image Upload
# ---------------------------
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Predict
    with st.spinner("Analyzing image..."):
        predicted_class, confidence, predicted_index = predict_disease(image)
        gradcam_image = generate_gradcam(image, predicted_index)

    # Show results
    st.success(f"🌿 Prediction: {predicted_class}")
    st.info(f"📊 Confidence: {confidence:.2f}%")

    # Show images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("🔥 Grad-CAM Heatmap")
        st.image(gradcam_image, use_container_width=True)

    st.caption("🔴 Red/Yellow areas = regions the AI focused on to make its prediction")
