import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr

# Title
st.title("📝 Handwritten Notes Reader")

# Upload image
uploaded_image = st.file_uploader("Upload a handwritten image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    return easyocr.Reader(['en'])

def preprocess(image):
    img = np.array(image.convert("L"))  # Convert to grayscale
    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    img = cv2.bilateralFilter(img, 11, 17, 17)
    return img

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_image)

    st.write("⏳ Processing...")

    preprocessed = preprocess(image)
    reader = load_model()
    result = reader.readtext(preprocessed)

    extracted_text = " ".join([item[1] for item in result])

    st.subheader("🧾 Extracted Text:")
    st.text_area("Text from image", extracted_text, height=300)
