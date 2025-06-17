# app.py

import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import json
import os

# -------- Configuration --------
MODEL_PATH = "plant_disease_model.h5"   # Path to your saved Keras model
CLASS_NAMES_PATH = "class_names.json"   # Path to JSON file with class names list
IMG_HEIGHT = 180                        # Must match the height used during training
IMG_WIDTH = 180                         # Must match the width used during training
# ---------------------------------

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

@st.cache_resource(show_spinner=False)
def load_model():
    """Load and return the Keras model. Cached so it's loaded only once."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        st.stop()
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    return model

@st.cache_data(show_spinner=False)
def load_class_names():
    """Load and return the list of class names from JSON."""
    if not os.path.exists(CLASS_NAMES_PATH):
        st.error(f"Class names file not found at {CLASS_NAMES_PATH}")
        st.stop()
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        if not isinstance(class_names, list):
            st.error(f"{CLASS_NAMES_PATH} does not contain a JSON list")
            st.stop()
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        st.stop()
    return class_names

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Given a PIL Image, resize to (IMG_WIDTH, IMG_HEIGHT), normalize to [0,1],
    and return a batch array shape (1, IMG_HEIGHT, IMG_WIDTH, 3).
    """
    # Convert to RGB (in case image is grayscale or RGBA)
    image = image.convert('RGB')
    # Resize
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    # Convert to numpy array
    img_array = np.array(image).astype('float32') / 255.0
    # Expand dims to batch
    input_arr = np.expand_dims(img_array, axis=0)  # shape (1, H, W, 3)
    return input_arr

def main():
    st.title("🌿 Plant Disease Detection")
    st.write("Upload an image of a plant leaf; the model will predict its disease status.")

    # Load resources
    model = load_model()
    class_names = load_class_names()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except UnidentifiedImageError:
            st.error("Cannot open uploaded file as an image. Please upload a valid image.")
            return
        except Exception as e:
            st.error(f"Error reading image: {e}")
            return

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")  # spacer
        st.write("Classifying...")

        # Preprocess
        try:
            input_arr = preprocess_image(image)
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return

        # Predict
        try:
            preds = model.predict(input_arr)
        except Exception as e:
            st.error(f"Inference error: {e}")
            return

        # Process output
        if preds.ndim == 2 and preds.shape[0] == 1:
            pred_idx = int(np.argmax(preds, axis=1)[0])
            confidence = float(np.max(preds, axis=1)[0])
        else:
            st.error(f"Unexpected model output shape: {preds.shape}")
            return

        # Map to class name
        if 0 <= pred_idx < len(class_names):
            pred_class = class_names[pred_idx]
        else:
            pred_class = "Unknown"

        # Display results
        st.markdown(f"**Predicted class:** {pred_class}")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

        # Optionally show top-3 predictions
        top_k = 3
        if preds.shape[1] >= top_k:
            top_indices = np.argsort(preds[0])[::-1][:top_k]
            st.write("**Top predictions:**")
            for idx in top_indices:
                name = class_names[idx] if idx < len(class_names) else f"Index {idx}"
                conf = float(preds[0][idx])
                st.write(f"- {name}: {conf * 100:.2f}%")
    else:
        st.info("Please upload an image file to get a prediction.")

    # Sidebar info
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app uses a TensorFlow/Keras model (`plant_disease_model.h5`) to classify plant leaf images.
        Ensure `plant_disease_model.h5` and `class_names.json` are in the same directory as this script.
        """
    )

if __name__ == "__main__":
    main()
