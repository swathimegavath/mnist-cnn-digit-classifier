"""
app.py
======
Streamlit web GUI for the MNIST Handwritten Digit Recogniser.

Features
--------
- Upload a handwritten digit image (.png / .jpg / .jpeg)
- Preprocesses the image (grayscale → 28×28 → normalise)
- Runs the trained CNN model and displays the predicted digit
- Shows the confidence scores for all 10 digits as a bar chart

Usage
-----
    streamlit run app.py
"""

import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Handwritten Digit Recogniser",
    page_icon="🔢",
    layout="centered",
)


# ── Load model (cached so it is only loaded once) ────────────────────────────
MODEL_PATH = "mnist_cnn_model.keras"

@st.cache_resource
def load_model():
    """Load the trained CNN model from disk."""
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file `{MODEL_PATH}` not found.  "
            "Please run `python train_model.py` first."
        )
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(uploaded_file) -> np.ndarray:
    """
    Preprocess an uploaded image for model inference.

    Steps
    -----
    1. Open with Pillow and convert to greyscale (L mode).
    2. Invert if the background is light (MNIST digits are white-on-black).
    3. Resize to 28 × 28 using Lanczos resampling.
    4. Normalise pixel values to [0, 1].
    5. Add batch and channel dimensions → (1, 28, 28, 1).
    """
    img = Image.open(uploaded_file).convert("L")      # greyscale
    img = ImageOps.autocontrast(img)                  # stretch contrast

    # Invert if background appears lighter than foreground
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    img = img.resize((28, 28), Image.LANCZOS)         # resize
    arr = np.array(img, dtype="float32") / 255.0      # normalise
    arr = arr.reshape(1, 28, 28, 1)                   # batch + channel
    return arr, img


def plot_confidence(probabilities: np.ndarray):
    """Return a matplotlib figure showing per-class confidence."""
    fig, ax = plt.subplots(figsize=(7, 3))
    colours = ["#4C72B0"] * 10
    predicted = int(np.argmax(probabilities))
    colours[predicted] = "#DD8452"           # highlight predicted digit

    ax.bar(range(10), probabilities * 100, color=colours)
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit", fontsize=11)
    ax.set_ylabel("Confidence (%)", fontsize=11)
    ax.set_title("Model Confidence per Class", fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ── App layout ────────────────────────────────────────────────────────────────
st.title("🔢 Handwritten Digit Recogniser")
st.markdown(
    "Upload a handwritten digit image (0 – 9) and the CNN model will "
    "predict which digit it is."
)
st.divider()

# Load model eagerly so any error shows at the top
model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Model:** Convolutional Neural Network  
        **Dataset:** MNIST (60,000 training images)  
        **Framework:** TensorFlow / Keras  

        **Preprocessing steps applied to your image:**
        1. Convert to greyscale  
        2. Auto-contrast adjustment  
        3. Background inversion (if needed)  
        4. Resize to 28 × 28 px  
        5. Pixel normalisation [0, 1]  
        """
    )
    st.divider()
    st.markdown("**Accepted formats:** `.png`, `.jpg`, `.jpeg`")

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Browse or drag-and-drop a handwritten digit image",
    type=["png", "jpg", "jpeg"],
    label_visibility="visible",
)

if uploaded_file is not None:
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    # ── Left column: uploaded image ───────────────────────────────────────────
    with col1:
        st.subheader("Uploaded Image")
        st.image(uploaded_file, use_container_width=True)   # ← fixed deprecation

    # ── Preprocess and predict ────────────────────────────────────────────────
    with st.spinner("Analysing image …"):
        try:
            arr, resized_img = preprocess_image(uploaded_file)
            probabilities    = model.predict(arr, verbose=0)[0]   # shape (10,)
            predicted_digit  = int(np.argmax(probabilities))
            confidence       = float(probabilities[predicted_digit]) * 100
        except Exception as exc:
            st.error(f"Error during processing: {exc}")
            st.stop()

    # ── Right column: prediction result ──────────────────────────────────────
    with col2:
        st.subheader("Prediction")
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius: 16px;
                padding: 28px;
                text-align: center;
                border: 2px solid #DD8452;
            ">
                <div style="font-size: 100px; line-height: 1; color: #DD8452;">
                    {predicted_digit}
                </div>
                <div style="font-size: 18px; color: #e0e0e0; margin-top: 12px;">
                    Predicted Digit
                </div>
                <div style="font-size: 15px; color: #a0a0a0; margin-top: 6px;">
                    Confidence: <strong style="color:#DD8452;">{confidence:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("&nbsp;")
        st.subheader("Preprocessed (28 × 28)")
        st.image(
            resized_img.resize((112, 112), Image.NEAREST),
            caption="Image fed to the model",
        )

    # ── Confidence chart ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Confidence Scores")
    fig = plot_confidence(probabilities)
    st.pyplot(fig)
    plt.close(fig)

    # ── Raw scores table ──────────────────────────────────────────────────────
    with st.expander("Show raw probability scores"):
        import pandas as pd
        df = pd.DataFrame(
            {"Digit": list(range(10)),
             "Probability (%)": [f"{p * 100:.2f}" for p in probabilities]}
        ).set_index("Digit")
        st.dataframe(df, use_container_width=True)

else:
    # Placeholder
    st.info(
        "👆 Upload an image above to get started.  \n"
        "The image should contain a **single handwritten digit** on a plain background."
    )
