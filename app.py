import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --- Config ---
MODEL_PATH = "best_pretrained_model.keras"
EXAMPLES_DIR = "examples"
CLASSES = ['lung_scc', 'lung_aca', 'lung_n', 'colon_n', 'colon_aca']
CLASS_LABELS = {
    'lung_scc': 'Lung Squamous Cell Carcinoma',
    'lung_aca': 'Lung Adenocarcinoma',
    'lung_n': 'Benign Lung Tissue',
    'colon_n': 'Benign Colon Tissue',
    'colon_aca': 'Colon Adenocarcinoma'
}
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Histopathology Classifier", page_icon="🔬", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🔬 Lung & Colon Histopathology Classifier")
st.write("Upload a histopathological image, or click an example below to try it instantly.")

# --- Example gallery ---
if "selected_example" not in st.session_state:
    st.session_state.selected_example = None

example_files = sorted(os.listdir(EXAMPLES_DIR)) if os.path.isdir(EXAMPLES_DIR) else []

if example_files:
    st.markdown("**Try an example:**")
    cols = st.columns(len(example_files))
    for col, fname in zip(cols, example_files):
        class_key = fname.split('.')[0]
        with col:
            img_path = os.path.join(EXAMPLES_DIR, fname)
            st.image(img_path, use_container_width=True)
            if st.button(CLASS_LABELS.get(class_key, class_key), key=fname):
                st.session_state.selected_example = img_path

st.markdown("---")
uploaded_file = st.file_uploader("...or upload your own image", type=["jpg", "jpeg", "png"])

# --- Decide image source: upload takes priority over a previously clicked example ---
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state.selected_example = None
elif st.session_state.selected_example:
    image = Image.open(st.session_state.selected_example).convert("RGB")

if image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Selected Image", use_container_width=True)

    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing..."):
        preds = model.predict(img_array)[0]

    pred_idx = np.argmax(preds)
    pred_class = CLASSES[pred_idx]
    confidence = preds[pred_idx] * 100

    with col2:
        st.subheader("Prediction")
        st.markdown(f"### {CLASS_LABELS[pred_class]}")
        st.metric("Confidence", f"{confidence:.2f}%")
        st.progress(int(confidence))

        with st.expander("See all class probabilities"):
            for cls, prob in sorted(zip(CLASSES, preds), key=lambda x: -x[1]):
                st.write(f"{CLASS_LABELS[cls]}: {prob*100:.2f}%")
else:
    st.info("Upload an image or click an example above to get a prediction.")