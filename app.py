"""
Face Mask Detection - Web App 😷
Tasveer upload karo, AI batayega mask hai ya nahi!
"""

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Face Mask Detection", page_icon="😷")

@st.cache_resource
def load_models():
    mask_model = load_model("mask_detector.h5")
    face_net = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    return mask_model, face_net

mask_model, face_net = load_models()

st.title("😷 Face Mask Detection")
st.write("Tasveer upload karo — AI har chehre pe batayega: **mask hai ya nahi!**")

uploaded = st.file_uploader("📤 Tasveer choose karo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    img = np.array(pil_img)
    st.image(img, caption="Uploaded tasveer", use_container_width=True)

    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    found = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            found = True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face = cv2.resize(face, (128, 128))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            pred = mask_model.predict(face, verbose=0)[0][0]

            if pred > 0.5:
                st.error(f"❌ Chehra {i+1}: No Mask ({pred*100:.1f}%)")
            else:
                st.success(f"✅ Chehra {i+1}: Mask hai ({(1-pred)*100:.1f}%)")

    if not found:
        st.warning("⚠️ Koi chehra nahi mila! Saaf tasveer upload karo.")

st.markdown("---")
st.caption("Built with TensorFlow + OpenCV + Streamlit | by Engineer Muhammad Shehryar Khan")
