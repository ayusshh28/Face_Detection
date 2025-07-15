import streamlit as st
import cv2
import numpy as np
import joblib

# Load model, PCA, encoder, and Haar cascade for face detection
model = joblib.load('model.pkl')
pca = joblib.load('pca.pkl')

model_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


st.set_page_config(page_title="Gender Detection", page_icon="🧑‍🤖", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #4a6fa5, #7c83db, #b083f7);

        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app title
st.markdown("<h1 style='text-align: center; color: #1c1c1c;'>🔍 Predict Gender from Face</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: #4b4b4b; font-weight: normal;'>Simply upload an image — we’ll do the detecting.</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_clr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize for display consistency
    # img_clr = cv2.resize(img_clr, (600, 600))

    # Convert to grayscale for face detection
    img_gray = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces_list = model_face.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=4)

    for (x, y, w, h) in faces_list:
            # Extract and preprocess the face
            face = img_gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = face / 255.0
            face_flattened = face.flatten().reshape(1, -1)
            face_pca = pca.transform(face_flattened)

            # Prediction
            pred = model.predict(face_pca)
            

            # Draw rectangle and label
            cv2.rectangle(img_clr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_clr,pred[0] ,(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Convert BGR to RGB for Streamlit
    display_img_rgb = cv2.cvtColor(img_clr, cv2.COLOR_BGR2RGB)
    st.image(display_img_rgb, caption="Predicted Gender with Bounding Box", use_column_width=True)
