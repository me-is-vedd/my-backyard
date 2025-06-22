import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
from sklearn.neighbors import KNeighborsClassifier

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_path = "Screenshot 2025-06-19 134127.png"  # Change to your image filename
img_base64 = get_base64_of_bin_file(img_path)

# Set Streamlit background color to red using HTML/CSS
st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Sign Language Translator")

model = joblib.load('model.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

st.write("Show your hand sign in front of the webcam. Press the checkbox to start/stop.")

cap = None

if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        prediction_text = ""
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                if len(landmarks) == 63:
                    prediction = model.predict([landmarks])
                    prediction_text = f'Prediction: {prediction[0]}'
                    cv2.putText(frame, prediction_text, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    st.write('Camera stopped.')
