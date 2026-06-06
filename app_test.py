import streamlit as st
import cv2
import os
import gdown
import tempfile
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from PIL import Image
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- CONFIGURATION & DRIVE IDs ---
ALPHABET_MODEL_ID = "1Cwrlihua2N9Z-2W_RxyV-KzCPFSwplw6"
WORD_MODEL_ID = "1ON3LrBqyBCsW7k6kk35IDbujs-k8CXBK"
ALPHABET_MODEL_PATH = 'best_image_alphabet.pt'
WORD_MODEL_PATH = 'best_video_words.pt'

# --- STARTUP: DOWNLOAD MODELS ---
def download_models():
    models_to_download = [(ALPHABET_MODEL_ID, ALPHABET_MODEL_PATH), (WORD_MODEL_ID, WORD_MODEL_PATH)]
    for file_id, output_path in models_to_download:
        if not os.path.exists(output_path):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=False)

download_models()

@st.cache_resource
def load_alphabet_model(): return YOLO(ALPHABET_MODEL_PATH)

@st.cache_resource
def load_word_model(): return YOLO(WORD_MODEL_PATH)

# --- LIVE WEBCAM TRANSFORMER ---
class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_word_model()
        self.window = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, verbose=False)
        label = results[0].names[results[0].probs.top1]
        
        self.window.append(label)
        if len(self.window) > 10: self.window.pop(0)
        
        most_common = Counter(self.window).most_common(1)[0][0]
        
        # Highlight Prediction
        cv2.putText(img, f"Prediction: {most_common}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.rectangle(img, (40, 10), (500, 70), (0, 255, 0), 2)
        return img

# --- APP UI ---
st.set_page_config(page_title="ASL Recognition System", layout="wide")
st.title("🤟 ASL Sign Language Recognition")

tab1, tab2, tab3, tab4 = st.tabs(["🔤 Alphabet", "📝 Word (Img)", "🎥 Word (Video)", "🔴 Live Webcam"])

with tab1:
    uploaded = st.file_uploader("Alphabet Image", type=['jpg', 'jpeg', 'png'], key="alpha")
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=300)
        if st.button("Identify Alphabet"):
            res = load_alphabet_model()(img, verbose=False)
            st.success(f"Result: {res[0].names[res[0].probs.top1]}")

with tab2:
    uploaded = st.file_uploader("Word Image", type=['jpg', 'jpeg', 'png'], key="w_i")
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=300)
        if st.button("Identify Word"):
            res = load_word_model().predict(source=img, save=False)
            st.success(f"Result: {res[0].names[res[0].probs.top1]}")

with tab3:
    st.header("🎥 Background Word Analysis")
    uploaded = st.file_uploader("Upload video", type=['mp4', 'mov', 'avi'], key="w_v")
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        tfile.close()
        if st.button("🔍 Start AI Word Analysis"):
            cap = cv2.VideoCapture(tfile.name)
            WINDOW_SIZE, VOTE_THRESHOLD = 12, 8
            final_word, last_word, win = "", None, []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                label = load_word_model()(frame, verbose=False)[0].names[0]
                win.append(label)
                if len(win) > WINDOW_SIZE: win.pop(0)
                if len(win) == WINDOW_SIZE:
                    common, count = Counter(win).most_common(1)[0]
                    if count >= VOTE_THRESHOLD and common != last_word and common not in ["Nothing", "Space"]:
                        final_word += f" {common}"; last_word = common; win = []
            st.success(f"🏆 Final Word(s): {final_word.strip()}")
            cap.release()
            os.unlink(tfile.name)

with tab4:
    st.header("🔴 Live ASL Word Recognition")
    webrtc_streamer(key="asl-live", video_transformer_factory=ASLTransformer,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
