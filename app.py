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

# --- CONFIGURATION ---
ALPHABET_MODEL_ID = "1Cwrlihua2N9Z-2W_RxyV-KzCPFSwplw6"
WORD_MODEL_ID = "1ON3LrBqyBCsW7k6kk35IDbujs-k8CXBK"
ALPHABET_MODEL_PATH = 'best_image_alphabet.pt'
WORD_MODEL_PATH = 'best_video_words.pt'

# --- STARTUP: DOWNLOAD MODELS ---
@st.cache_resource
def download_models():
    models_to_download = [(ALPHABET_MODEL_ID, ALPHABET_MODEL_PATH), (WORD_MODEL_ID, WORD_MODEL_PATH)]
    for file_id, output_path in models_to_download:
        if not os.path.exists(output_path):
            gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)

download_models()

@st.cache_resource
def load_models():
    return YOLO(ALPHABET_MODEL_PATH), YOLO(WORD_MODEL_PATH)

# --- LIVE WEBCAM TRANSFORMER ---
class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        # Load model within the transformer scope
        self.model = YOLO(WORD_MODEL_PATH)
        self.window = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Inference
        results = self.model(img, verbose=False)
        if len(results[0].probs) > 0:
            label = results[0].names[results[0].probs.top1]
            self.window.append(label)
            if len(self.window) > 10: self.window.pop(0)
            
            most_common = Counter(self.window).most_common(1)[0][0]
            
            # Highlight Prediction with professional styling
            cv2.rectangle(img, (40, 10), (500, 70), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, f"Prediction: {most_common}", (50, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        return img

# --- APP UI ---
st.set_page_config(page_title="ASL Recognition System", layout="wide")
st.title("🔠✋👌✌✊🔠 ASL Sign Language Recognition")

tab1, tab2, tab3, tab4 = st.tabs(["🔤 Alphabet (Img)", "📝 Word (Img)", "🎥 Word (Video)", "🔴 Live Webcam"])

# Logic for Tab 1 & 2
for tab, key, title, model_load in [(tab1, "alpha", "Alphabet", load_models()[0]), (tab2, "w_i", "Word", load_models()[1])]:
    with tab:
        uploaded = st.file_uploader(f"{title} Image", type=['jpg', 'jpeg', 'png'], key=key)
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, width=300)
            if st.button(f"Identify {title}"):
                res = model_load(img, verbose=False)
                st.success(f"Result: {res[0].names[res[0].probs.top1]}")

# Logic for Tab 3 (Video Analysis)
with tab3:
    st.header("🎥 Background Word Analysis")
    uploaded = st.file_uploader("Upload video", type=['mp4', 'mov', 'avi'], key="w_v")
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        tfile.close()
        if st.button("🔍 Start AI Word Analysis"):
            cap = cv2.VideoCapture(tfile.name)
            win = []
            progress = st.progress(0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                label = load_models()[1](frame, verbose=False)[0].names[0]
                win.append(label)
                if len(win) > 12: win.pop(0)
                if len(win) == 12:
                    common = Counter(win).most_common(1)[0][0]
                    if common not in ["Nothing", "Space"]: st.write(f"Detected: {common}")
            cap.release()
            os.unlink(tfile.name)

# Logic for Tab 4 (Live Webcam)
with tab4:
    st.header("🔴 Live ASL Word Recognition")
    webrtc_streamer(
        key="asl-live", 
        video_transformer_factory=ASLTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
