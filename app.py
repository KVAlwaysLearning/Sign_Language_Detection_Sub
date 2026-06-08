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
from datetime import datetime, timedelta, timezone

# --- CONFIGURATION ---
# We define file paths locally
ALPHABET_MODEL_PATH = 'best_image_alphabet.pt'
WORD_MODEL_PATH = 'best_video_words.pt'


# --- STARTUP: DOWNLOAD MODELS ---
@st.cache_resource
def download_models():
    # Retrieve secrets dynamically
    ids = st.secrets["drive_ids"]
    models_to_download = [
        (ids["alphabet_model"], 'best_image_alphabet.pt'), 
        (ids["word_model"], 'best_video_words.pt')
    ]
    
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

# Define IST timezone (UTC + 5:30)
ist_timezone = timezone(timedelta(hours=5, minutes=30))
now_ist = datetime.now(ist_timezone)

current_hour = now_ist.hour
current_time_str = now_ist.strftime("%I:%M %p")

# Check if within 6 PM (18) and 10 PM (22)
if not (18 <= current_hour < 22):
    st.error(f"⏰ Service Unavailable. Current IST time is {current_time_str}.")
    st.info("The application is only operational between 6:00 PM and 10:00 PM IST.")
    st.stop()

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
    st.header("🎥 Word Identification (Background Analysis)")
    uploaded_video = st.file_uploader("Upload a sign video for analysis", type=['mp4', 'mov', 'avi'], key="w_v")
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close() 

        st.info("✅ Video uploaded. Click the button below to start the background AI analysis.")

        if st.button("🔍 Start AI Word Analysis"):
            cap = cv2.VideoCapture(tfile.name)
            
            if not cap.isOpened():
                st.error("Error: System could not read the video file.")
            else:
                WINDOW_SIZE, VOTE_THRESHOLD = 12, 8
                final_word, last_word, prediction_window = "", None, []
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = 0
                progress_bar = st.progress(0)
                status_text = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    word_model = load_models()[1]
                    results = word_model(frame, verbose=False)
                    label = results[0].names[results[0].probs.top1]
                    
                    prediction_window.append(label)
                    if len(prediction_window) > WINDOW_SIZE: 
                        prediction_window.pop(0)
                    
                    if len(prediction_window) == WINDOW_SIZE:
                        counts = Counter(prediction_window)
                        common, count = counts.most_common(1)[0]
                        
                        if count >= VOTE_THRESHOLD and common != last_word:
                            if common not in ["Nothing", "Space"]:
                                final_word += f" {common}"
                                last_word = common
                                prediction_window = []

                    current_frame += 1
                    if current_frame % 5 == 0:
                        progress_val = min(current_frame / total_frames, 1.0)
                        progress_bar.progress(progress_val)
                        status_text.text(f"Processing frame {current_frame}/{total_frames}...")

                cap.release()
                progress_bar.empty()
                status_text.empty()
                
                if final_word.strip():
                    st.success(f"🏆 **Final Identified Word(s):** {final_word.strip()}")
                else:
                    st.warning("⚠️ No sign language words were clearly detected in this video.")

        if os.path.exists(tfile.name):
            try: os.unlink(tfile.name)
            except: pass

# Logic for Tab 4 (Live Webcam)
with tab4:
    st.header("🔴 Live ASL Word Recognition")
    webrtc_streamer(
        key="asl-live", 
        video_transformer_factory=ASLTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
