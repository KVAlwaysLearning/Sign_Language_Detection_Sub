To structure your **Sign Language Detection** project for your GitHub repository, I have prepared a comprehensive `README.md` that covers the project's features, setup, and structure, including your explicit requirements for `requirements.txt` and `packages.txt`.

---

# Sign Language Detection System

An AI-driven platform designed to recognize American Sign Language (ASL). The system supports both static image recognition (Alphabet/Words) and real-time live video analysis, with built-in operational time constraints.

## 📁 Repository Contents

* **`Sign_Language_working.ipynb`**: The research notebook covering dataset preparation, custom model training, and performance metrics (including confusion matrices).
* **`app.py`**: A production-ready Streamlit application implementing the inference pipeline, webcam streaming, and UI.
* **`requirements.txt`**: A comprehensive list of Python dependencies required for the project.
* **`packages.txt`**: A list of system-level packages required for specialized library support (such as video streaming codecs and OS-level dependencies).

## 🚀 Features

* **Multi-Modal Recognition**: Predicts alphabets and words from uploaded images or live video feeds.
* **Live Webcam Integration**: Uses `streamlit-webrtc` to provide real-time, low-latency sign language translation directly in the browser.
* **Operational Logic**: Implements time-based access control, ensuring the system is only operational during defined business hours (e.g., 6 PM – 10 PM).
* **Robust Analytics**: Includes frame-based voting logic to ensure prediction stability and reduce flickering during video analysis.

## 🛠️ Setup & Installation

### 1. Prerequisites

Clone this repository:

```bash
git clone https://github.com/KVAlwaysLearning/Sign_Language_Detection_Sub
cd Sign_Language_Detection_Sub

```

### 2. Install Dependencies

Install all required libraries and system packages:

```bash
pip install -r requirements.txt
# If deploying on Linux environments (e.g., Streamlit Cloud):
sudo apt-get install -y $(cat packages.txt)

```

**Key Dependencies:**

* `streamlit`: The interactive web interface.
* `streamlit-webrtc`: Real-time browser-based video streaming.
* `ultralytics`: YOLO model inference engine.
* `opencv-python`: Advanced computer vision and frame manipulation.
* `gdown`: Automated model weight retrieval from Google Drive.

### 3. Model Initialization

The application automatically downloads pre-trained weights (`best_image_alphabet.pt`, `best_video_words.pt`) into the root directory on the first launch. Ensure your environment has write permissions.

## 💻 Usage

### Running the App

Launch the web interface locally:

```bash
streamlit run app.py

```

### Exploring the Research

You can open `Sign_Language_working.ipynb` in any Jupyter-compatible environment to inspect the model training logic, data preprocessing steps, and evaluation charts.

## 📂 Project Structure

```text
├── app.py             # Streamlit web application
├── Sign_Language_working.ipynb # Research and training notebook
├── requirements.txt   # Python dependencies
├── packages.txt       # System-level dependencies
└── README.md          # Project documentation

```

## 🔗 Links

* **Live App**: [Sign Language Detection App](https://signlanguagedetectionsub-app.streamlit.app/)
* **GitHub Repo**: [Sign Language Detection Repository](https://github.com/KVAlwaysLearning/Sign_Language_Detection_Sub)

---

## Visuals:

<img width="743" height="560" alt="image_alphabet_1" src="https://github.com/user-attachments/assets/111aa2dd-9a9b-41c1-9e8a-cb2f5feeee38" />
<img width="779" height="534" alt="image_word_1" src="https://github.com/user-attachments/assets/a97a7229-fe9b-499a-8afd-72b4583ea56c" />
<img width="968" height="495" alt="Live_webcam_1" src="https://github.com/user-attachments/assets/cf59655c-cafe-4594-8057-d56ccb6cdb25" />
<img width="836" height="500" alt="video_word_1" src="https://github.com/user-attachments/assets/0e3d951a-31b0-489b-b5c7-7bc059467681" />
