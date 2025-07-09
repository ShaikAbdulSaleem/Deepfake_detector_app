import streamlit as st
import cv2
import numpy as np
import pickle
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image # For image processing with MTCNN
import tempfile # To handle uploaded video files

# --- Configuration and Model Paths ---
CLASSIFIER_MODEL_PATH = "deepfake_classifier.pkl"
SAMPLE_VIDEO_REAL = "sample_real.mp4" # Add a small sample real video in your repo
SAMPLE_VIDEO_FAKE = "sample_fake.mp4" # Add a small sample fake video in your repo

# --- 1. Load Pre-trained Models (Cached for efficiency) ---
@st.cache_resource
def load_mtcnn():
    """Loads the MTCNN face detection model."""
    return MTCNN(image_size=160, margin=0, min_face_size=20,
                 thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                 device='cpu') # Ensure CPU for Streamlit Cloud

@st.cache_resource
def load_resnet():
    """Loads the InceptionResnetV1 face embedding model."""
    return InceptionResnetV1(pretrained='vggface2').eval().to('cpu') # Ensure CPU

@st.cache_resource
def load_classifier():
    """Loads the trained deepfake classifier."""
    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        st.error(f"Classifier model not found at {CLASSIFIER_MODEL_PATH}. Please train the model first.")
        st.stop() # Stop the app if model is missing
    with open(CLASSIFIER_MODEL_PATH, 'rb') as f:
        classifier = pickle.load(f)
    return classifier

mtcnn = load_mtcnn()
resnet = load_resnet()
classifier = load_classifier()

# --- 2. Face Embedding Extraction Function (for inference) ---
def extract_face_embedding_from_video(video_path, mtcnn_model, resnet_model, frames_to_process=10):
    cap = cv2.VideoCapture(video_path)
    embeddings = []
    
    # Check if video opened successfully
    if not cap.isOpened():
        st.error(f"Error: Could not open video file {video_path}")
        return None

    frame_count = 0
    # Use st.spinner for user feedback during processing
    with st.spinner("Processing video and extracting faces..."):
        while(cap.isOpened() and frame_count < frames_to_process):
            ret, frame = cap.read()
            if not ret:
                break

            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            face_cropped = mtcnn_model(img_pil)
            
            if face_cropped is not None:
                if face_cropped.dim() == 4 and face_cropped.shape[0] > 1:
                    face_cropped = face_cropped[0].unsqueeze(0) # Take first face
                
                with torch.no_grad():
                    embedding = resnet_model(face_cropped.to('cpu')).cpu().numpy().flatten()
                embeddings.append(embedding)
            frame_count += 1
        
    cap.release()
    
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0) # Average embeddings across frames
    else:
        return None # No faces found

# --- 3. Streamlit UI ---
st.set_page_config(page_title="Deepfake Detector", layout="centered", initial_sidebar_state="auto")

st.title("📹 Deepfake Video Detector")
st.markdown("Upload a video and let the model predict if it's real or a deepfake!")
st.warning("This is a demo for educational purposes. Performance may vary. This model does not use GPUs.")

# Option to upload video
uploaded_file = st.file_uploader("Choose a video file (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

# Option to use sample videos
st.markdown("---")
st.subheader("Or try with a sample video:")
col1, col2 = st.columns(2)

sample_video_choice = None
if col1.button("Use Sample Real Video", use_container_width=True):
    sample_video_choice = SAMPLE_VIDEO_REAL
if col2.button("Use Sample Fake Video", use_container_width=True):
    sample_video_choice = SAMPLE_VIDEO_FAKE

video_to_process = None
if uploaded_file is not None:
    st.info("Uploaded video selected.")
    # Save uploaded file to a temporary file for OpenCV to read
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_to_process = temp_file.name
    st.video(video_to_process)

elif sample_video_choice:
    if os.path.exists(sample_video_choice):
        st.info(f"Using sample video: {os.path.basename(sample_video_choice)}")
        video_to_process = sample_video_choice
        st.video(video_to_process)
    else:
        st.error(f"Sample video '{sample_video_choice}' not found in the repository. Please ensure it's included for the demo.")
        video_to_process = None # Reset if not found

if video_to_process:
    if st.button("Analyze Video"):
        st.write("---")
        st.subheader("Analysis Results:")
        
        # Extract features
        embedding = extract_face_embedding_from_video(video_to_process, mtcnn, resnet)

        if embedding is not None:
            # Predict
            prediction = classifier.predict(embedding.reshape(1, -1))[0]
            probability = classifier.predict_proba(embedding.reshape(1, -1))[0]

            col_pred, col_prob = st.columns(2)
            if prediction == 0:
                col_pred.success("Prediction: **REAL** video")
            else:
                col_pred.error("Prediction: **FAKE** video")
            
            col_prob.metric("Real Probability", f"{probability[0]*100:.2f}%")
            col_prob.metric("Fake Probability", f"{probability[1]*100:.2f}%")

            if prediction == 0:
                st.balloons()
            else:
                st.snow() # Or some other visual cue for fake
        else:
            st.warning("No faces detected in the video or an error occurred during processing.")
    
    # Clean up temporary file if uploaded
    if uploaded_file is not None and os.path.exists(video_to_process):
        try:
            os.unlink(video_to_process) # Delete the temporary file
        except Exception as e:
            st.error(f"Error cleaning up temporary file: {e}")

st.markdown("---")
st.info("Developed by Your Name/Organization. Model uses FaceNet for embeddings and Logistic Regression for classification.")
