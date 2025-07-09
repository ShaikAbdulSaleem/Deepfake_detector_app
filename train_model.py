import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from tqdm import tqdm

# --- Configuration ---
DATASET_DIR = "deepfake_dataset" # Replace with your actual dataset path
REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")
MODEL_SAVE_PATH = "deepfake_classifier.pkl"
EMBEDDINGS_CACHE = "embeddings_cache.pkl"

# --- 1. Initialize Face Detection and Embedding Models ---
# MTCNN for face detection and alignment
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
              thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
              device='cpu' # Force CPU for training as well for consistency
             )

# InceptionResnetV1 for face embedding
# pretrained='vggface2' downloads a pre-trained model on VGGFace2 dataset
resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu') # Force CPU

# --- 2. Feature Extraction Function ---
def extract_face_embedding(video_path, mtcnn_model, resnet_model):
    cap = cv2.VideoCapture(video_path)
    embeddings = []
    frame_count = 0
    # Process a subset of frames to speed up training data creation
    # For a real project, you might process more frames or a specific interval
    frames_to_process = 10 # Process 10 frames per video

    while(cap.isOpened() and frame_count < frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to PIL Image for MTCNN
        img_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face and get cropped image
        # mtcnn returns a tensor of shape (num_faces, channels, height, width)
        # We assume one main face per video for simplicity
        face_cropped = mtcnn(img_pil) 
        
        if face_cropped is not None:
            # If multiple faces, take the first one or average them
            if face_cropped.dim() == 4 and face_cropped.shape[0] > 1:
                face_cropped = face_cropped[0].unsqueeze(0) # Take first face
            
            # Calculate embedding
            with torch.no_grad():
                embedding = resnet(face_cropped.to('cpu')).cpu().numpy().flatten()
            embeddings.append(embedding)
        frame_count += 1
    
    cap.release()
    
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0) # Average embeddings across frames
    else:
        return None # No faces found

# --- 3. Prepare Data for Classifier Training ---
all_embeddings = []
labels = []
video_paths = []

# Check if cache exists
if os.path.exists(EMBEDDINGS_CACHE):
    print(f"Loading embeddings from cache: {EMBEDDINGS_CACHE}")
    with open(EMBEDDINGS_CACHE, 'rb') as f:
        all_embeddings, labels = pickle.load(f)
else:
    print("Extracting embeddings from videos (this may take a while)...")
    # Process Real Videos
    for video_name in tqdm(os.listdir(REAL_DIR), desc="Processing Real Videos"):
        if video_name.endswith((".mp4", ".avi", ".mov")): # Add other video formats if needed
            video_path = os.path.join(REAL_DIR, video_name)
            embedding = extract_face_embedding(video_path, mtcnn, resnet)
            if embedding is not None:
                all_embeddings.append(embedding)
                labels.append(0) # 0 for Real
                video_paths.append(video_path)

    # Process Fake Videos
    for video_name in tqdm(os.listdir(FAKE_DIR), desc="Processing Fake Videos"):
        if video_name.endswith((".mp4", ".avi", ".mov")): # Add other video formats if needed
            video_path = os.path.join(FAKE_DIR, video_name)
            embedding = extract_face_embedding(video_path, mtcnn, resnet)
            if embedding is not None:
                all_embeddings.append(embedding)
                labels.append(1) # 1 for Fake
                video_paths.append(video_path)

    # Cache embeddings
    with open(EMBEDDINGS_CACHE, 'wb') as f:
        pickle.dump((all_embeddings, labels), f)
    print(f"Embeddings cached to {EMBEDDINGS_CACHE}")

if not all_embeddings:
    print("No embeddings extracted. Please check your dataset path and video files.")
    exit()

X = np.array(all_embeddings)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. Train a Simple Classifier ---
print("\nTraining Logistic Regression Classifier...")
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train, y_train)

# --- 5. Evaluate the Classifier ---
y_pred = classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# --- 6. Save the Trained Classifier ---
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(classifier, f)
print(f"\nClassifier saved to {MODEL_SAVE_PATH}")

print("Training pipeline complete.")
