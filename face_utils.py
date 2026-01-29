from deepface import DeepFace
import numpy as np
import cv2
import os
import tempfile

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_image_from_bytes(image_bytes):
    """
    Load image from bytes into a numpy array (BGR for OpenCV/DeepFace).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_bgr

def get_face_embedding(image_bytes):
    """
    Detects a face in the image and returns its 128-d embedding using DeepFace.
    Returns (embedding, error_message).
    """
    try:
        img = load_image_from_bytes(image_bytes)
        if img is None:
            return None, "Invalid image data"

        # Create a temporary file to save the image (DeepFace works better with file paths)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, img)

        try:
            # Use Facenet model for 128-d embeddings (lightweight and accurate)
            embedding_objs = DeepFace.represent(
                img_path=tmp_path,
                model_name="Facenet",
                enforce_detection=True,
                detector_backend="opencv"  # Lightweight detector
            )
            
            if not embedding_objs:
                return None, "No face detected"
            
            if len(embedding_objs) > 1:
                return None, "Multiple faces detected. Please show only one face."
            
            # Return the embedding as a list
            embedding = embedding_objs[0]["embedding"]
            return embedding, None
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except ValueError as e:
        error_str = str(e)
        if "Face could not be detected" in error_str:
            return None, "No face detected"
        return None, error_str
    except Exception as e:
        return None, str(e)

def compare_faces(known_embedding, new_embedding, threshold=10.0):
    """
    Compare two face embeddings using Euclidean distance.
    DeepFace Facenet threshold is typically around 10.0 for Euclidean distance.
    Returns True if match, False otherwise.
    """
    if known_embedding is None or new_embedding is None:
        return False

    def normalize_embedding(emb):
        """Convert embedding to a numpy array"""
        if emb is None:
            return None
        if isinstance(emb, np.ndarray):
            return emb
        if isinstance(emb, list):
            return np.array(emb)
        if isinstance(emb, dict):
            if all(str(k).isdigit() for k in emb.keys()):
                return np.array([emb[str(i)] for i in range(len(emb))])
            if 'embedding' in emb:
                return normalize_embedding(emb['embedding'])
            return np.array(list(emb.values()))
        if isinstance(emb, str):
            import json
            try:
                return normalize_embedding(json.loads(emb))
            except:
                return None
        return None

    known = normalize_embedding(known_embedding)
    new = normalize_embedding(new_embedding)
    
    if known is None or new is None:
        print(f"[compare_faces] Failed to normalize embeddings")
        return False

    # Validate dimensions (Facenet produces 128-d embeddings)
    if len(known) != len(new):
        print(f"[compare_faces] Dimension mismatch: known={len(known)}, new={len(new)}")
        return False

    # Calculate Euclidean distance
    distance = np.linalg.norm(known - new)
    
    print(f"[compare_faces] Distance: {distance:.4f}, Threshold: {threshold}")
    
    # Lower distance = better match
    return distance < threshold
