import face_recognition
import numpy as np
import cv2

def load_image_from_bytes(image_bytes):
    """
    Load image from bytes into a numpy array (RGB for face_recognition).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    # Convert BGR to RGB (face_recognition uses RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def get_face_embedding(image_bytes):
    """
    Detects a face in the image and returns its 128-d embedding.
    Returns (embedding, error_message).
    """
    try:
        img = load_image_from_bytes(image_bytes)
        if img is None:
            return None, "Invalid image data"

        # Detect face locations
        face_locations = face_recognition.face_locations(img, model="hog")
        
        if not face_locations:
            return None, "No face detected"
            
        if len(face_locations) > 1:
            return None, "Multiple faces detected. Please show only one face."

        # Get face encoding (128-dimensional)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        
        if not face_encodings:
            return None, "Could not generate face encoding"

        # Return the first embedding as a list
        return face_encodings[0].tolist(), None

    except Exception as e:
        return None, str(e)

def compare_faces(known_embedding, new_embedding, threshold=0.6):
    """
    Compare two face embeddings using Euclidean distance.
    face_recognition default threshold is 0.6.
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

    # Validate dimensions (face_recognition produces 128-d embeddings)
    if len(known) != len(new):
        print(f"[compare_faces] Dimension mismatch: known={len(known)}, new={len(new)}")
        return False

    # Calculate Euclidean distance
    distance = np.linalg.norm(known - new)
    
    print(f"[compare_faces] Distance: {distance:.4f}, Threshold: {threshold}")
    
    # Lower distance = better match
    return distance < threshold
