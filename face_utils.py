from deepface import DeepFace
import numpy as np
import cv2
import os

# Ensure deepface weights path is writable if needed, or handle download issues
# os.environ["DEEPFACE_HOME"] = "./.deepface"

def load_image_from_bytes(image_bytes):
    """
    Load image from bytes into a numpy array (BGR for OpenCV/DeepFace).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_face_embedding(image_bytes):
    """
    Detects a face in the image and returns its 512-d embedding (ArcFace).
    Returns (embedding, error_message).
    """
    try:
        img = load_image_from_bytes(image_bytes)
        if img is None:
            return None, "Invalid image data"

        # Generate embedding
        # model_name: ArcFace (SOTA), VGG-Face, Facenet, etc.
        # detector_backend: opencv (fast), retinaface (accurate but slow), mtcnn
        embedding_objs = DeepFace.represent(
            img_path=img,
            model_name="ArcFace",
            detector_backend="opencv",  # Lightweight detector for low memory environments
            enforce_detection=True,
            align=True
        )

        if not embedding_objs or len(embedding_objs) == 0:
            return None, "No face detected"
            
        if len(embedding_objs) > 1:
            return None, "Multiple faces detected. Please show only one face."

        # Return the first embedding
        return embedding_objs[0]["embedding"], None

    except ValueError as ve:
        # DeepFace raises ValueError when "Face could not be detected" if enforce_detection=True
        if "Face could not be detected" in str(ve):
             return None, "No face detected"
        return None, str(ve)
    except Exception as e:
        return None, str(e)

def compare_faces(known_embedding, new_embedding, threshold=0.40):
    """
    Compare two ArcFace embeddings using Cosine Similarity.
    DeepFace default threshold for ArcFace is usually around 0.50.
    Returns True if match, False otherwise.
    """
    if known_embedding is None or new_embedding is None:
        return False

    # Handle different formats of embeddings from database
    def normalize_embedding(emb):
        """Convert embedding to a flat list of floats"""
        if emb is None:
            return None
        # If it's already a list of numbers, use it directly
        if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], (int, float)):
            return emb
        # If it's a dict (e.g., from JSON storage), try to extract values
        if isinstance(emb, dict):
            # Some databases store as {"0": val, "1": val, ...}
            if all(k.isdigit() for k in emb.keys()):
                return [emb[str(i)] for i in range(len(emb))]
            # Or it might have an 'embedding' key
            if 'embedding' in emb:
                return normalize_embedding(emb['embedding'])
            # Try to get values directly
            return list(emb.values())
        # If it's a string, try to parse as JSON
        if isinstance(emb, str):
            import json
            try:
                return normalize_embedding(json.loads(emb))
            except:
                return None
        return emb

    known = normalize_embedding(known_embedding)
    new = normalize_embedding(new_embedding)
    
    if known is None or new is None:
        print(f"[compare_faces] Failed to normalize embeddings. Known type: {type(known_embedding)}, New type: {type(new_embedding)}")
        return False

    # Validate embedding dimensions match (ArcFace produces 512-d embeddings)
    if len(known) != len(new):
        print(f"[compare_faces] Dimension mismatch: known={len(known)}, new={len(new)}. Skipping this comparison.")
        return False
    
    # Skip embeddings that are not the expected 512 dimensions
    EXPECTED_DIM = 512
    if len(known) != EXPECTED_DIM or len(new) != EXPECTED_DIM:
        print(f"[compare_faces] Invalid embedding dimension: got {len(known)} and {len(new)}, expected {EXPECTED_DIM}. Skipping.")
        return False

    # Manual Cosine Similarity
    a = np.array(known, dtype=np.float64)
    b = np.array(new, dtype=np.float64)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return False
        
    cosine_similarity = dot_product / (norm_a * norm_b)
    
    # ArcFace: Higher cosine similarity means better match.
    # Usually, if distance (1 - cosine) < threshold, it's a match.
    # Convert cosine to distance:
    distance = 1 - cosine_similarity
    
    print(f"[compare_faces] Distance: {distance:.4f}, Threshold: {threshold}")
    
    # DeepFace typically uses distance < 0.68 for ArcFace
    # Let's be slightly stricter or stick to defaults.
    # 0.68 is standard for ArcFace.
    
    return distance < threshold

