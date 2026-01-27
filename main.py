import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
from face_utils import get_face_embedding, compare_faces
import json

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Warning: SUPABASE_URL or SUPABASE_KEY not found in environment variables.")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

app = FastAPI()

# Add CORS middleware to allow requests from mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Face Recognition API is running"}

@app.post("/register-face")
async def register_face(user_id: str = Form(...), image: UploadFile = File(...)):
    """
    Receives an image and user_id.
    Detects face, generates embedding, and stores it in Supabase.
    SECURITY: Checks if face already exists to prevent duplicate registrations.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not configured")

    try:
        contents = await image.read()
        embedding, error = get_face_embedding(contents)
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        # SECURITY CHECK: Check if this face already exists in database
        existing_response = supabase.table("employees").select("id, full_name, face_embedding").execute()
        existing_employees = existing_response.data if existing_response.data else []
        
        for emp in existing_employees:
            stored_embedding = emp.get("face_embedding")
            if stored_embedding:
                if compare_faces(stored_embedding, embedding):
                    # Face already registered with another account
                    existing_name = emp.get("full_name", "another user")
                    existing_id = emp.get("id", "unknown")
                    raise HTTPException(
                        status_code=409, 
                        detail=f"This face is already registered with employee ID: {existing_id[:8]}... Please use your existing account."
                    )
            
        # Store in Supabase
        # Note: We are storing the embedding as a JSON array
        data = {
            "id": user_id,
            "face_embedding": embedding
        }
        
        # Upsert into employees table
        response = supabase.table("employees").upsert(data).execute()
        
        return {"message": "Face registered successfully", "user_id": user_id}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in register_face: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from datetime import datetime
import pytz

@app.post("/verify-face")
async def verify_face(
    user_id: str = Form(...), 
    image: UploadFile = File(...),
    latitude: float = Form(None),
    longitude: float = Form(None),
    office_id: str = Form("OFFICE_MOCK_01"),
    wifi_confidence: int = Form(100)
):
    """
    Receives image, user_id, and location data.
    Verifies face using DeepFace.
    Logs to attendance_logs on success.
    Returns verification status + captured location/time.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not configured")

    try:
        # 1. Get stored embedding
        response = supabase.table("employees").select("face_embedding").eq("id", user_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="User not found or face not registered")
            
        stored_embedding = response.data[0].get("face_embedding")
        
        if not stored_embedding:
             raise HTTPException(status_code=400, detail="No face registered for this user")

        # 2. Get embedding from new image
        contents = await image.read()
        new_embedding, error = get_face_embedding(contents)
        
        if error:
            raise HTTPException(status_code=400, detail=f"Face detection failed: {error}")

        # 3. Compare
        is_match = compare_faces(stored_embedding, new_embedding)
        
        if is_match:
            # 4. Generate Timestamp (UTC)
            now_utc = datetime.now(pytz.utc)
            timestamp_str = now_utc.isoformat()
            
            # 5. Log to Supabase
            log_data = {
                "user_id": user_id,
                "office_id": office_id,
                "confidence_score": wifi_confidence,
                "status": "present",
                "timestamp": timestamp_str,
                "latitude": latitude,
                "longitude": longitude
                # TODO: Upload image to storage and save URL here if needed
            }
            
            # Log attendance asynchronously (or sync for now to ensure strictness)
            log_response = supabase.table("attendance_logs").insert(log_data).execute()

            return {
                "verified": True, 
                "message": "Face verified successfully",
                "location": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "timestamp": timestamp_str
            }
        else:
            return {"verified": False, "message": "Face verification failed"}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in verify_face: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WiFi threshold - requires strong signal like being in the same room as router
WIFI_THRESHOLD = 80  # 80% signal strength required

@app.post("/smart-attendance")
async def smart_attendance(
    image: UploadFile = File(...),
    latitude: float = Form(None),
    longitude: float = Form(None),
    office_id: str = Form("OFFICE_MOCK_01"),
    wifi_confidence: int = Form(0)
):
    """
    Unified endpoint for face-based attendance:
    1. Validates WiFi signal strength (must be >= 80%)
    2. Extracts face embedding from image
    3. Searches all employees for a matching face
    4. If match found -> verify ID and mark attendance
    5. If no match -> register as new employee with generated ID
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not configured")

    try:
        # 1. Validate WiFi strength - must be strong (like being near the office router)
        if wifi_confidence < WIFI_THRESHOLD:
            return {
                "success": False,
                "action": "wifi_weak",
                "message": f"WiFi signal too weak ({wifi_confidence}%). Required: {WIFI_THRESHOLD}%",
                "wifi_required": WIFI_THRESHOLD,
                "wifi_current": wifi_confidence
            }

        # 2. Extract face embedding from the uploaded image
        contents = await image.read()
        new_embedding, error = get_face_embedding(contents)
        
        if error:
            return {
                "success": False,
                "action": "face_error",
                "message": f"Face detection failed: {error}"
            }

        # 3. Get all employees with face embeddings
        response = supabase.table("employees").select("id, full_name, face_embedding").execute()
        employees = response.data if response.data else []

        # 4. Search for matching face
        matched_employee = None
        for emp in employees:
            stored_embedding = emp.get("face_embedding")
            if stored_embedding:
                if compare_faces(stored_embedding, new_embedding):
                    matched_employee = emp
                    break

        now_utc = datetime.now(pytz.utc)
        timestamp_str = now_utc.isoformat()

        if matched_employee:
            # RETURNING USER - Mark attendance
            user_id = matched_employee["id"]
            user_name = matched_employee.get("full_name", "Unknown")

            # Log attendance with location data
            log_data = {
                "user_id": user_id,
                "office_id": office_id,
                "confidence_score": wifi_confidence,
                "status": "present",
                "timestamp": timestamp_str,
                "latitude": latitude,
                "longitude": longitude
            }
            supabase.table("attendance_logs").insert(log_data).execute()

            return {
                "success": True,
                "action": "attendance_marked",
                "message": f"Welcome back, {user_name}! Attendance marked.", 
                "user_id": user_id,
                "user_name": user_name,
                "timestamp": timestamp_str,
                "is_new_user": False,
                "location": {
                    "latitude": latitude,
                    "longitude": longitude
                }
            }
        else:
            # NEW USER - Auto-register
            new_user_id = str(uuid.uuid4())
            
            # Save new employee with face embedding
            new_employee_data = {
                "id": new_user_id,
                "face_embedding": new_embedding,
                "office_id": office_id
            }
            supabase.table("employees").insert(new_employee_data).execute()

            # Also log first attendance with location data
            log_data = {
                "user_id": new_user_id,
                "office_id": office_id,
                "confidence_score": wifi_confidence,
                "status": "present",
                "timestamp": timestamp_str,
                "latitude": latitude,
                "longitude": longitude
            }
            supabase.table("attendance_logs").insert(log_data).execute()

            return {
                "success": True,
                "action": "registered",
                "message": "New face registered! Your employee ID has been created.",
                "user_id": new_user_id,
                "timestamp": timestamp_str,
                "is_new_user": True,
                "location": {
                    "latitude": latitude,
                    "longitude": longitude
                }
            }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in smart_attendance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/smart-checkout")
async def smart_checkout(
    image: UploadFile = File(...),
    latitude: float = Form(None),
    longitude: float = Form(None),
    office_id: str = Form("OFFICE_MOCK_01"),
    wifi_confidence: int = Form(0)
):
    """
    Endpoint for face-based checkout:
    1. Validates WiFi signal strength (must be >= 80%)
    2. Extracts face embedding from image
    3. Searches all employees for a matching face
    4. If match found -> logs checkout
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not configured")

    try:
        # 1. Validate WiFi strength
        if wifi_confidence < WIFI_THRESHOLD:
            return {
                "success": False,
                "action": "wifi_weak",
                "message": f"WiFi signal too weak ({wifi_confidence}%). Required: {WIFI_THRESHOLD}%",
                "wifi_required": WIFI_THRESHOLD,
                "wifi_current": wifi_confidence
            }

        # 2. Extract face embedding from the uploaded image
        contents = await image.read()
        new_embedding, error = get_face_embedding(contents)
        
        if error:
            return {
                "success": False,
                "action": "face_error",
                "message": f"Face detection failed: {error}"
            }

        # 3. Get all employees with face embeddings
        response = supabase.table("employees").select("id, full_name, face_embedding").execute()
        employees = response.data if response.data else []

        # 4. Search for matching face
        matched_employee = None
        for emp in employees:
            stored_embedding = emp.get("face_embedding")
            if stored_embedding:
                if compare_faces(stored_embedding, new_embedding):
                    matched_employee = emp
                    break

        now_utc = datetime.now(pytz.utc)
        timestamp_str = now_utc.isoformat()

        if matched_employee:
            # FOUND USER - Mark checkout
            user_id = matched_employee["id"]
            user_name = matched_employee.get("full_name", "Unknown")

            # Log checkout with location data
            log_data = {
                "user_id": user_id,
                "office_id": office_id,
                "confidence_score": wifi_confidence,
                "status": "checkout",
                "timestamp": timestamp_str,
                "latitude": latitude,
                "longitude": longitude
            }
            supabase.table("attendance_logs").insert(log_data).execute()

            return {
                "success": True,
                "action": "checkout_marked",
                "message": f"Goodbye, {user_name}! Checkout recorded.",
                "user_id": user_id,
                "user_name": user_name,
                "timestamp": timestamp_str,
                "location": {
                    "latitude": latitude,
                    "longitude": longitude
                }
            }
        else:
            return {
                "success": False,
                "action": "face_not_found",
                "message": "Face not recognized. Please check in first."
            }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in smart_checkout: {e}")
        raise HTTPException(status_code=500, detail=str(e))
