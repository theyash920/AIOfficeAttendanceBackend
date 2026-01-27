import requests
import os
import sys

BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "test_face.jpg"
TEST_USER_ID = "test-user-001"

def test_root():
    """Test the root endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"GET /: {response.status_code}") 
        print(response.json())
        assert response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Is it running?")
        sys.exit(1)

def test_register_face():
    """Test the face registration endpoint."""
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Skipping registration test: {TEST_IMAGE_PATH} not found.")
        print("Please place a generic face image named 'test_face.jpg' in this directory.")
        return False

    print(f"\nTesting Registration for user: {TEST_USER_ID}")
    with open(TEST_IMAGE_PATH, "rb") as f:
        files = {"image": ("test_face.jpg", f, "image/jpeg")}
        data = {"user_id": TEST_USER_ID}
        
        response = requests.post(f"{BASE_URL}/register-face", data=data, files=files)
        print(f"POST /register-face: {response.status_code}")
        print(response.json())
        
        if response.status_code == 200:
            print("✅ Registration Successful")
            return True
        else:
            print("❌ Registration Failed")
            return False

def test_verify_face():
    """Test the face verification endpoint."""
    if not os.path.exists(TEST_IMAGE_PATH):
        return

    print(f"\nTesting Verification for user: {TEST_USER_ID}")
    with open(TEST_IMAGE_PATH, "rb") as f:
        files = {"image": ("test_face.jpg", f, "image/jpeg")}
        data = {"user_id": TEST_USER_ID}
        
        response = requests.post(f"{BASE_URL}/verify-face", data=data, files=files)
        print(f"POST /verify-face: {response.status_code}")
        print(response.json())
        
        if response.status_code == 200 and response.json().get("verified") is True:
            print("✅ Verification Successful")
        else:
            print("❌ Verification Failed")

if __name__ == "__main__":
    print("--- Starting API Tests ---")
    test_root()
    if test_register_face():
        test_verify_face()
    print("\n--- Test Complete ---")
