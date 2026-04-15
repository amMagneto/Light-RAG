import requests

# The URL where your FastAPI backend is running
BASE_URL = "http://127.0.0.1:8000"

def get_api_response(query, session_id=None):
    """
    Calls the /chat endpoint of the FastAPI backend.
    Sends the user query and optional session_id for memory.
    """
    payload = {"query": query, "session_id": session_id}
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error calling backend: {e}")
    return None


def upload_doc(file):
    """Sends a binary file to the FastAPI /upload endpoint."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{BASE_URL}/upload", files=files)
        return response.status_code == 200
    except Exception as e:
        print(f"Upload error: {e}")
        return False