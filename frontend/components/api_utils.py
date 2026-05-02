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
        if response.status_code != 200:
            return {
                "ok": False,
                "message": f"Upload failed ({response.status_code})",
                "data": None,
            }

        payload = response.json()
        if payload.get("status") == "indexed":
            return {
                "ok": True,
                "message": f"Indexed {payload.get('chunks', 0)} chunks",
                "data": payload,
            }

        return {
            "ok": False,
            "message": payload.get("reason", "Document was not indexed"),
            "data": payload,
        }
    except Exception as e:
        return {
            "ok": False,
            "message": f"Upload error: {e}",
            "data": None,
        }


def reset_index():
    """Rebuilds the vector index from backend/docs."""
    try:
        response = requests.post(f"{BASE_URL}/admin/reset-index")
        if response.status_code == 200:
            payload = response.json()
            return {
                "ok": True,
                "message": f"Index rebuilt. Total chunks: {payload.get('chunk_count', 0)}",
                "data": payload,
            }
        return {
            "ok": False,
            "message": f"Reset failed ({response.status_code})",
            "data": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "message": f"Reset error: {e}",
            "data": None,
        }