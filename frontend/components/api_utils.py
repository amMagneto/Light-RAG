import requests

BASE_URL = "http://127.0.0.1:8000"
API = f"{BASE_URL}/api/v1"


def get_api_response(query, session_id=None):
    """Calls the grounded /chat endpoint. Returns dict with answer, citations, confidence, trace."""
    payload = {"query": query}
    if session_id:
        payload["session_id"] = session_id
    try:
        response = requests.post(f"{API}/chat", json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 503:
            return {
                "answer": "LLM unavailable: check GROQ_API_KEY in .env (it may be expired).",
                "session_id": session_id,
                "citations": [],
            }
    except Exception as e:
        print(f"Error calling backend: {e}")
    return None


def search(query, top_k=8):
    """Pure retrieval endpoint: candidates + verdict, no generation."""
    try:
        response = requests.post(f"{API}/search", json={"query": query, "top_k": top_k}, timeout=60)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error calling backend: {e}")
    return None


def upload_doc(file):
    """Uploads a document; the backend parses, chunks, embeds and indexes it."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API}/documents", files=files, timeout=300)
        if response.status_code == 200:
            payload = response.json()
            return {
                "ok": True,
                "message": f"Indexed {payload.get('chunk_count', 0)} chunks from {payload.get('filename')}",
                "data": payload,
            }
        detail = {}
        try:
            detail = response.json()
        except Exception:
            pass
        return {
            "ok": False,
            "message": detail.get("detail", f"Upload failed ({response.status_code})"),
            "data": None,
        }
    except Exception as e:
        return {"ok": False, "message": f"Upload error: {e}", "data": None}


def list_documents():
    try:
        response = requests.get(f"{API}/documents", timeout=30)
        if response.status_code == 200:
            return response.json().get("documents", [])
    except Exception as e:
        print(f"Error listing documents: {e}")
    return []


def delete_document(document_id):
    try:
        response = requests.delete(f"{API}/documents/{document_id}", timeout=60)
        return response.status_code == 200
    except Exception as e:
        print(f"Error deleting document: {e}")
        return False


def reindex():
    """Rebuilds the vector index from stored chunk rows in the database."""
    try:
        response = requests.post(f"{API}/admin/reindex", timeout=600)
        if response.status_code == 200:
            payload = response.json()
            return {
                "ok": True,
                "message": f"Index rebuilt. Total chunks: {payload.get('chunk_count', 0)}",
                "data": payload,
            }
        return {"ok": False, "message": f"Reindex failed ({response.status_code})", "data": None}
    except Exception as e:
        return {"ok": False, "message": f"Reindex error: {e}", "data": None}


def health():
    try:
        response = requests.get(f"{API}/health", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None
