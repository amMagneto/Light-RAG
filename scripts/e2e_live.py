"""Live end-to-end validation against real Groq API.

Pacing: ~9 LLM calls total (search-plan + 2 chats x [plan, grade, generate, extract]),
spread over >60s — comfortably under the client-side RPM limiter (25).
"""
import subprocess
import sys
import time
from pathlib import Path

import requests

PORT = 8200
BASE = f"http://127.0.0.1:{PORT}/api/v1"
DOC = Path("evals/corpus/security_policy.md")

server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)

try:
    print("waiting for server + embedding model...", flush=True)
    health = None
    for _ in range(60):
        time.sleep(2)
        try:
            health = requests.get(f"{BASE}/health", timeout=5).json()
            if health.get("database") == "ok":
                break
        except Exception:
            continue
    assert health, "server never became healthy"
    print("HEALTH:", health)
    assert health["llm_available"] is True, "GROQ key not loaded"

    upload = requests.post(
        f"{BASE}/documents",
        files={"file": (DOC.name, DOC.read_bytes(), "text/markdown")},
        timeout=300,
    ).json()
    print("UPLOAD:", upload)
    assert upload["status"] == "READY" and upload["chunk_count"] >= 4

    search = requests.post(
        f"{BASE}/search",
        json={"query": "what do vendors need before signing", "top_k": 4},
        timeout=120,
    ).json()
    print("SEARCH:", {
        "verdict": search["verdict"],
        "top_doc": search["candidates"][0]["document_name"],
        "intent": search["trace"]["intent"],
        "normalized": search["normalized_query"][:60],
    })
    assert search["candidates"][0]["document_name"] == "security_policy.md"

    time.sleep(10)
    chat1 = requests.post(
        f"{BASE}/chat",
        json={"query": "What certification must vendors hold before we sign a contract?", "top_k": 6},
        timeout=180,
    ).json()
    print("CHAT1:", {
        "citations": len(chat1["citations"]),
        "confidence": chat1["confidence"],
        "claims": len(chat1["claims"]),
        "verdict": chat1["trace"]["verdict"],
        "stages": chat1["trace"]["stage_latencies_ms"],
        "answer": chat1["answer"][:140],
    })
    assert chat1["citations"], "expected grounded citations"
    assert "ISO-27001" in chat1["answer"], "answer should contain the fact"

    session_id = chat1["session_id"]
    time.sleep(10)
    chat2 = requests.post(
        f"{BASE}/chat",
        json={"query": "And how often are access reviews performed?", "session_id": session_id, "top_k": 6},
        timeout=180,
    ).json()
    print("CHAT2:", {
        "same_session": chat2["session_id"] == session_id,
        "normalized": (chat2.get("trace") or {}).get("normalized_query"),
        "citations": len(chat2["citations"]),
        "confidence": chat2["confidence"],
        "answer": chat2["answer"][:140],
    })
    assert chat2["session_id"] == session_id
    assert "quarterly" in chat2["answer"].lower(), "follow-up should resolve context"

    history = requests.get(f"{BASE}/sessions/{session_id}", timeout=30).json()
    print("SESSION messages:", [(m["role"], m["content"][:40]) for m in history["messages"]])
    assert len(history["messages"]) == 4, "user+assistant persisted per turn"

    print("\nLIVE E2E: ALL CHECKS PASSED")
finally:
    server.terminate()
    try:
        server.wait(timeout=10)
    except Exception:
        server.kill()
