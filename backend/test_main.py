import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

if __package__ in (None, ""):
    # Support running as: python backend/test_main.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from backend.main import app
else:
    from .main import app

client = TestClient(app)

# API AND INTEGRATION TESTS
def test_chat_new_session():
    """"Ensuring a fresh user gets an answer and a new Session ID"""
    payload = {"query": "What is the first principle of engineering?"}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "session_id" in data
    assert len(data["session_id"]) > 0

def test_chat_continuation_memory():
    """"Verifying the system remembers context"""
    turn1 = client.post("/chat", json={"query": "Tell me about Elon Musk's rocket cost in 2002."})
    session_id = turn1.json()["session_id"]

    # turn2 tests the history-aware re-writer
    turn2 = client.post("/chat", json={"query": "Why did he walk away from the Russians?", "session_id": session_id})

    assert turn2.status_code == 200
    assert session_id == turn2.json()["session_id"]
    # the answer should reference the $65 million cost
    answer_text = turn2.json()["answer"].lower()
    assert any(word in answer_text for word in ["rocket","cost","million","65"])

def test_empty_query_edge_case():
    """Graceful handling of empty string inputs"""
    response = client.post("/chat", json={"query": ""})
    assert response.status_code == 200
    assert len(response.json()["answer"])>0

def test_invalid_json_payload():
    """Ensuring Pydantic schemas reject bad data formats."""
    response = client.post("/chat", json = {'query': 12345})
    assert response.status_code ==422

# rag robustness test and precison test

@pytest.mark.parametrize("query, expected_keywords", [
    ("How does  AI help doctors?", "diagnostics"),
    ("What are the key causes of climate change?", "greenhouse"),
    ("What is first principles thinking?", ("analogy", "ground up", "basic truths")),
])

def test_document_retrieval_precision(query, expected_keywords):
    """Verify that specific facts are retreived from the correct documents."""
    response = client.post("/chat", json={"query": query})
    assert response.status_code == 200
    answer= response.json()["answer"].lower()
    if isinstance(expected_keywords, (tuple, list, set)):
        assert any(keyword.lower() in answer for keyword in expected_keywords)
    else:
        assert expected_keywords.lower() in answer

def test_cross_session_isolation():
    # session a: blockchain context
    """Verify that session A's memory does not bleed into session B."""
    res_a = client.post("/chat", json={"query": "Explain blockchain decentralization."})
    id_a = res_a.json()["session_id"]

    # session b: health care context
    res_b= client.post("/chat", json= {"query":"Explain AI in medical imaging."})
    id_b = res_b.json()["session_id"]

    # follow up on session A, should retrieve 'ledger' nor 'doctor'
    res_a_2 = client.post("/chat", json = {"query":"Summarize that.", "session_id": id_a})
    answer_a = res_a_2.json()["answer"].lower()
    assert "ledger" in answer_a
    assert "doctor" not in answer_a


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))