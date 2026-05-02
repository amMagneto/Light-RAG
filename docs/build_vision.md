# Light-RAG: Build Vision & Requirements

## 1. Reframe the Narrative — The README Itself
The current README reads like a setup guide. An S-tier project README sells a vision and proves technical depth simultaneously.
Structure it like this:

```markdown
# Light-RAG

> Your documents. Your intelligence. Zero hallucination.

[Demo GIF] [Live Demo Badge] [License] [Stars]

## The Problem
Enterprise knowledge dies in PDFs. Teams re-answer the same 
questions weekly because institutional memory isn't queryable.

## What Light-RAG Does Differently
- Not a chatbot wrapper — a retrieval-first architecture
- Cites every answer with source + page number
- Handles 1-page resumes to 600-page books with dynamic chunking
- Self-hostable, no data leaves your infrastructure

## Architecture Decision Records
Why ChromaDB over Pinecone? Why e5-large over OpenAI embeddings?
[Link to /docs/architecture.md]
```
The ADR (Architecture Decision Records) section alone signals senior engineering maturity.

## 2. Features That Actually Differentiate It
### Answer Provenance — The #1 Trust Signal
```python
# Every answer must return citations
{
  "answer": "The candidate has 5 years of Python experience.",
  "sources": [
    {
      "file": "resume.pdf",
      "page": 2,
      "chunk": "Python (5 years), FastAPI, LangChain...",
      "confidence": 0.87
    }
  ]
}
```
No other basic RAG demo does this properly. It immediately looks production-grade.

### Confidence Scoring with Fallback Transparency
```python
if max_score < 0.25:
    return {
        "answer": None,
        "reason": "insufficient_context",
        "suggestion": "Try rephrasing or upload more relevant documents"
    }
```
Instead of a blunt "I don't know", explain why and guide the user.

### Multi-Document Cross-Referencing
Let users upload 5 documents and ask "What do these contracts have in common?" — most RAG demos are single-doc only. This is a genuine differentiator.

## 3. Technical Upgrades That Show on a Resume
### Async ingestion pipeline
Don't make users wait for indexing:
```python
@app.post("/upload")
async def upload(file: UploadFile, background_tasks: BackgroundTasks):
    background_tasks.add_task(index_document, file)
    return {"status": "indexing", "job_id": uuid4()}

@app.get("/status/{job_id}")
async def check_status(job_id: str): ...
```

### Evaluation harness
This is what makes it S-tier:
```python
# /evals/run.py
# Ground truth Q&A pairs you test against automatically
test_cases = [
    {"question": "What is the candidate's highest degree?",
     "expected": "Master's in Computer Science",
     "document": "resume.pdf"}
]
# Reports retrieval accuracy, answer faithfulness, latency
```
Being able to say "my RAG pipeline scores 91% faithfulness on my eval suite" is something almost no portfolio project has.

### Hybrid search
Add BM25 alongside vector search as discussed earlier. Cite it explicitly in the README as a deliberate architectural choice.

## 4. Observability — Makes It Look Production-Ready
Add a `/metrics` dashboard (even a simple Streamlit page):
- Queries per session
- Average retrieval score
- Chunks retrieved per query
- Most queried documents
- "I don't know" rate over time

```python
# Log every query to SQLite (you already have the DB)
{
  "query": "...",
  "chunks_retrieved": 4,
  "avg_score": 0.71,
  "answered": true,
  "latency_ms": 340,
  "model": "gpt-oss-120b"
}
```
A dashboard showing your system's own performance is the difference between a demo and a product.
