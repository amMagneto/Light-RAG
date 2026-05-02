# Light-RAG: The Vertical AI Search Engine

Most search engines help you find information the rest of the world already knows. Light-RAG is different. It is designed for the information that only *you* know. 

In a world where horizontal AI tries to solve every problem for everyone, the real value lies in vertical integration. Light-RAG is a proprietary intelligence layer that transforms your unstructured institutional knowledge into a functional, queryable asset. We didn't build a chatbot; we built a system that bridges the gap between static documents and actionable insight.

The future doesn't belong to those who use the most data, but to those who own the most relevant context. Light-RAG is the infrastructure for that future.

## Key Features

- **Dynamic Chunking:** Automatically adjusts chunk sizes and overlaps based on document size and page count for optimal retrieval.
- **Conversational Memory:** Uses a history-aware retriever to handle follow-up questions and maintain context across a session.
- **Multi-Format Support:** Seamlessly ingests PDF and DOCX files.
- **Traceability:** Logs every interaction and document metadata into a SQLite database for auditability and performance tracking.
- **Real-time Indexing:** Uploaded documents are indexed on-the-fly, making them immediately queryable.

## System Architecture

Light-RAG utilizes a modular architecture:

- **Frontend:** Built with **Streamlit**, providing a clean interface for document management and chat.
- **Backend:** A **FastAPI** service that orchestrates the RAG pipeline.
- **Vector Store:** **ChromaDB** handles high-performance semantic search using `intfloat/e5-base-v2` embeddings.
- **LLM Integration:** **LangChain** and **Groq** power the context-aware generation.
- **Database:** **SQLite** stores application logs and document metadata.

## Getting Started

### 1. Environment Setup

Clone the repository and initialize a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory:

```text
GROQ_API_KEY=your_key_here
```

### 3. Running the System

**Step A: Start the Backend**
```powershell
uvicorn backend.main:app --reload
```

**Step B: Start the Frontend**
```powershell
streamlit run app.py
```

Open `http://localhost:8501` to start chatting with your documents.

## Database Schema

The system uses `rag_app.db` (SQLite) with the following tables:

- `application_logs`: Stores `session_id`, `user_query`, `gpt_response`, and `model`.
- `document_metadata`: Tracks `filename`, `chunk_size`, `chunk_overlap`, `page_count`, and `file_size_kb`.

## Testing

The project includes a suite of unit and integration tests. To run them:

```powershell
$env:PYTHONPATH=".;backend"
python -m pytest
```

## Usage

1. **Upload:** Use the sidebar to upload PDF or DOCX files.
2. **Chat:** Ask questions. The system retrieves relevant context from your documents and provides a synthesized answer.
3. **Reset Index:** Use the "Reset Index" button in the sidebar to rebuild the vector store from the stored files in `backend/docs`.
