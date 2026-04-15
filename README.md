# Light-RAG: The Vertical AI Search Engine

Most search engines help you find information the rest of the world already knows. Light-RAG is different. It is designed for the information that only *you* know. 

In a world where horizontal AI tries to solve every problem for everyone, the real value lies in vertical integration. Light-RAG is a proprietary intelligence layer that transforms your unstructured institutional knowledge into a functional, queryable asset. We didn't build a chatbot; we built a system that bridges the gap between static documents and actionable insight.

The future doesn't belong to those who use the most data, but to those who own the most relevant context. Light-RAG is the infrastructure for that future.

## System Architecture

Light-RAG utilizes a high-performance RAG (Retrieval-Augmented Generation) pipeline:
- **Ingestion:** Automated document processing and indexing.
- **Retrieval:** Vector-based semantic search via ChromaDB.
- **Generation:** Context-aware synthesis using state-of-the-art LLMs via LangChain.

## Getting Started

To deploy the system locally, follow these steps.

### 1. Environment Setup

Clone the repository and initialize a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory. You will need to provide your API keys for the LLM provider (e.g., Groq or OpenAI):

```text
GROQ_API_KEY=your_key_here
# Add other necessary environment variables
```

### 3. Running the Engine

The system consists of a FastAPI backend and a Streamlit frontend.

**Step A: Start the Backend**
```powershell
uvicorn backend.main:app --reload
```

**Step B: Start the Frontend**
```powershell
streamlit run app.py
```

Once both are running, open the Streamlit interface (usually at `localhost:8501`). Upload your documents through the sidebar to index them, and use the chat interface to query your custom knowledge base.

## Usage

1. **Upload:** Use the sidebar to upload PDF or DOCX files.
2. **Index:** The system automatically builds the vector store upon upload.
3. **Query:** Ask complex questions about your specific documents. The system will provide answers derived exclusively from your data.
