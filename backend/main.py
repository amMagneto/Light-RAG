import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File

try:
    # Package-style imports (e.g. uvicorn backend.main:app)
    from .schemas import ChatRequest, ChatResponse
    from .db_utils import create_application_logs, insert_application_logs, get_chat_history
    from .langchain_utils import get_conversational_rag_chain
    from .chroma_utils import build_vector_store, load_and_chunk_document
except ImportError:
    # Local-module imports (e.g. from backend dir: uvicorn main:app)
    from schemas import ChatRequest, ChatResponse
    from db_utils import create_application_logs, insert_application_logs, get_chat_history
    from langchain_utils import get_conversational_rag_chain
    from chroma_utils import build_vector_store, load_and_chunk_document

app = FastAPI()

# Define the absolute path for the docs folder used for ingestion uploads.
UPLOAD_DIR = Path(__file__).resolve().parent / "docs"
UPLOAD_DIR.mkdir(exist_ok=True)

# 1. startip logic: initialise DB and Load RAG Chain
create_application_logs()
vectorstore = build_vector_store(str(UPLOAD_DIR))
rag_chain = get_conversational_rag_chain(vectorstore)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """The ingestion trigger: save a file and refresh the vector DB."""
    safe_name = Path(file.filename or "uploaded_file").name
    file_path = UPLOAD_DIR / safe_name

    # 1. Save the uploaded file to local disk.
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Add the document to the vectorstore directly
    chunks = load_and_chunk_document(str(file_path))
    
    global vectorstore, rag_chain
    if chunks:
        vectorstore.add_documents(chunks)
        # Refresh chain with updated vectorstore
        rag_chain = get_conversational_rag_chain(vectorstore)
    else:
        return {
            "status": "skipped",
            "filename": safe_name,
            "chunks": 0,
            "reason": "unsupported_or_empty_document"
        }

    return {
        "status": "indexed",
        "filename": safe_name,
        "chunks": len(chunks)
    }

@app.get("/debug/chunks")
def debug_chunks():
    """Debug endpoint to verify ChromaDB contents."""
    global vectorstore
    try:
        col = vectorstore._collection
        return {
            "collection_name": col.name,
            "chunk_count": col.count(),
            "sample_metadata": col.peek(3)["metadatas"] if col.count() > 0 else []
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/reset-index")
def reset_index():
    """Clear vector index and rebuild from files in backend/docs."""
    global vectorstore, rag_chain
    try:
        vectorstore.delete_collection()
    except Exception:
        pass

    vectorstore = build_vector_store(str(UPLOAD_DIR))
    rag_chain = get_conversational_rag_chain(vectorstore)

    return {
        "status": "rebuilt",
        "chunk_count": vectorstore._collection.count(),
        "docs_dir": str(UPLOAD_DIR)
    }

@app.post("/chat", response_model = ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # generate a new session id if not provided
    session_id = request.session_id or str(uuid.uuid4())

    # 1: retrieve chat history from DB
    history = get_chat_history(session_id)

    # 2: Invoke conversational RAG chain (it's history aware)
    response = rag_chain.invoke({
        "input":request.query,
        "chat_history": history
    })
    context_docs = response.get("context", [])
    if not context_docs:
        answer = "I don't know based on the provided sources."
    else:
        answer = response["answer"]

    # 3: persist this interaction in DB
    insert_application_logs(session_id, request.query, answer, "gpt-oss-120b")

    return ChatResponse(answer= answer, session_id = session_id)



