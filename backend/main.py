import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from .schemas import ChatRequest, ChatResponse
from .db_utils import create_application_logs, insert_application_logs, get_chat_history
from .langchain_utils import get_conversational_rag_chain
from .chroma_utils import build_vector_store

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

    # 2. Re-run ingestion and refresh global RAG objects.
    global vectorstore, rag_chain
    vectorstore = build_vector_store(str(UPLOAD_DIR))
    rag_chain = get_conversational_rag_chain(vectorstore)

    return {"message": f"Successfully indexed {safe_name}"}

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



