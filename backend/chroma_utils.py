import os
from pathlib import Path
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# Suppress benign warnings if needed (optional)
# from transformers import logging as transformers_logging
# transformers_logging.set_verbosity_error()

try:
    from .db_utils import insert_document_metadata
except ImportError:
    from db_utils import insert_document_metadata

# initialise the embedding model
embedding_function = HuggingFaceEmbeddings(model_name= "intfloat/e5-base-v2")

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "light_rag_docs"

def get_chunking_config(file_size_kb: int, page_count: int = None) -> dict:
    """Returns chunk_size and overlap based on document size."""
    if page_count:
        if page_count <= 2:          # Resume, letter, single page
            return {"chunk_size": 300, "chunk_overlap": 50}
        elif page_count <= 15:       # Reports, proposals, articles
            return {"chunk_size": 700, "chunk_overlap": 150}
        elif page_count <= 100:      # Manuals, theses, long reports
            return {"chunk_size": 1200, "chunk_overlap": 250}
        else:                        # Books, legal volumes
            return {"chunk_size": 2000, "chunk_overlap": 400}

    # Fallback: use file size if page count unavailable
    if file_size_kb < 50:
        return {"chunk_size": 300, "chunk_overlap": 50}
    elif file_size_kb < 500:
        return {"chunk_size": 700, "chunk_overlap": 150}
    elif file_size_kb < 5000:
        return {"chunk_size": 1200, "chunk_overlap": 250}
    else:
        return {"chunk_size": 2000, "chunk_overlap": 400}

# initialise lazy laoding of documents
def load_and_chunk_document(file_path: str) -> list:
    file_size_kb = os.path.getsize(file_path) / 1024
    ext = file_path.split(".")[-1].lower()

    # Load
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        page_count = len(docs)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        page_count = None
    else:
        return []

    # Get dynamic config
    config = get_chunking_config(file_size_kb, page_count)
    
    # Store config in SQLite for traceability
    insert_document_metadata(
        os.path.basename(file_path), 
        config["chunk_size"], 
        config["chunk_overlap"], 
        page_count, 
        file_size_kb
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=["\n\n\n", "\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.page_content = f"passage: {chunk.page_content}"
        chunk.metadata["chunk_size_used"] = config["chunk_size"]
        chunk.metadata["file_size_kb"] = round(file_size_kb, 2)

    return chunks

# initialise the vector store
def build_vector_store(docs_path):
    vectorstore = Chroma(
        embedding_function=embedding_function,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    
    # Avoid duplication: skip ingestion if store is already populated
    if vectorstore._collection.count() > 0:
        return vectorstore

    all_chunks = []
    for filename in os.listdir(docs_path):
        file_path = os.path.join(docs_path, filename)
        if os.path.isfile(file_path):
            chunks = load_and_chunk_document(file_path)
            all_chunks.extend(chunks)
    
    if all_chunks:
        vectorstore.add_documents(all_chunks)
    
    return vectorstore