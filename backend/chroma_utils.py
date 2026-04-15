import os
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# initialise the embedding model
embedding_function = HuggingFaceEmbeddings(model_name= "intfloat/e5-base-v2")
# configure the splitter
text_splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# initialise lazy laoding of documents
def lazy_load_documents(docs_path):
    for filename in os.listdir(docs_path):
        file_path = os.path.join(docs_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue
        #lazy page level
        for doc in loader.lazy_load():
            yield doc

# initialise the vector store
def build_vector_store(docs_path, batch_size=50):
    vectorstore = Chroma(
        embedding_function=embedding_function,
        persist_directory="./chroma_db"
    )
    batch=[]
    for doc in lazy_load_documents(docs_path):
        chunks = text_splitter.split_documents([doc])
        batch.extend(chunks)
        if len(batch) >= batch_size:
            vectorstore.add_documents(batch)
            batch=[]
    if batch:
        vectorstore.add_documents(batch)
    
    return vectorstore