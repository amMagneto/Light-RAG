import sqlite3
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

DB_NAME= "rag_app.db"

def get_db_connection():
    # row factory allows us to access columns by name
    conn= sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS application_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_query TEXT,
            gpt_response TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            chunk_size INTEGER,
            chunk_overlap INTEGER,
            page_count INTEGER,
            file_size_kb REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()
def insert_application_logs(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?,?,?,?)",
        (session_id, user_query, gpt_response, model)
    )
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at",
        (session_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    # TRANSFORM ROWS INTO MESSAGE OBJECTS
    messages = []
    for row in rows:
        messages.append(HumanMessage(content=row['user_query']))
        messages.append(AIMessage(content=row['gpt_response']))
        
        
    return messages

def insert_document_metadata(filename, chunk_size, chunk_overlap, page_count, file_size_kb):
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO document_metadata (filename, chunk_size, chunk_overlap, page_count, file_size_kb)
        VALUES (?, ?, ?, ?, ?)
        """,
        (filename, chunk_size, chunk_overlap, page_count, file_size_kb)
    )
    conn.commit()
    conn.close()
