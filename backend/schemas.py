from pydantic import BaseModel
from typing import Optional

# what the user sends to the API
class ChatRequest(BaseModel):
    query: str
    session_id : Optional[str] = None

# what the API sends back to the user
class ChatResponse(BaseModel):
    answer : str
    session_id : str
    