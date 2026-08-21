from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.exceptions import LLMError
from app.schemas.query import QueryRequest
from app.schemas.responses import ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: QueryRequest) -> ChatResponse:
    ctx = request.app.state.ctx
    try:
        return await ctx.query_service.execute_chat(payload)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc


@router.post("/search")
async def search(request: Request, payload: QueryRequest):
    ctx = request.app.state.ctx
    return await ctx.query_service.execute_search(payload)
