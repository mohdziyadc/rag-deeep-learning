import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from config.config import get_settings
from generation.llm_client import LLMClient
from generation.generator import Generator
from retrieval.bridge import RetrievalBridge
from models.dtos import ChatRequest


router = APIRouter()
settings = get_settings()
llm = LLMClient(settings.llm_api_key, settings.llm_base_url, settings.llm_model)
generator = Generator(llm, settings.llm_model, settings.max_tokens)
retriever = RetrievalBridge()


@router.post("/chat")
async def chat(req: ChatRequest):
    chunks = await retriever.retrieve(query=req.question, top_k=req.top_k)

    if req.stream:
        async def event_stream():
            for event in generator.stream(
                question=req.question,
                messages=req.messages,
                chunks=chunks,
                quote=req.qoute
            ):
                yield f"data: {json.dumps(event)}\n\n"
        
        """
        Each SSE message looks like. This can be used by FE and use a string buffer to display it as a continous stream for end user
        data: {"type":"delta","data":"the fluctuations"}
        data: {"type":"delta","data":" in population"}
        data: {"type":"delta","data":" is because of"}
        """
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    # if no stream, return final answer 
    response = generator.generate(req.question, req.messages, chunks, req.qoute)
    return response