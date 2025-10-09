from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import uuid

from app.llm import setup_qa_chain
from app.vectorstore import retriever, vectorstore
from app.memory import add_to_conversation, get_conversation_context
from app.helpers import strip_markdown
from config import SYSTEM_PROMPT


router = APIRouter()

qa_chain = setup_qa_chain(retriever)

class ChatRequest(BaseModel):
    question: str
    session_id: str = None

@router.post("/chat")
async def chat(request: Request):
    """Chat endpoint: returns full answer from vectorstore."""
    data = await request.json()
    question = data.get("question", "")
    session_id = data.get("session_id", str(uuid.uuid4()))

    print("User Q:", question)
    print("Session ID:", session_id)

    # Add user question to conversation
    add_to_conversation(session_id, "user", question)

    # Get conversation context
    conversation_context = get_conversation_context(session_id)

    # Use just the question for the new system (conversation context is handled by the system prompt)
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    print("Answer:", answer)

    # Add bot response to conversation
    add_to_conversation(session_id, "assistant", answer)

    clean_answer = strip_markdown(answer)
    return {"answer": clean_answer, "session_id": session_id}

# ---------------- Streaming Chat Endpoint ----------------

@router.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    question = data.get("question", "")
    session_id = data.get("session_id", str(uuid.uuid4()))

    # Add user question to conversation
    add_to_conversation(session_id, "user", question)

    # Get conversation context
    conversation_context = get_conversation_context(session_id)

    # Use just the question for the new system (conversation context is handled by the system prompt)
    result = qa_chain.invoke({"query": question})
    answer = result["result"]

    # Add bot response to conversation
    add_to_conversation(session_id, "assistant", answer)

    clean_answer = strip_markdown(answer)
    return PlainTextResponse(content=clean_answer)