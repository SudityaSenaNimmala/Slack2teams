from typing import List, Dict

conversation_memory: Dict[str, List[Dict[str, str]]] = {}

def get_or_create_session(session_id: str) -> List[Dict[str, str]]:
    """Get or create a conversation session."""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    return conversation_memory[session_id]

def add_to_conversation(session_id: str, role: str, content: str):
    """Add a message to the conversation history."""
    conversation = get_or_create_session(session_id)
    conversation.append({"role": role, "content": content})
    # Keep only last 10 messages to prevent context overflow
    if len(conversation) > 10:
        conversation.pop(0)

def get_conversation_context(session_id: str) -> str:
    """Get formatted conversation context."""
    conversation = get_or_create_session(session_id)
    if not conversation:
        return ""
    
    context = "\n\nPrevious conversation:\n"
    for msg in conversation[-5:]:  # Last 5 messages for context
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role}: {msg['content']}\n"
    
    return context