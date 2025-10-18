from typing import List, Dict
import json
import os

# File to store user chat histories
CHAT_HISTORY_FILE = "data/user_chat_histories.json"

def load_chat_histories() -> Dict[str, List[Dict[str, str]]]:
    """Load chat histories from file."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def save_chat_histories(conversation_memory: Dict[str, List[Dict[str, str]]]):
    """Save chat histories to file."""
    os.makedirs(os.path.dirname(CHAT_HISTORY_FILE), exist_ok=True)
    with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(conversation_memory, f, ensure_ascii=False, indent=2)

# Load existing chat histories
conversation_memory: Dict[str, List[Dict[str, str]]] = load_chat_histories()

def get_or_create_user_conversation(user_id: str) -> List[Dict[str, str]]:
    """Get or create a conversation for a specific user."""
    if user_id not in conversation_memory:
        conversation_memory[user_id] = []
        save_chat_histories(conversation_memory)
    return conversation_memory[user_id]

def add_to_conversation(user_id: str, role: str, content: str):
    """Add a message to the user's conversation history."""
    conversation = get_or_create_user_conversation(user_id)
    conversation.append({"role": role, "content": content})
    # Keep only last 20 messages to prevent context overflow
    if len(conversation) > 20:
        conversation.pop(0)
    save_chat_histories(conversation_memory)

def get_conversation_context(user_id: str) -> str:
    """Get formatted conversation context for a user."""
    conversation = get_or_create_user_conversation(user_id)
    if not conversation:
        return ""
    
    context = "\n\nPrevious conversation:\n"
    for msg in conversation[-5:]:  # Last 5 messages for context
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role}: {msg['content']}\n"
    
    return context

def get_user_chat_history(user_id: str) -> List[Dict[str, str]]:
    """Get full chat history for a user."""
    return get_or_create_user_conversation(user_id)

def clear_user_chat_history(user_id: str):
    """Clear chat history for a specific user."""
    if user_id in conversation_memory:
        conversation_memory[user_id] = []
        save_chat_histories(conversation_memory)