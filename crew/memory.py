"""Session-Based Multi-Turn Conversation Memory for Nykaa Support Agent (Task 8).

Implements:
- LangChain InMemoryChatMessageHistory keyed by session_id.
- Multi-turn state tracking within a process.
- Fresh conversation state isolation / session reset.
"""

from typing import Dict, List, Optional
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# In-memory dictionary store mapping session_id -> InMemoryChatMessageHistory
_SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retrieves or creates the InMemoryChatMessageHistory for a given session."""
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


def add_user_message(session_id: str, message: str) -> None:
    """Appends a user message to the session history."""
    history = get_session_history(session_id)
    history.add_user_message(message)


def add_ai_message(session_id: str, message: str) -> None:
    """Appends an AI message to the session history."""
    history = get_session_history(session_id)
    history.add_ai_message(message)


def get_session_messages(session_id: str) -> List[BaseMessage]:
    """Returns all message objects for a session."""
    history = get_session_history(session_id)
    return history.messages


def format_session_history(session_id: str) -> str:
    """Formats the session history into a readable multi-turn string for agent context."""
    messages = get_session_messages(session_id)
    if not messages:
        return ""

    formatted_lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_lines.append(f"Assistant: {msg.content}")
        else:
            formatted_lines.append(f"{msg.type.capitalize()}: {msg.content}")

    return "\n".join(formatted_lines)


def clear_session(session_id: str) -> None:
    """Clears all conversation history for a given session."""
    if session_id in _SESSION_STORE:
        _SESSION_STORE[session_id].clear()
        del _SESSION_STORE[session_id]


def extract_contextual_order_id(session_id: str, current_query: str) -> Optional[str]:
    """Resolves an order ID from the current query or falls back to session history."""
    import re
    # Check current query first
    match = re.search(r"\b(ORD-\d{4})\b", current_query, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Otherwise scan prior messages in the session
    history_text = format_session_history(session_id)
    history_match = re.search(r"\b(ORD-\d{4})\b", history_text, re.IGNORECASE)
    if history_match:
        return history_match.group(1).upper()

    return None

