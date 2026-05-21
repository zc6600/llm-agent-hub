"""
Utility functions for memory systems.

This module provides common utilities for message normalization and other
shared operations across different memory implementations.
"""

from typing import Any, Dict, List


def normalize_message(msg: Any) -> Dict[str, str]:
    """
    Normalize a message object or dict to a standard dict format.
    
    This function handles various message formats:
    - Dict with 'role' and 'content' keys
    - Objects with 'content' attribute (LangChain messages)
    - Objects with 'type' attribute (e.g., HumanMessage, AIMessage)
    
    Args:
        msg: Message to normalize (dict or object).
    
    Returns:
        Dict with 'role' and 'content' keys.
    
    Examples:
        >>> normalize_message({"role": "user", "content": "Hello"})
        {'role': 'user', 'content': 'Hello'}
        
        >>> class HumanMessage:
        ...     type = "human"
        ...     content = "Hi"
        >>> normalize_message(HumanMessage())
        {'role': 'user', 'content': 'Hi'}
    """
    role = "user"
    content = ""
    
    # Handle dict format
    if isinstance(msg, dict):
        role = msg.get("role", "user")
        content = msg.get("content") or ""
    
    # Handle object with content attribute (LangChain messages, etc.)
    elif hasattr(msg, "content"):
        content = str(msg.content)
        
        # Try to infer role from 'type' attribute
        if hasattr(msg, "type"):
            msg_type = str(msg.type).lower()
            if msg_type in ("human", "user"):
                role = "user"
            elif msg_type in ("ai", "assistant", "model"):
                role = "assistant"
            elif msg_type == "system":
                role = "system"
            else:
                role = "user"  # Default to user for unknown types
        
        # Fallback to 'role' attribute if 'type' doesn't exist
        elif hasattr(msg, "role"):
            role = str(msg.role)
    
    # Handle tuple format (role, content)
    elif isinstance(msg, (tuple, list)) and len(msg) == 2:
        role = str(msg[0])
        content = str(msg[1])
    
    # Fallback: treat as string content
    else:
        content = str(msg)
        role = "user"
    
    return {"role": role, "content": content}


def normalize_messages(msgs: List[Any]) -> List[Dict[str, str]]:
    """
    Normalize a list of messages to standard dict format.
    
    Args:
        msgs: List of messages in any format.
    
    Returns:
        List of dicts with 'role' and 'content' keys.
    
    Examples:
        >>> normalize_messages([
        ...     {"role": "user", "content": "Hi"},
        ...     {"role": "assistant", "content": "Hello"}
        ... ])
        [{'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello'}]
    """
    return [normalize_message(msg) for msg in msgs or []]


def format_messages_as_text(msgs: List[Dict[str, str]]) -> str:
    """
    Format normalized messages as readable text.
    
    Args:
        msgs: List of normalized message dicts.
    
    Returns:
        Formatted string with one message per line.
    
    Examples:
        >>> format_messages_as_text([
        ...     {"role": "user", "content": "Hi"},
        ...     {"role": "assistant", "content": "Hello"}
        ... ])
        '- user: Hi\\n- assistant: Hello'
    """
    lines = []
    for msg in msgs or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:  # Only include non-empty messages
            lines.append(f"- {role}: {content}")
    return "\n".join(lines)
