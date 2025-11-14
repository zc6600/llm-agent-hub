"""
Utility functions for the Deep Diver agent.
"""

from typing import List, Dict, Any, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


def format_messages_for_llm(messages: List[Union[Dict[str, Any], BaseMessage]]) -> str:
    """
    Format messages for LLM consumption.
    
    Args:
        messages: List of message dictionaries or BaseMessage objects
        
    Returns:
        Formatted string
    """
    formatted_parts = []
    
    for msg in messages:
        if isinstance(msg, BaseMessage):
            role = msg.__class__.__name__.replace("Message", "").lower()
            content = msg.content
        elif isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
        else:
            continue
        
        formatted_parts.append(f"{role.upper()}: {content}")
    
    return "\n\n".join(formatted_parts)


def extract_user_question(messages: List[Union[Dict[str, Any], BaseMessage]]) -> str:
    """
    Extract the user's question from messages.
    
    Args:
        messages: List of message dictionaries or BaseMessage objects
        
    Returns:
        User's question string
    """
    # Find the last user/human message
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
        elif isinstance(msg, dict):
            if msg.get("role") in ["user", "human"]:
                return msg.get("content", "")
    
    # If no user message found, return empty string or first message
    if messages:
        if isinstance(messages[0], BaseMessage):
            return messages[0].content
        elif isinstance(messages[0], dict):
            return messages[0].get("content", "")
    
    return ""



def format_experience_pool(experience_pool: List[Dict[str, Any]]) -> str:
    """
    Format experience pool for display or further processing.
    
    Args:
        experience_pool: List of experience entries
        
    Returns:
        Formatted string
    """
    if not experience_pool:
        return "Experience pool is empty."
    
    formatted_lines = ["Experience Pool:", "=" * 50]
    
    for i, exp in enumerate(experience_pool, 1):
        hypothesis = exp.get("hypothesis", "Unknown")
        result = exp.get("result", "unknown")
        confidence = exp.get("confidence", 0.0)
        iteration = exp.get("iteration", 0)
        evidence = exp.get("evidence", [])
        
        formatted_lines.append(f"\n{i}. {hypothesis}")
        formatted_lines.append(f"   Status: {result}")
        formatted_lines.append(f"   Confidence: {confidence:.2f}")
        formatted_lines.append(f"   Iteration: {iteration}")
        
        if evidence:
            formatted_lines.append(f"   Evidence: {len(evidence)} items")
    
    return "\n".join(formatted_lines)


def should_refine_hypothesis(
    hypothesis: str,
    verification_result: str,
    confidence: float
) -> bool:
    """
    Determine if a hypothesis should be refined rather than accepted/rejected.
    
    Args:
        hypothesis: The hypothesis content
        verification_result: Result of verification
        confidence: Confidence score
        
    Returns:
        True if hypothesis should be refined, False otherwise
    """
    # Refine if confidence is in the uncertain range (0.4 - 0.6)
    if 0.4 <= confidence <= 0.6:
        return True
    
    # Refine if verification result is uncertain
    if verification_result == "uncertain":
        return True
    
    # Refine if confidence is moderate but doesn't match verification
    if verification_result == "accepted" and confidence < 0.6:
        return True
    
    if verification_result == "rejected" and confidence > 0.4:
        return True
    
    return False
