"""
Memory system protocols and interfaces.

This module defines the standard interfaces for memory providers,
enabling type checking and clear interface contracts.
"""

from typing import Protocol, List, Dict, Any, Optional


class MessageLike(Protocol):
    """
    Protocol for message-like objects.
    
    Any object with 'role' and 'content' attributes can be treated as a message.
    This includes both dict-like objects and structured message classes.
    """
    role: str
    content: str


class MemoryProvider(Protocol):
    """
    Standard interface for memory providers.
    
    This protocol defines the contract that all memory systems should implement
    to be compatible with agents and other components.
    """
    
    def get_context(
        self, 
        messages: List[Any],
        query: Optional[str] = None,
        max_recent: int = 10
    ) -> str:
        """
        Get formatted context string from memory.
        
        Args:
            messages: List of messages (can be dicts, objects, or any format).
                     The implementation should normalize them internally.
            query: Optional query to tailor the context or trigger semantic search.
            max_recent: Maximum number of recent messages to include.
        
        Returns:
            Formatted context string to be added to the system prompt.
        """
        ...
    
    def update_memory(
        self, 
        messages: List[Any]
    ) -> None:
        """
        Update memory with new messages.
        
        Args:
            messages: List of messages to analyze and potentially save to memory.
        """
        ...
    
    def search(
        self, 
        query: str
    ) -> str:
        """
        Search memory for relevant information.
        
        Args:
            query: The search query.
        
        Returns:
            Search results as a string.
        """
        ...
