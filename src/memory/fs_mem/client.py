import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..utils import normalize_message, normalize_messages, format_messages_as_text
from ..config import FMemConfig, DEFAULT_CONFIG

from .core.context_builder import ContextBuilder
from .core.memory_searcher import MemorySearcher
from .core.memory_writer import MemoryWriter
from .core.directory_manager import DirectoryManager

logger = logging.getLogger(__name__)


class FMemClient:
    """
    A standalone client for File System Memory (F-Mem).
    
    This client implements the MemoryProvider protocol and handles:
    1. Task directory management (creation, structure ensuring).
    2. Context retrieval (reading from memory).
    3. Memory writing (saving important information from conversation).
    4. Memory search.
    
    Implements the MemoryProvider protocol for compatibility with agents.
    """

    def __init__(
        self, 
        task_dir: str, 
        llm: Optional[Any] = None, 
        config: Optional[FMemConfig] = None
    ):
        """
        Initialize the FMemClient.

        Args:
            task_dir: The directory where memory will be stored.
            llm: Optional LLM instance for writing/summarizing memory. 
                 Required for update_memory/save_context.
            config: Optional F-Mem configuration. If not provided, uses default config.
        """
        self.task_dir = task_dir
        self.llm = llm
        self.config = config or DEFAULT_CONFIG
        
        # Configure logging
        logger.setLevel(self.config.log_level)
        
        # Ensure basic structure exists
        self._ensure_task_structure()
        
        # Initialize core components
        self._cb = ContextBuilder(task_dir, config=self.config)
        self._ms = MemorySearcher(
            task_dir, 
            llm=llm, 
            summary_length=self.config.summary_length,
            config=self.config
        )
        self._mw = MemoryWriter(
            task_dir, 
            llm=llm, 
            buffer_size=self.config.buffer_size,
            config=self.config
        ) if llm else None
        self._dm = DirectoryManager(
            task_dir=task_dir, 
            base_path=os.path.dirname(task_dir), 
            llm=llm,
            config=self.config
        )
        
        logger.info(f"FMemClient initialized with task_dir: {task_dir}")

    def _ensure_task_structure(self):
        """Ensure the task directory and subdirectories exist."""
        Path(self.task_dir).mkdir(parents=True, exist_ok=True)
        for category in self.config.memory_categories:
            Path(self.task_dir, category).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Task structure ensured for: {self.task_dir}")

    def get_context(
        self, 
        messages: List[Any],
        query: Optional[str] = None,
        max_recent: int = None
    ) -> str:
        """
        Get the formatted context string, including fixed memory and recent conversation.
        
        This is the new interface that implements the MemoryProvider protocol.

        Args:
            messages: List of messages in any format (will be normalized internally).
            query: Optional query to trigger semantic search in memory.
            max_recent: Maximum number of recent messages to include. 
                       Uses config default if not provided.

        Returns:
            A formatted context string to be added to system prompt.
        """
        if max_recent is None:
            max_recent = self.config.max_recent_messages
        
        # Normalize all messages
        normalized = normalize_messages(messages)
        
        # Get fixed context from memory files
        fixed_ctx = self._cb.build_context()
        
        # Get recent messages
        recent = normalized[-max_recent:] if normalized else []
        recent_block = format_messages_as_text(recent)
        
        # Perform semantic search if query is provided
        search_results = ""
        if query:
            try:
                search_results = self.search(query)
                logger.debug(f"Semantic search performed for query: {query}")
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")
        
        # Format final context
        parts = []
        
        if fixed_ctx:
            parts.append(f"=== Current State and Laws to Obey ===\n{fixed_ctx}")
        
        if recent_block:
            parts.append(f"=== Recent Conversation (last {len(recent)}) ===\n{recent_block}")
        
        if search_results:
            parts.append(f"=== Relevant Memory (searched: {query}) ===\n{search_results}")
        
        context = "\n\n".join(parts)
        logger.debug(f"Context built: {len(context)} characters")
        
        return context

    def update_memory(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """
        Update memory with new messages (implements MemoryProvider protocol).
        
        Analyzes the conversation and writes important information to memory.
        This is the new standardized method name.
        
        Args:
            messages: List of messages to analyze and potentially save.
            
        Returns:
            List of files written/modified.
        """
        if not self._mw:
            raise ValueError(
                "LLM is required for updating memory. "
                "Pass 'llm' to FMemClient constructor."
            )
        
        # Normalize messages
        normalized = normalize_messages(messages)
        
        # Add messages to writer buffer
        for msg in normalized:
            if msg["content"]:
                self._mw.add_message(msg["role"], msg["content"])
        
        # Trigger write
        files_written = self._mw.write_memory()
        logger.info(f"Memory updated: {len(files_written)} files written")
        
        return files_written

    def save_context(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """
        Save conversation context to memory (legacy method name).
        
        This is an alias for update_memory() to maintain backward compatibility.
        
        Args:
            messages: List of messages to save.
            
        Returns:
            List of files written/modified.
        """
        logger.warning(
            "save_context() is deprecated, use update_memory() instead. "
            "save_context() will be removed in a future version."
        )
        return self.update_memory(messages)

    def search(self, query: str) -> str:
        """
        Search the memory for relevant information (implements MemoryProvider protocol).
        
        Args:
            query: The search query.
            
        Returns:
            Search results as a string.
        """
        logger.debug(f"Searching memory for: {query}")
        return self._ms.search_memory(query)

    def create_task_dir(
        self, 
        task_description: str, 
        llm_response: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new task directory based on description.
        
        Args:
            task_description: Description of the task.
            llm_response: Optional pre-computed LLM response for directory creation.
            
        Returns:
            Path to the created directory.
        """
        state = {"task_description": task_description}
        path = self._dm.create_task_directory(state, llm_response=llm_response)
        logger.info(f"Task directory created: {path}")
        return path

    # Legacy interface support (for backward compatibility with BaseAgent)
    def get_context_legacy(
        self, 
        query: Optional[str] = None, 
        recent_messages: List[Any] = None, 
        max_recent: int = 10
    ) -> str:
        """
        Legacy get_context interface for backward compatibility.
        
        This method supports the old signature where recent_messages is passed separately.
        
        Args:
            query: Optional query parameter (not used in current implementation).
            recent_messages: List of recent messages.
            max_recent: Maximum number of recent messages to include.
            
        Returns:
            Formatted context string.
        """
        logger.warning(
            "Using legacy get_context interface. "
            "Consider migrating to new interface: get_context(messages, query, max_recent)"
        )
        
        if recent_messages is None:
            recent_messages = []
        
        return self.get_context(
            messages=recent_messages,
            query=query,
            max_recent=max_recent
        )
