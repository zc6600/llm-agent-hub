import os
import logging
from typing import Any, Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from ..utils import normalize_message

logger = logging.getLogger(__name__)


class FMemChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self, 
        task_dir: str, 
        *, 
        summary_length: int = 500, 
        max_messages: int = 10, 
        auto_ensure_structure: bool = True
    ):
        self.task_dir = task_dir
        self.summary_length = summary_length
        self.max_messages = max_messages
        
        # Import here to avoid circular dependency
        from .. import FMemClient
        
        if auto_ensure_structure:
            from pathlib import Path
            Path(task_dir).mkdir(parents=True, exist_ok=True)
        
        # Use new FMemClient
        self._client = FMemClient(task_dir=task_dir)
        self._messages: List[BaseMessage] = []
        
        logger.info(f"FMemChatMessageHistory initialized for: {task_dir}")

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        
        # Periodically update memory
        if len(self._messages) % self.max_messages == 0 and self._client.llm:
            try:
                self._client.update_memory(self._messages)
            except Exception as e:
                logger.error(f"Failed to update memory: {e}")

    def add_user_message(self, message: str) -> None:
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        self._messages.clear()

    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages

    def get_fmem_context(self, query: Optional[str] = None) -> str:
        """Get F-Mem context."""
        return self._client.get_context(
            messages=self._messages,
            query=query,
            max_recent=self.max_messages
        )

    def search_memory(self, query: str) -> str:
        """Search memory."""
        return self._client.search(query)


def _messages_to_string(messages: List[BaseMessage]) -> str:
    """Convert LangChain messages to string (utility function)."""
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"AI: {msg.content}")
        elif isinstance(msg, SystemMessage):
            lines.append(f"System: {msg.content}")
        else:
            lines.append(f"{msg.type}: {msg.content}")
    return "\n".join(lines)