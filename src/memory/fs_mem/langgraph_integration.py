import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..utils import normalize_messages

logger = logging.getLogger(__name__)

# Backward compatibility - keep the old function but mark as deprecated
def convert_langgraph_messages(messages: List[Any]) -> List[Dict[str, str]]:
    """
    DEPRECATED: Use memory.utils.normalize_messages instead.
    
    This function is kept for backward compatibility only.
    """
    logger.warning(
        "convert_langgraph_messages is deprecated. "
        "Use memory.utils.normalize_messages instead."
    )
    return normalize_messages(messages)


class FMemAgent:
    def __init__(self, task_dir: Optional[str] = None, auto_init: bool = True, max_messages: int = 10):
        if task_dir is None:
            task_dir = os.getenv("FMEM_TASK_DIR") or os.getenv("F_MEM_TASK_DIR") or "./agent_memory"
        self.task_dir = str(Path(task_dir).resolve())
        self.max_messages = max_messages
        os.environ["FMEM_TASK_DIR"] = self.task_dir
        os.environ["F_MEM_TASK_DIR"] = self.task_dir
        
        # Import here to avoid circular dependency
        from . import FMemClient
        
        if auto_init:
            # Ensure task structure
            Path(self.task_dir).mkdir(parents=True, exist_ok=True)
        
        # Use FMemClient instead of legacy FMem
        self._client = FMemClient(task_dir=self.task_dir)
        logger.info(f"FMemAgent initialized with task_dir: {self.task_dir}")

    def sync_messages(self, langgraph_state: Dict[str, Any]) -> Dict[str, Any]:
        """Sync messages from LangGraph state (deprecated method)."""
        logger.warning("sync_messages is deprecated and does nothing in new implementation")
        return {}

    def get_context(self, langgraph_state: Dict[str, Any]) -> str:
        """Get memory context for LangGraph state."""
        msgs = langgraph_state.get("messages", [])
        return self._client.get_context(messages=msgs, max_recent=self.max_messages)

    def call_with_memory(
        self, 
        langgraph_state: Dict[str, Any], 
        llm: Any, 
        system_prompt: Optional[str] = None, 
        **llm_kwargs
    ) -> Any:
        """Call LLM with memory context."""
        from langchain_core.messages import SystemMessage
        
        context = self.get_context(langgraph_state)
        
        # Compose system prompt
        if system_prompt is None:
            system_prompt = ""
        full_system_prompt = system_prompt + ("\n\n" + context if context else "")
        
        # Build messages
        messages = [SystemMessage(content=full_system_prompt)]
        messages.extend(langgraph_state.get("messages", []))
        
        # Invoke LLM
        if "tools" in llm_kwargs:
            response = llm.bind_tools(llm_kwargs["tools"]).invoke(messages)
        else:
            response = llm.invoke(messages)
        
        # Update memory if client has LLM
        if self._client.llm:
            try:
                all_msgs = langgraph_state.get("messages", []) + [response]
                self._client.update_memory(all_msgs)
            except Exception as e:
                logger.error(f"Failed to update memory: {e}")
        
        return response

    def search(self, query: str) -> str:
        """Search memory."""
        return self._client.search(query)

    def get_state(self) -> Dict[str, Any]:
        """Get internal state (deprecated)."""
        logger.warning("get_state is deprecated in new implementation")
        return {}


def create_fmem_tools(task_dir: Optional[str] = None) -> List:
    from langchain_core.tools import tool
    
    if task_dir is None:
        task_dir = os.getenv("FMEM_TASK_DIR") or os.getenv("F_MEM_TASK_DIR") or "./agent_memory"
    
    # Import here to avoid circular dependency
    from . import FMemClient
    client = FMemClient(task_dir=task_dir)
    
    @tool
    def search_project_memory(query: str) -> str:
        """Search the project memory for relevant information."""
        try:
            result = client.search(query)
            return "Memory Search Results:\n\n" + result
        except Exception as e:
            logger.error(f"Search tool failed: {e}")
            return "Error: " + str(e)
    
    @tool
    def save_project_decision(decision: str, category: str = "knowledge") -> str:
        """Save an important project decision or information to memory."""
        try:
            p = Path(task_dir) / category
            p.mkdir(parents=True, exist_ok=True)
            import re, time
            fn = re.sub(r"[^\w\-]+", "_", decision.strip())[:50] or f"decision_{int(time.time())}"
            fp = p / (fn + ".txt")
            fp.write_text(decision, encoding="utf-8")
            logger.info(f"Saved decision to: {fp}")
            return "✓ Decision saved to " + str(fp)
        except Exception as e:
            logger.error(f"Save tool failed: {e}")
            return "Error: " + str(e)
    
    return [search_project_memory, save_project_decision]