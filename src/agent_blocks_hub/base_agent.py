from typing import List, Optional, Dict, Any
import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from memory.utils import normalize_messages

logger = logging.getLogger(__name__)


class BaseAgent:
    def __init__(
        self, 
        llm: BaseChatModel, 
        tools: Optional[List[BaseTool]] = None, 
        system_prompt: Optional[str] = None,
        auto_save_memory: bool = True
    ):
        """
        Initialize BaseAgent.
        
        Args:
            llm: Language model to use.
            tools: Optional list of tools.
            system_prompt: Optional default system prompt.
            auto_save_memory: If True, automatically save conversation to memory after invoke.
                            Only works if memory provider has update_memory() method.
        """
        self.llm = llm
        self.tools = list(tools or [])
        self.system_prompt = system_prompt or ""
        self.auto_save_memory = auto_save_memory

    def bind_tools(self, tools: List[BaseTool]) -> None:
        self.tools.extend(tools or [])

    def _compose_system_prompt(self, base_prompt: Optional[str], context: Optional[str]) -> str:
        sp = (base_prompt if base_prompt is not None else self.system_prompt) or ""
        if context:
            return sp + ("\n\n" + context if sp else context)
        return sp

    def _build_langchain_messages(self, msgs: List[Dict[str, str]], sp: Optional[str]) -> List[Any]:
        """Build LangChain messages from normalized message dicts."""
        out: List[Any] = []
        
        # Add system prompt if provided
        if sp:
            out.append(SystemMessage(content=sp))
        
        # Convert normalized messages to LangChain messages
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            
            if role in ("user", "human"):
                out.append(HumanMessage(content=content))
            elif role in ("assistant", "ai", "model"):
                out.append(AIMessage(content=content))
            elif role == "system":
                out.append(SystemMessage(content=content))
            else:
                # Default to human message for unknown roles
                out.append(HumanMessage(content=content))
        
        return out

    def invoke(
        self, 
        state: Dict[str, Any], 
        memory: Optional[Any] = None, 
        system_prompt: Optional[str] = None,
        auto_save: Optional[bool] = None
    ) -> Any:
        """
        Invoke the agent with optional memory integration.
        
        Usage:
            # Simple usage with automatic memory read/write
            agent = BaseAgent(llm=my_llm)
            memory = FMemClient(task_dir="./memory", llm=my_llm)
            
            state = {"messages": [{"role": "user", "content": "Hello"}]}
            response = agent.invoke(state, memory=memory)
            # Memory is automatically read for context and written after response
        
        Args:
            state: The current state (must contain 'messages').
            memory: Optional memory provider (FMemClient or compatible).
                   - Reads context via memory.get_context(messages)
                   - Writes back via memory.update_memory(messages) if auto_save=True
            system_prompt: Optional system prompt override.
            auto_save: Override the default auto_save_memory setting for this call.
        
        Returns:
            LLM response (AIMessage or similar).
        """
        ctx = None
        messages = state.get("messages") or []
        should_save = auto_save if auto_save is not None else self.auto_save_memory
        
        # === STEP 1: Read memory context ===
        if memory is not None:
            # Check for new MemoryProvider interface (get_context with messages param)
            if hasattr(memory, "get_context"):
                try:
                    # Try new interface: get_context(messages, query, max_recent)
                    ctx = memory.get_context(messages=messages)
                    logger.debug("Memory context retrieved via get_context(messages)")
                except TypeError:
                    # Fallback to legacy interface
                    try:
                        # Normalize messages for legacy interface
                        normalized = normalize_messages(messages)
                        ctx = memory.get_context(recent_messages=normalized)
                        logger.debug("Memory context retrieved via legacy get_context(recent_messages)")
                    except Exception as e:
                        logger.warning(f"Failed to get memory context with recent_messages: {e}")
                        # If still fails, just get context without messages
                        try:
                            ctx = memory.get_context()
                            logger.debug("Memory context retrieved via get_context() without messages")
                        except Exception as e2:
                            logger.error(f"Failed to get any memory context: {e2}")
            
            # Check for old FMem interface (update_state)
            elif hasattr(memory, "update_state"):
                try:
                    updated = memory.update_state(dict(state))
                    ctx = updated.get("context")
                    state = updated
                    logger.debug("Memory context retrieved via update_state")
                except Exception as e:
                    logger.error(f"Failed to update state with memory: {e}")

        # === STEP 2: Compose system prompt with memory context ===
        sp = self._compose_system_prompt(system_prompt, ctx)
        
        # === STEP 3: Normalize messages and build LangChain messages ===
        normalized = normalize_messages(messages)
        langchain_msgs = self._build_langchain_messages(normalized, sp)
        
        # === STEP 4: Invoke LLM ===
        try:
            if self.tools:
                response = self.llm.bind_tools(self.tools).invoke(langchain_msgs)
            else:
                response = self.llm.invoke(langchain_msgs)
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            # Fallback or retry logic if needed
            if self.tools:
                response = self.llm.bind_tools(self.tools).invoke(langchain_msgs)
            else:
                response = self.llm.invoke(langchain_msgs)
        
        # === STEP 5: Auto-save to memory (if enabled) ===
        if memory is not None and should_save:
            if hasattr(memory, "update_memory"):
                try:
                    # Include the new response in the conversation history
                    updated_messages = messages + [response]
                    memory.update_memory(updated_messages)
                    logger.debug(f"Memory auto-saved with {len(updated_messages)} messages")
                except Exception as e:
                    logger.error(f"Failed to auto-save memory: {e}")
            else:
                logger.debug("Memory provider does not support update_memory(), skipping auto-save")
        
        return response

    def call_with_memory(
        self, 
        state: Dict[str, Any], 
        memory: Any, 
        system_prompt: Optional[str] = None,
        auto_save: Optional[bool] = None
    ) -> Any:
        """
        Alias for invoke with memory (for backward compatibility).
        
        Args:
            state: The current state.
            memory: Memory provider.
            system_prompt: Optional system prompt override.
            auto_save: Override auto_save_memory setting.
        """
        return self.invoke(state, memory=memory, system_prompt=system_prompt, auto_save=auto_save)


def create_base_agent(
    llm: BaseChatModel, 
    tools: Optional[List[BaseTool]] = None, 
    system_prompt: Optional[str] = None,
    auto_save_memory: bool = True
) -> BaseAgent:
    """
    Factory function to create a BaseAgent.
    
    Args:
        llm: Language model.
        tools: Optional tools.
        system_prompt: Optional system prompt.
        auto_save_memory: Enable automatic memory saving.
    
    Returns:
        BaseAgent instance.
    """
    return BaseAgent(
        llm=llm, 
        tools=tools, 
        system_prompt=system_prompt,
        auto_save_memory=auto_save_memory
    )