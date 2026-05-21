"""
F-Mem integration usage examples.

Demonstrates how to use F-Mem with BaseAgent for automatic memory management.
"""

from agent_blocks_hub import BaseAgent
from memory import FMemClient
from llm_provider import get_llm

# ============================================================================
# Example 1: Basic Usage with Auto Memory
# ============================================================================

def example_basic_auto_memory():
    """Most common usage: automatic memory read and write."""
    
    # Setup
    llm = get_llm()
    agent = BaseAgent(llm=llm, auto_save_memory=True)  # Enable auto-save (default)
    memory = FMemClient(task_dir="./my_project_memory", llm=llm)
    
    # Conversation state
    state = {
        "messages": [
            {"role": "user", "content": "What's the project structure?"}
        ]
    }
    
    # Just invoke with memory - that's it!
    # Memory is automatically:
    # 1. Read to provide context
    # 2. Written after response
    response = agent.invoke(state, memory=memory)
    
    print("Response:", response.content)
    # Memory now contains this conversation


# ============================================================================
# Example 2: Manual Memory Control
# ============================================================================

def example_manual_memory_control():
    """Disable auto-save and manually control when to save."""
    
    llm = get_llm()
    agent = BaseAgent(llm=llm, auto_save_memory=False)  # Disable auto-save
    memory = FMemClient(task_dir="./my_memory", llm=llm)
    
    state = {"messages": [{"role": "user", "content": "Hello"}]}
    
    # Read memory but don't auto-save
    response = agent.invoke(state, memory=memory)
    
    # Manually decide when to save
    if is_important_conversation(response):
        memory.update_memory(state["messages"] + [response])


# ============================================================================
# Example 3: Per-Call Memory Control
# ============================================================================

def example_per_call_control():
    """Override auto-save on a per-call basis."""
    
    llm = get_llm()
    agent = BaseAgent(llm=llm, auto_save_memory=True)  # Default: auto-save ON
    memory = FMemClient(task_dir="./memory", llm=llm)
    
    # Save this conversation
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "Important info"}]},
        memory=memory,
        auto_save=True  # Explicitly save
    )
    
    # Don't save this one (e.g., temporary query)
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather?"}]},
        memory=memory,
        auto_save=False  # Skip saving
    )


# ============================================================================
# Example 4: Multi-turn Conversation
# ============================================================================

def example_multi_turn_conversation():
    """Maintain conversation history with memory."""
    
    llm = get_llm()
    agent = BaseAgent(llm=llm)
    memory = FMemClient(task_dir="./chat_memory", llm=llm)
    
    # Track conversation in state
    state = {"messages": []}
    
    # Turn 1
    state["messages"].append({"role": "user", "content": "My name is Alice"})
    response1 = agent.invoke(state, memory=memory)
    state["messages"].append(
        {"role": "assistant", "content": response1.content}
    )
    
    # Turn 2 - memory will remember "Alice"
    state["messages"].append({"role": "user", "content": "What's my name?"})
    response2 = agent.invoke(state, memory=memory)
    # Response will likely include "Alice" from memory context


# ============================================================================
# Example 5: Custom System Prompt with Memory
# ============================================================================

def example_custom_system_prompt():
    """Use custom system prompt alongside memory context."""
    
    llm = get_llm()
    agent = BaseAgent(
        llm=llm,
        system_prompt="You are a helpful coding assistant."
    )
    memory = FMemClient(task_dir="./code_memory", llm=llm)
    
    state = {
        "messages": [
            {"role": "user", "content": "How do I optimize this function?"}
        ]
    }
    
    # System prompt will be: "You are a helpful coding assistant.\n\n<memory context>"
    response = agent.invoke(state, memory=memory)


# ============================================================================
# Example 6: With Tools
# ============================================================================

def example_with_tools():
    """Use memory with tool-calling agents."""
    
    from langchain_core.tools import tool
    
    @tool
    def search_codebase(query: str) -> str:
        """Search the codebase."""
        return f"Found results for: {query}"
    
    llm = get_llm()
    agent = BaseAgent(llm=llm, tools=[search_codebase])
    memory = FMemClient(task_dir="./tool_memory", llm=llm)
    
    state = {
        "messages": [
            {"role": "user", "content": "Find all authentication code"}
        ]
    }
    
    # Agent can use tools AND access memory
    response = agent.invoke(state, memory=memory)


# ============================================================================
# Example 7: Semantic Search with Query
# ============================================================================

def example_semantic_search():
    """Use query parameter to trigger semantic search."""
    
    llm = get_llm()
    agent = BaseAgent(llm=llm)
    memory = FMemClient(task_dir="./project_memory", llm=llm)
    
    # Manually get context with search
    messages = [{"role": "user", "content": "How does authentication work?"}]
    
    # Search for relevant memory about authentication
    context = memory.get_context(
        messages=messages,
        query="authentication implementation"  # Triggers semantic search
    )
    
    # Use in agent
    response = agent.invoke(
        {"messages": messages},
        memory=memory,
        system_prompt=f"Use this context:\n{context}"
    )


# ============================================================================
# Example 8: Configuration
# ============================================================================

def example_with_configuration():
    """Use FMemConfig for fine-grained control."""
    
    from memory import FMemConfig
    import logging
    
    # Custom configuration
    config = FMemConfig(
        buffer_size=20,              # Save every 20 messages
        max_recent_messages=15,       # Include last 15 messages in context
        max_file_size=5*1024*1024,   # 5MB file size limit
        log_level=logging.DEBUG,      # Verbose logging
        enable_validation=True        # Validate LLM tool calls
    )
    
    llm = get_llm()
    agent = BaseAgent(llm=llm)
    memory = FMemClient(task_dir="./memory", llm=llm, config=config)
    
    state = {"messages": [{"role": "user", "content": "Hello"}]}
    response = agent.invoke(state, memory=memory)


# ============================================================================
# Utility Helper
# ============================================================================

def is_important_conversation(response) -> bool:
    """Determine if conversation should be saved."""
    # Your logic here
    return len(response.content) > 50


if __name__ == "__main__":
    print("F-Mem Usage Examples")
    print("====================\n")
    
    print("Example 1: Basic auto memory")
    example_basic_auto_memory()
    
    print("\nExample 2: Manual control")
    example_manual_memory_control()
    
    print("\nSee source code for more examples!")
