"""
Parallel Agent - Unified interface for parallel execution with multiple strategies.

This module provides a unified interface for executing tasks in parallel with different
execution strategies optimized for different use cases and LLM capabilities.

Execution Strategies:
--------------------
1. **direct** (RECOMMENDED for DeepSeek V3 and models with poor tool calling)
   - Directly calls tool.run() without LLM tool calling
   - Fastest and most reliable for simple information gathering
   - No LLM needed if summarization is disabled
   
2. **tool_calling** (for GPT-4, Claude with good tool calling support)
   - LLM decides when/how to call tools using llm.bind_tools()
   - More intelligent tool usage decisions
   - Requires model with good tool calling support
   
3. **react** (for complex reasoning requiring multi-step thought chains)
   - Full ReAct reasoning loops with observation-thought-action cycles
   - Most comprehensive but slowest
   - Best for complex research tasks

Public API:
----------
- create_parallel_agent: Main unified factory function (RECOMMENDED)
- create_parallel_tool_agent: Legacy wrapper for backward compatibility
- create_parallel_direct_tool_agent: Legacy wrapper for backward compatibility
- create_parallel_react_agent: Legacy wrapper for backward compatibility

Example Usage:
-------------
```python
from agent_blocks_hub.parallel_agent import create_parallel_agent
from llm_tool_hub.scientific_research_tool import SearchSemanticScholar

# Direct mode (recommended for DeepSeek V3)
agent = create_parallel_agent(
    tools=[SearchSemanticScholar()],
    mode="direct",
    enable_summarization=False,
    verbose=True
)

result = agent.invoke({
    "parallel_tool_agent_messages": [
        "transformer neural networks",
        "attention mechanisms"
    ]
})

# Access results
for idx, result_data in result["agent_results"].items():
    print(f"Query {idx}: {result_data['result']}")
```

Migration Guide:
---------------
Old code:
    from agent_blocks_hub.parallel_tool_agent import create_parallel_tool_agent
    agent = create_parallel_tool_agent(llm=llm, tools=tools)

New code (recommended):
    from agent_blocks_hub.parallel_agent import create_parallel_agent
    agent = create_parallel_agent(llm=llm, tools=tools, mode="direct")
"""

from .factory import (
    create_parallel_agent,
    create_parallel_tool_agent,
    create_parallel_direct_tool_agent,
    create_parallel_react_agent,
)

__all__ = [
    "create_parallel_agent",
    "create_parallel_tool_agent",
    "create_parallel_direct_tool_agent", 
    "create_parallel_react_agent",
]
