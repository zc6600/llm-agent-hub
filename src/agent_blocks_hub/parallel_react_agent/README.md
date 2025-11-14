# Parallel React Agent

A LangGraph-based system for parallel multi-query research with intelligent result synthesis.

## Overview

The Parallel React Agent enables efficient parallel processing of multiple research queries:

1. **Parallel Execution**: Multiple ReAct agents process different queries simultaneously
2. **Unified Tools**: All agents share the same set of tools
3. **Consistent Prompting**: All agents follow the same system prompt guidelines
4. **Result Synthesis**: A Summarizing Agent integrates all results into a coherent, logical summary

## Architecture

```
Input Queries
     ↓
[Initialize] - Set up LLM, tools, prompts
     ↓
[Parallel React Agents] - Run multiple agents in parallel
├─ Agent 1: Query1 → (Act → Observe → Think)
├─ Agent 2: Query2 → (Act → Observe → Think)
└─ Agent N: QueryN → (Act → Observe → Think)
     ↓
[Summarizing Agent] - Analyze & synthesize all results
     ↓
Final Summary
```

## Quick Start

```python
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from agent_blocks_hub.parallel_react_agent import create_parallel_react_agent

# Initialize LLM and tools
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
tools = [DuckDuckGoSearchRun()]

# Create the agent
agent = create_parallel_react_agent(
    llm=llm,
    tools=tools,
    system_prompt="Focus on recent developments and evidence-based analysis"
)

# Run multiple queries in parallel
result = agent.invoke({
    "parallel_react_agent_messages": [
        "What are recent advances in AI safety?",
        "What are key benchmarks for evaluating AI?",
        "How are multimodal models being used?"
    ]
})

# Get the synthesized summary
print(result["final_summary"])
```

## Input/Output Format

### Input

Invoke the agent with a dictionary containing:

```python
{
    "parallel_react_agent_messages": [
        "Query 1",
        "Query 2",
        "Query N"
    ]
}
```

### Output

The agent returns a dictionary with:

```python
{
    "parallel_react_agent_messages": [...],  # Original queries
    "agent_results": {
        0: {
            "query_index": 0,
            "query": "Query 1",
            "result": "Agent 1's response...",
            "intermediate_steps": [...],  # ReAct loop steps
            "success": True,
            "error": None
        },
        ...
    },
    "final_summary": "Synthesized summary of all results..."
}
```

## Key Components

### State (`state.py`)

- `ParallelReactAgentState`: Main state container
- `AgentResult`: Individual agent result structure

### Prompts (`prompts.py`)

- `PARALLEL_REACT_AGENT_SYSTEM_PROMPT`: Built-in prompt for individual agents
- `SUMMARIZING_AGENT_SYSTEM_PROMPT`: Built-in prompt for summary agent
- `get_combined_system_prompt()`: Combines architecture + user prompts

### Nodes (`nodes.py`)

- `initialize_state()`: Set up configuration
- `run_parallel_agents()`: Execute agents in parallel
- `summarizing_agent()`: Generate comprehensive summary

### Agent (`agent.py`)

- `create_parallel_react_agent()`: Main API to create agents
- `get_compiled_graph()`: With optional graph visualization

## Features

### Parallel Execution
- All queries processed simultaneously using asyncio
- Fallback to sequential execution if asyncio unavailable
- Independent tool execution per agent

### Flexible Prompting
- Combine architecture system prompt with user prompts
- Customizable behavior per use case
- Clear separation of concerns

### Result Synthesis
- Summarizing Agent analyzes structure and logic
- Integrates multiple perspectives coherently
- Logical connections between results

### Visualization
- Generates PNG graph visualization (requires pygraphviz)
- Falls back to Mermaid format if needed
- Useful for debugging and documentation

## Configuration

### System Prompts

The agent uses combined system prompts:

```python
agent = create_parallel_react_agent(
    llm=llm,
    tools=tools,
    system_prompt="Additional domain-specific instructions"
)
```

The final prompt will be:
1. Built-in architecture prompt
2. User-provided system prompt (if given)

### Tools

Any tools compatible with LangChain can be used:

```python
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun

tools = [
    DuckDuckGoSearchRun(),
    ArxivQueryRun(),
]

agent = create_parallel_react_agent(llm=llm, tools=tools)
```

## Examples

See `example/parallel_react_agent/` for detailed examples:
- Basic usage with search tools
- Custom tools integration
- Multi-step research workflows
- Integration with other agents

## Performance Considerations

### Parallel Execution
- 3 queries with 5 parallel agents typically complete in ~1-2 minutes
- Time depends on LLM latency and tool response times
- Ideal for independent research tasks

### Memory
- Each agent maintains separate state
- Results collected after all agents complete
- Linear memory scaling with query count

### Tool Usage
- Tools are called independently by each agent
- No coordination or caching between agents
- Consider rate limits for external APIs

## Troubleshooting

### Graph Visualization Not Working
```python
from agent_blocks_hub.parallel_react_agent import get_compiled_graph

# This will fallback to Mermaid format if pygraphviz unavailable
agent = get_compiled_graph(llm, tools, save_graph_image=True)
```

### Asyncio Issues
The system automatically falls back to sequential execution if asyncio 
is not available or event loop issues occur.

### Tool Errors
Individual agent errors are caught and reported. Other agents continue
processing. Check `agent_results[i]["error"]` for error details.

## Architecture Decisions

1. **Parallel over Sequential**: Maximize throughput for independent queries
2. **Summarizing Agent**: Logical integration rather than simple concatenation
3. **State Composition**: Inherit from appropriate base to get desired functionality
4. **Modular Design**: Follow deep_diver pattern for maintainability

## Related Projects

- `deep_diver`: Single-query multi-step scientific research agent
- `paper_planner`: Literature review and research planning
- `llm_tool_hub`: Base utilities and tool infrastructure
