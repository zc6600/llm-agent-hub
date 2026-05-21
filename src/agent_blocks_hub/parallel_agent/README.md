# Parallel Agent

## Overview

`parallel_agent` provides a unified interface for parallel execution with three strategies:
- `direct`: calls tool `run()/invoke()` directly without LLM tool binding; fastest for simple information gathering.
- `tool_calling`: binds tools to the LLM; the LLM decides when/how to call tools; best for models with strong tool-calling.
- `react`: full ReAct reasoning loops with multi-step thought and tool interaction; best for complex research tasks.

Unified factory: `create_parallel_agent` (`src/agent_blocks_hub/parallel_agent/factory.py:17`).

## Prerequisites

- Requires `langgraph`, `langchain-core`, etc.
- Remarks or summarization require a configured LLM and API key (`src/llm_provider.py:64`).

## Quick Start

```python
from agent_blocks_hub.parallel_agent import create_parallel_agent
from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
from llm_provider import get_llm

llm = get_llm()
tool = SearchSemanticScholar()

agent = create_parallel_agent(
    llm=llm,
    tools=[tool],
    mode="direct",
    enable_remark=True,
    enable_summarization=True,
)

result = agent.invoke({
    "parallel_agent_message": [
        "transformer neural networks",
        "attention mechanism deep learning",
    ]
})

print(result["tool_result"])
print(result["remark"])
print(result["tool_result_with_remark"])
print(result.get("summary", ""))
```

## Input Keys

- `direct` / `tool_calling`: `parallel_agent_message` (list of queries)
- `react`: `parallel_react_agent_messages` (list)

Reader references:
- direct: `src/agent_blocks_hub/parallel_agent/parallel_direct_tool_agent/nodes.py:86`
- tool_calling: `src/agent_blocks_hub/parallel_agent/parallel_tool_agent/nodes.py:85`
- react: `src/agent_blocks_hub/parallel_agent/parallel_react_agent/nodes.py:76`

## Unified Return Schema

`invoke(...)` returns a dict that includes:
- `agent_results` / `tool_results`: original per-query dict (kept for compatibility).
- `tool_result`: `{idx: str}` main text per query.
- `remark`: `{idx: Optional[str]}` short per-query annotation (requires `enable_remark=True` and an LLM).
- `tool_result_with_remark`: `{idx: str}` concatenation of `tool_result[idx]` and `remark[idx]` if present.
- `summary`: `str` global summary (alias of `final_summary`; requires `enable_summarization=True`).

Unified wrapper implementation: `src/agent_blocks_hub/parallel_agent/factory.py:214`.

## Parameters

- Common:
  - `llm`: language model (required for remarks/summarization).
  - `tools`: list of tools.
  - `system_prompt`: optional; for summarization or react system prompts.
  - `verbose`: enable detailed logging.
  - `mode`: `"direct" | "tool_calling" | "react"`.
  - `enable_summarization`: enable global summary (`summary`).
  - `enable_remark`: enable per-query remark (`remark`).
  - `remark_prompt`: custom prompt for remarks.
  - `summarization_prompt`: custom prompt for summarization.
  - `tool_name` (direct): specific tool name to use (default first).

Factory passthrough:
- direct: `src/agent_blocks_hub/parallel_agent/factory.py:160-173`
- tool_calling: `src/agent_blocks_hub/parallel_agent/factory.py:177-186`
- react: `src/agent_blocks_hub/parallel_agent/factory.py:190-197`

## Mode Differences and Outputs

- `direct`:
  - `result` is the raw tool output; `remark` provides a short interpretation.
  - `summary` provides global synthesis if enabled.
  - Result construction: `src/agent_blocks_hub/parallel_agent/parallel_direct_tool_agent/nodes.py:219`.

- `tool_calling`:
  - `result` is LLM answer combined with tool execution overview.
  - `remark` and `summary` as above.
  - Result construction: `src/agent_blocks_hub/parallel_agent/parallel_tool_agent/nodes.py:241`.

- `react`:
  - `result` is the final ReAct answer per query; supports `remark` and global `summary`.
  - Result construction: `src/agent_blocks_hub/parallel_agent/parallel_react_agent/nodes.py:230`.

## Examples

### 1. Direct mode (no summarization)
```python
agent = create_parallel_agent(
    tools=[SearchSemanticScholar()],
    mode="direct",
    enable_summarization=False,
)
res = agent.invoke({"parallel_agent_message": ["BERT", "Transformer"]})
print(res["tool_result"])
```

### 2. Direct mode + remarks
```python
llm = get_llm()
agent = create_parallel_agent(
    llm=llm,
    tools=[SearchSemanticScholar()],
    mode="direct",
    enable_remark=True,
)
res = agent.invoke({"parallel_agent_message": ["BERT"]})
print(res["remark"]) 
print(res["tool_result_with_remark"]) 
```

### 3. Direct mode + summarization
```python
llm = get_llm()
agent = create_parallel_agent(
    llm=llm,
    tools=[SearchSemanticScholar()],
    mode="direct",
    enable_summarization=True,
)
res = agent.invoke({"parallel_agent_message": ["BERT", "Attention"]})
print(res.get("summary", ""))
```

### 4. Tool calling mode
```python
llm = get_llm(model="gpt-4")
agent = create_parallel_agent(
    llm=llm,
    tools=[SearchSemanticScholar()],
    mode="tool_calling",
    enable_remark=True,
)
res = agent.invoke({"parallel_agent_message": ["Transformer"]})
print(res["tool_result"]) 
print(res["remark"])      
```

### 5. React mode
```python
llm = get_llm(model="gpt-4")
agent = create_parallel_agent(
    llm=llm,
    tools=[SearchSemanticScholar()],
    mode="react",
    enable_summarization=True,
)
res = agent.invoke({"parallel_react_agent_messages": ["What is LangGraph?"]})
print(res["tool_result"]) 
print(res.get("summary", ""))
```

## Compatibility and Migration

- Compatibility: `agent_results` and `tool_results` are retained; the four unified keys are added by the wrapper (`factory.py:214`).
- Legacy factories (`create_parallel_tool_agent`, `create_parallel_direct_tool_agent`, `create_parallel_react_agent`) remain available (`src/agent_blocks_hub/parallel_agent/__init__.py:62`), but `create_parallel_agent` is recommended.

## FAQ

- How to enable remarks?
  - Set `enable_remark=True` and provide a valid `llm`; direct mode validation is in `src/agent_blocks_hub/parallel_agent/parallel_direct_tool_agent/nodes.py:90-97`.

- Why does a single `result` look like concatenation?
  - In direct mode, `result` is raw tool output; use `remark` for short interpretation or `summary` for global synthesis.

- How to customize styles?
  - Use `remark_prompt` and `summarization_prompt` (passed through in all modes).