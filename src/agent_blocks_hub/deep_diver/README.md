# Deep Diver Agent

A scientific method-based research agent built with LangGraph.

## Overview

The Deep Diver agent follows a rigorous research methodology inspired by:

- **Karl Popper's Philosophy of Science**: Falsificationism and critical rationalism
- **Lean Startup Methodology**: Build-Measure-Learn iterative cycles

## Installation

```bash
# Install the package with dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

## Quick Start

```python
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from multi_agent_hub.deep_diver import create_deepdiver_agent

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Initialize tools
search_tool = DuckDuckGoSearchRun()

# Create agent
agent = create_deepdiver_agent(
    llm=llm,
    tools=[search_tool],
    system_prompt="You are a thorough research agent.",
    max_iterations=3
)

# Ask a question
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is LangGraph?"}]
})

# Access results
print(result["final_answer"])
print(f"Confidence: {result['confidence_score']:.2f}")
```

## Workflow

The agent follows these steps:

### 1. Formulate Problem

Decomposes complex questions into testable sub-problems.

### 2. Gather Information

Uses provided tools to collect relevant data.

### 3. Generate Hypothesis (Iterative)

Creates falsifiable hypotheses based on gathered information.

### 4. Verify Hypothesis (Iterative)

Tests hypotheses using tools and evidence, adds verified knowledge to experience pool.

### 5. Final Answer

Synthesizes all verified hypotheses into a comprehensive answer.

## Key Features

- **Scientific Rigor**: Follows falsificationism principles
- **Iterative Learning**: Learns from both successes and failures
- **Experience Pool**: Accumulates verified knowledge across iterations
- **Confidence Scoring**: Quantifies answer reliability
- **Tool Integration**: Works with any LangChain tools
- **Transparent Process**: Full visibility into reasoning steps

## Configuration

### Parameters

- `llm`: Language model (any LangChain BaseChatModel)
- `tools`: List of tools for information gathering (LangChain BaseTool)
- `system_prompt`: Custom instructions for the agent
- `max_iterations`: Maximum hypothesis-verification cycles (default: 3)

### Output Structure

```python
{
    "original_question": str,
    "decomposed_problems": List[str],
    "gathered_information": List[Dict],
    "hypotheses": List[Hypothesis],
    "experience_pool": List[Dict],
    "final_answer": str,
    "messages": List[Message]
}
```

## Examples

See the `example/deep_diver/` directory for detailed examples:

- `01_basic_usage.py`: Simple research question
- `02_custom_tools.py`: Using custom tools for specific domains

## Architecture

Built on LangGraph's StateGraph, the agent maintains state through a workflow graph with conditional edges for iteration control.

### Node Functions

- `formulate_problem`: Problem decomposition
- `gather_information`: Tool-based information collection
- `generate_hypothesis`: Hypothesis generation
- `verify_hypothesis`: Hypothesis testing
- `final_answer`: Answer synthesis

### State Management

Uses `DeepDiverState` (extends MessagesState) to track:

- Research progress
- Hypotheses and verification status
- Experience pool
- Tool results
- Confidence metrics

## Testing

```bash
# Run all tests
pytest tests/deep_diver/

# Quick test
python tests/deep_diver/test_quick.py
```

## License

MIT License - see LICENSE file for details.
