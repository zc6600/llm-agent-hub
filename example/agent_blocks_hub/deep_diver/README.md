# Deep Diver Agent Examples

This directory contains examples demonstrating the Deep Diver agent, a LangGraph-based research agent that follows the scientific method.

## Overview

The Deep Diver agent implements a research workflow inspired by:
- **Karl Popper's Philosophy of Science**: Falsificationism and critical rationalism
- **Lean Startup Methodology**: Build-Measure-Learn cycle

## Workflow

The agent follows a five-step scientific method:

1. **Formulate Problem**: Decompose complex questions into testable sub-problems
2. **Gather Information**: Collect relevant data using available tools
3. **Generate Hypothesis**: Create falsifiable hypotheses (iterative)
4. **Verify Hypothesis**: Test and validate hypotheses, add to experience pool (iterative)
5. **Final Answer**: Synthesize verified knowledge into comprehensive answer

## Examples

### 01_basic_usage.py
Basic example showing how to:
- Initialize the Deep Diver agent
- Provide a simple research question
- Get structured results with confidence scores

### 02_custom_tools.py
Advanced example demonstrating:
- Custom tool integration
- Complex research questions
- Multi-iteration hypothesis testing

### 03_llm_controlled_iteration.py
Demonstrates intelligent iteration control:
- LLM decides when to continue vs. finish
- Quality-based stopping criteria
- Dynamic iteration behavior based on evidence quality

## Usage

```python
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from multi_agent_hub.deep_diver import create_deepdiver_agent

# Initialize
llm = ChatOpenAI(model="gpt-4")
agent = create_deepdiver_agent(
    llm=llm,
    tools=[DuckDuckGoSearchRun()],
    system_prompt="Your research instructions...",
    max_iterations=3
)

# Invoke
result = agent.invoke({
    "messages": [{"role": "user", "content": "Your question?"}]
})
```

## Key Features

- **Scientific Method**: Structured approach to research
- **Hypothesis Testing**: Iterative refinement of knowledge
- **Experience Pool**: Accumulates verified knowledge
- **Confidence Scoring**: Quantifies answer reliability
- **Tool Integration**: Flexible tool ecosystem

## Requirements

- langgraph
- langchain
- langchain-openai (or other LLM providers)
- langchain-community (for search tools)
