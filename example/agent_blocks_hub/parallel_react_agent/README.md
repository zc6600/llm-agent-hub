# Parallel React Agent Examples

This directory contains example scripts demonstrating the Parallel React Agent system.

## Examples

### 01_basic_usage.py

Basic example showing:
- Creating a Parallel React Agent with LLM and tools
- Processing multiple queries in parallel
- Receiving and printing results

**Run:**
```bash
python example/parallel_react_agent/01_basic_usage.py
```

**Expected Output:**
- Individual results from each parallel agent
- Synthesized summary combining all results
- Execution time and success metrics

---

### 02_ai_safety_research.py

**Use Case:** Multi-angle research on complex topics (Academic/Research)

Research AI safety from three independent angles in parallel:
- **Angle 1:** Technical safety measures (RLHF, constitutional AI, mechanistic interpretability)
- **Angle 2:** Governance and policy approaches (regulations, international frameworks)
- **Angle 3:** Frontier risks and challenges (jailbreaks, deceptive alignment)

**Demonstrates:**
- How to structure complex research questions
- Parallel processing of independent research angles
- Synthesis of diverse findings into coherent narrative

**Run:**
```bash
python example/parallel_react_agent/02_ai_safety_research.py
```

**Use This When:**
- Researching topics that have multiple dimensions
- Need comprehensive overview from different perspectives
- Want to avoid single-angle bias

---

### 03_competitive_analysis.py

**Use Case:** Competitive market analysis (Business Intelligence)

Analyze competitive landscape around LangChain:
- **Angle 1:** Competitor landscape and feature comparison
- **Angle 2:** Pricing and licensing strategies
- **Angle 3:** Competitive advantages and weaknesses

**Demonstrates:**
- Business intelligence applications
- Structured competitive analysis
- Parallel market research

**Run:**
```bash
python example/parallel_react_agent/03_competitive_analysis.py
```

**Use This When:**
- Evaluating market position and competitors
- Making product strategy decisions
- Benchmarking against alternatives

---

### 04_product_evaluation.py

**Use Case:** Technology evaluation and procurement (Decision Support)

Evaluate database technology (PostgreSQL vs MongoDB) across dimensions:
- **Dimension 1:** Technical capabilities and performance
- **Dimension 2:** User experience and ecosystem maturity
- **Dimension 3:** Cost and total cost of ownership

**Demonstrates:**
- Multi-dimensional product evaluation
- ROI and cost analysis
- Structured decision framework

**Run:**
```bash
python example/parallel_react_agent/04_product_evaluation.py
```

**Use This When:**
- Evaluating technology options
- Making procurement decisions
- Comparing products holistically

## Setup

Before running examples, ensure you have:

1. **Environment Variables Set**:
   ```bash
   export OPENAI_API_KEY=your_key_here
   # or for Anthropic:
   export ANTHROPIC_API_KEY=your_key_here
   ```

2. **Dependencies Installed**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Project Configured**:
   - The examples assume the project structure is set up correctly
   - `src/` directory is in Python path

## Common Issues

### Import Errors
If you get import errors, ensure you're running from the project root:
```bash
cd /path/to/llm-tool-hub
python example/parallel_react_agent/01_basic_usage.py
```

### LLM Configuration
Make sure your LLM provider is configured. The examples use `get_llm()` 
from `multi_agent_hub.llm_provider` which respects environment variables.

### Tool Issues
- **DuckDuckGo**: Should work without authentication
- **Custom Tools**: Implement LangChain's `BaseTool` interface

## Extending Examples

To create your own example:

1. Copy an existing example
2. Modify the queries and system prompt
3. Add tools as needed
4. Run and observe results

Example template:
```python
from agent_blocks_hub.parallel_react_agent import create_parallel_react_agent
from multi_agent_hub.llm_provider import get_llm

llm = get_llm()
tools = [...]  # Your tools

agent = create_parallel_react_agent(
    llm=llm,
    tools=tools,
    system_prompt="Your instructions"
)

result = agent.invoke({
    "parallel_react_agent_messages": ["Query 1", "Query 2", ...]
})

print(result["final_summary"])
```

## Advanced Examples (Coming Soon)

- Custom tools integration
- Multi-step research workflows
- Integration with other agent systems
- Performance benchmarking
