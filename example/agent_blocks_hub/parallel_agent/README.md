# Parallel Agent Examples

This directory contains examples for using the unified `parallel_agent` module with different execution strategies.

## 📁 File Structure

```
parallel_agent/
├── README.md                    # This file
├── 01_direct_mode.py           # Direct tool calling (RECOMMENDED for DeepSeek V3)
├── 02_tool_calling_mode.py     # LLM tool calling (for GPT-4, Claude)
├── 03_react_mode.py            # Full ReAct reasoning loops
├── 04_compare_modes.py         # Side-by-side comparison of all modes
└── 05_with_summarization.py   # Using LLM summarization
```

## 🎯 Quick Start

### Example 1: Direct Mode (Fastest, Recommended for DeepSeek V3)

```bash
python 01_direct_mode.py
```

**When to use:**
- Working with models that have poor tool calling support (e.g., DeepSeek V3)
- Simple information gathering tasks
- When you need maximum speed
- When tool.run() API is straightforward

### Example 2: Tool Calling Mode (For GPT-4, Claude)

```bash
python 02_tool_calling_mode.py
```

**When to use:**
- Working with GPT-4, Claude, or other models with good tool calling
- Need LLM to intelligently decide when/how to use tools
- Multiple tools available and LLM should choose

### Example 3: ReAct Mode (Most Comprehensive)

```bash
python 03_react_mode.py
```

**When to use:**
- Complex research tasks requiring multi-step reasoning
- Need full thought chains and observation-action cycles
- Quality is more important than speed

### Example 4: Compare All Modes

```bash
python 04_compare_modes.py
```

Runs all three modes side-by-side for comparison.

### Example 5: With Summarization

```bash
python 05_with_summarization.py
```

Shows how to use LLM summarization to synthesize results.

## 🔧 Configuration

All examples use `SearchSemanticScholar` tool. Make sure you have:

```python
from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
from langchain_openai import ChatOpenAI
from agent_blocks_hub.parallel_agent import create_parallel_agent
```

## 📊 Performance Comparison

| Mode | Speed | Accuracy | LLM Calls | Use Case |
|------|-------|----------|-----------|----------|
| **direct** | ⚡⚡⚡ Fast | ✓ Good | Minimal | Simple searches, DeepSeek V3 |
| **tool_calling** | ⚡⚡ Medium | ✓✓ Better | Moderate | GPT-4, multiple tools |
| **react** | ⚡ Slow | ✓✓✓ Best | Many | Complex reasoning tasks |

## 🚀 Migration from Old API

```python
# Old way (deprecated)
from agent_blocks_hub.parallel_tool_agent import create_parallel_tool_agent
agent = create_parallel_tool_agent(llm=llm, tools=tools)

# New way (recommended)
from agent_blocks_hub.parallel_agent import create_parallel_agent
agent = create_parallel_agent(llm=llm, tools=tools, mode="direct")
```

## 💡 Tips

1. **Start with `direct` mode** - It's the fastest and works with all models
2. **Use `tool_calling` only with GPT-4/Claude** - Other models have poor tool calling
3. **Reserve `react` for complex tasks** - It's slow but thorough
4. **Enable summarization selectively** - Only when you need synthesized results
5. **Test with small queries first** - Before scaling up
