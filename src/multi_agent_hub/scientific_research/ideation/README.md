# Ideation Agent - 6-Stage Research Ideation System

## Overview

The **Ideation Agent** is a sophisticated research ideation system that generates novel research ideas through a structured 6-stage process. It leverages LLMs and parallel information gathering to create innovative, verifiable research hypotheses.

**Note**: As of November 2025, the `ideation_lite` variant has been merged into this main implementation. The agent now uses optimized 2–3 focused queries per stage for balanced efficiency and quality.

## Architecture

### 6-Stage Ideation Process

```
Stage 1: Initial Information Gathering (Parallel)
    ↓
Stage 2: Research Gap Identification
    ↓
Stage 3: Gap-driven Information Gathering (Parallel)
    ↓
Stage 4: Bit-Flipping Creative Idea Generation
    ↓
Stage 5: Sub-hypothesis Generation with Experiments
    ↓
Stage 6: Final Comprehensive Report
```

### Detailed Stage Descriptions

#### Stage 1: Initial Parallel Information Gathering

- **Input**: Research question (`ideation_message`)
- **Process**:
  - Generate 2–3 focused queries from the research question
  - Execute queries in parallel using `create_parallel_react_agent`
  - Gather comprehensive background information
- **Output**: List of gathered information covering different aspects

#### Stage 2: Research Gap Identification
- **Input**: Gathered information from Stage 1
- **Process**:
  - Analyze all gathered information
  - Identify what is well-understood vs. unexplored
  - Pinpoint the primary research gap
- **Output**: Clear description of the research gap

#### Stage 3: Gap-driven Parallel Information Gathering

- **Input**: Research gap from Stage 2
- **Process**:
  - Generate 2–3 queries specifically targeting the gap
  - Search for existing solutions and approaches to similar problems
  - Execute queries in parallel using `create_parallel_react_agent`
- **Output**: Information about existing approaches and methodologies

#### Stage 4: Bit-Flipping Creative Idea Generation
- **Input**: Research gap + existing approaches
- **Process**:
  - Apply creative thinking techniques (bit-flipping, domain transfer, assumption inversion)
  - Generate 3-5 novel ideas for addressing the gap
  - Provide reasoning for each creative leap
- **Output**: List of creative ideas with explanations

#### Stage 5: Sub-hypothesis Generation
- **Input**: Creative ideas from Stage 4
- **Process**:
  - Convert each creative idea into a testable research question
  - Design concrete verification experiments
  - Specify expected outcomes and measurable metrics
- **Output**: Verifiable hypotheses with experimental designs

#### Stage 6: Final Comprehensive Report
- **Input**: All previous stage outputs
- **Process**:
  - Synthesize all information into a structured report
  - Create actionable research proposal
  - Provide clear next steps and research direction
- **Output**: Complete ideation report with all components

## Key Features

✅ **Parallel Processing**: Stages 1 and 3 use parallel queries for efficiency
✅ **Creative Thinking**: Bit-flipping techniques generate novel ideas
✅ **Structured Output**: All ideas are converted to testable hypotheses
✅ **Integration with Parallel ReAct Agent**: Leverages existing parallel execution infrastructure
✅ **Configurable**: Support for custom system prompts and tools
✅ **Comprehensive**: Full ideation pipeline in a single agent

## Usage

### Basic Example

```python
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from multi_agent_hub.scientific_research.ideation import create_ideation_agent

# Initialize components
llm = ChatOpenAI(model="gpt-4")
tools = [DuckDuckGoSearchRun()]

# Create the agent
agent = create_ideation_agent(
    llm=llm,
    tools=tools,
    system_prompt="Focus on sustainable technologies"
)

# Run ideation
result = agent.invoke({
    "ideation_message": "How can we improve renewable energy storage?"
})

# Access results
report = result["final_idea_report"]
print(report["comprehensive_report"])
```

### Advanced Example with Custom System Prompt

```python
custom_prompt = """
You are a specialized researcher in quantum computing.
Focus on practical applications and commercial viability.
Consider both current limitations and future possibilities.
"""

agent = create_ideation_agent(
    llm=llm,
    tools=tools,
    system_prompt=custom_prompt
)

result = agent.invoke({
    "ideation_message": "What are novel approaches to quantum error correction?"
})
```

## Output Structure

The agent returns a state containing:

```python
{
    "ideation_message": str,              # Original question
    "research_gap": str,                  # Identified gap
    "gathered_information": List,         # Stage 1 outputs
    "gap_driven_information": List,       # Stage 3 outputs
    "creative_ideas": [                   # Stage 4 outputs
        {
            "idea": str,
            "reasoning": str,
            "novelty_score": float
        },
        ...
    ],
    "sub_hypotheses": [                   # Stage 5 outputs
        {
            "research_question": str,
            "verification_experiment": str,
            "expected_outcome": str,
            "testable_metrics": List[str]
        },
        ...
    ],
    "final_idea_report": {                # Stage 6 outputs
        "title": str,
        "original_question": str,
        "research_gap": str,
        "proposed_ideas": List,
        "hypotheses_and_experiments": List,
        "comprehensive_report": str
    }
}
```

## File Structure

```
src/multi_agent_hub/scientific_research/ideation/
├── __init__.py                 # Public API exports
├── agent.py                    # Main agent creation function
├── state.py                    # State type definitions
├── nodes.py                    # Node functions for each stage
├── prompts.py                  # System prompts for each stage
└── README.md                   # This file
```

## API Reference

### `create_ideation_agent(llm, tools, system_prompt=None, verbose=False, mode="lite")`

Creates and compiles an Ideation Agent.

**Parameters:**

- `llm` (BaseChatModel): Language model for all stages
- `tools` (List[BaseTool]): Tools available to parallel agents
- `system_prompt` (Optional[str]): Custom system prompt for additional context
- `verbose` (bool): Whether to print intermediate progress and debug information (default: False)
- `mode` (str): Parallel agent implementation mode (default: "lite")
  - `"lite"`: Uses `parallel_react_agent` for faster execution with summary-based results
  - `"full"`: Uses `parallel_tool_agent` for detailed per-query results with individual summarization

**Returns:**

- Compiled LangGraph agent ready to invoke

**Mode Comparison:**

| Aspect | Lite Mode (default) | Full Mode |
|--------|---------------------|-----------|
| **Parallel Agent** | `parallel_react_agent` | `parallel_tool_agent` |
| **Result Aggregation** | Single final summary | Individual query results + summarization |
| **Execution Time** | ~60 seconds | ~120 seconds |
| **Token Usage** | ~9k–12k tokens | ~15k–20k tokens |
| **Information Granularity** | Consolidated summary | Detailed per-query breakdown |
| **Best Use Case** | Quick iteration, cost-aware workflows | Comprehensive analysis, detailed traceability |

**Example - Lite Mode (Default):**

```python
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

llm = ChatOpenAI(model="gpt-4")
tools = [DuckDuckGoSearchRun()]

# Lite mode (faster, fewer tokens)
agent = create_ideation_agent(
    llm=llm,
    tools=tools,
    system_prompt="Focus on emerging technologies",
    verbose=True,
    mode="lite"  # default
)

result = agent.invoke({
    "ideation_message": "How can AI improve scientific research?"
})
```

**Example - Full Mode:**

```python
# Full mode (comprehensive, detailed results)
agent_full = create_ideation_agent(
    llm=llm,
    tools=tools,
    system_prompt="Focus on emerging technologies",
    verbose=True,
    mode="full"
)

result_full = agent_full.invoke({
    "ideation_message": "How can AI improve scientific research?"
})
```

### `get_compiled_graph(llm, tools, system_prompt=None, save_graph_image=True, image_name="ideation_agent_graph.png")`

Creates the agent and optionally saves a visualization.

**Parameters:**
- `llm` (BaseChatModel): Language model
- `tools` (List[BaseTool]): Available tools
- `system_prompt` (Optional[str]): Ideation context
- `save_graph_image` (bool): Whether to save graph visualization
- `image_name` (str): Path/name for saved visualization

**Returns:**
- Compiled LangGraph agent

## Examples

See `example/deep_diver/01_ideation_agent.py` for a complete working example.

Run with:
```bash
python example/deep_diver/01_ideation_agent.py
```

## Lite Mode: Lightweight Variant

The current implementation incorporates the **lightweight variant** (previously `ideation_lite`) as the default mode. This approach optimizes for **speed and efficiency** while maintaining quality.

### What is Lite Mode?

**Lite Mode** reduces computational overhead by:

1. **Fewer Queries per Stage**: 2–3 queries instead of 3–5
   - Stage 1: Generate 2–3 initial background queries instead of 3–5
   - Stage 3: Generate 2–3 gap-driven queries instead of 3–5
   - **Impact**: ~40% reduction in parallel execution time

2. **Direct Result Summary**: Uses final summary directly instead of individual query results
   - More efficient for downstream processing
   - Reduces token usage by consolidating information

3. **Balanced Information Gathering**: Optimized for practical use cases
   - Sufficient coverage for most research ideation tasks
   - Faster iteration cycles for hypothesis testing

### Performance Comparison

| Aspect | Full Mode (Previous) | Lite Mode (Current) | Improvement |
|--------|----------------------|---------------------|------------|
| Queries per stage | 3–5 | 2–3 | 40% fewer |
| Parallel execution time | ~120s | ~60s | 2x faster |
| Token usage (approx.) | 15k–20k | 9k–12k | 40% less |
| Information coverage | Very comprehensive | Comprehensive | Sufficient for most cases |
| Report quality | Excellent | Excellent | Comparable |

### When to Use Lite Mode

Lite Mode is ideal for:

- ✅ **Rapid prototyping** of research ideas
- ✅ **Cost-conscious workflows** with budget constraints
- ✅ **Iterative exploration** with quick feedback loops
- ✅ **Production systems** with latency requirements
- ✅ **Broader topic exploration** before deep dives

### When You Might Want More Queries

Consider the full mode (3–5 queries per stage) if:

- 🔍 **Deep comprehensive research** is required
- 📚 **Complex domains** needing extensive background
- 🎯 **High-stakes decisions** requiring thorough analysis
- 🔬 **Novelty-focused work** requiring exhaustive coverage

*Note*: To use the full mode variant (3–5 queries), you can manually edit `prompts.py` to change the query generation instructions, or we can add a `mode` parameter in future updates.

### Example: Using Lite Mode

```python
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from multi_agent_hub.scientific_research.ideation import create_ideation_agent

# Initialize
llm = ChatOpenAI(model="gpt-4")
tools = [DuckDuckGoSearchRun()]

# Create agent with lite mode (default)
agent = create_ideation_agent(
    llm=llm,
    tools=tools,
    system_prompt="Focus on AI applications in healthcare"
)

# Fast execution (~60–90 seconds total)
result = agent.invoke({
    "ideation_message": "How can LLMs improve clinical decision support?"
})

print(f"Gap identified: {result['research_gap']}")
print(f"Ideas generated: {len(result['creative_ideas'])}")
print(f"Report:\n{result['final_idea_report']['comprehensive_report']}")
```

### Quick Start with Installation

### Prerequisites

- Python 3.8+
- langchain, langchain-openai, langgraph
- OpenAI API key (for using ChatOpenAI)

### Setup

```bash
pip install langchain langchain-openai langgraph
export OPENAI_API_KEY="your-api-key-here"
```

## Migration from ideation_lite

If you were previously using `ideation_lite`:

- **Old import**: `from multi_agent_hub.scientific_research.ideation_lite import create_ideation_agent`
- **New import**: `from multi_agent_hub.scientific_research.ideation import create_ideation_agent`
- **Behavior**: Identical API and output format (ideation_lite has been merged into ideation)
- **Performance**: Query count is now 2–3 per stage (matching previous ideation_lite behavior for optimal efficiency)

**No code changes needed**—simply update the import path and your code will work as before.

## Design Decisions

### Parallel Query Execution
- **Why**: Two stages (1 and 3) use parallel queries to maximize information gathering efficiency
- **How**: Internally calls `create_parallel_react_agent` for parallel execution
- **Result**: Faster ideation with diverse information sources

### Direct LLM Calls for Middle Stages
- **Why**: Stages 2, 4, 5, and 6 are analytical/creative, not information-seeking
- **How**: Direct LLM invocation without tools for these pure analysis stages
- **Result**: Simpler flow, faster execution

### Complete State Preservation
- **Why**: All intermediate outputs are preserved in the state
- **How**: Each stage updates specific state fields
- **Result**: Full traceability and ability to inspect ideation reasoning

### Structured Output Requirements
- **Why**: Ideas must be verifiable and actionable
- **How**: Stage 5 enforces conversion of ideas to testable hypotheses
- **Result**: Research-ready outputs

### Query Count Optimization (2–3 per stage)
- **Why**: Balances comprehensiveness with efficiency
- **Previous ideation**: Used 3–5 queries (more thorough but slower)
- **Previous ideation_lite**: Used 2–3 queries (faster but lighter)
- **Current unified approach**: Uses 2–3 queries as optimal balance
- **Result**: Faster execution without significant quality loss

## Extending the Agent

To customize the Ideation Agent:

1. **Modify Prompts**: Edit `prompts.py` for different ideation strategies
2. **Add Stages**: Extend `agent.py` and `nodes.py` with additional stages
3. **Change Tools**: Provide different tools to `create_ideation_agent()`
4. **Custom Prompts**: Pass custom `system_prompt` parameter

## Performance Considerations

- **Information Gathering**: Parallel stages (1, 3) are the bottleneck - depends on tool performance
- **LLM Calls**: 6 direct LLM calls for analytical stages (2, 4, 5, 6)
- **Total Stages**: 6 sequential stages minimum
- **Timeout**: Depends on parallel agent timeouts + LLM response times

## Troubleshooting

### No ideas generated
- Check that `gathered_information` is properly populated
- Verify LLM is generating JSON responses correctly
- Check system prompt for overly restrictive constraints

### Parallel execution fails
- Ensure tools are properly initialized and working
- Check network connectivity for search tools
- Review error messages in stage 1.5 and 3.5 logs

### Poor quality ideas
- Improve the `ideation_message` with more context
- Adjust `system_prompt` to guide creative thinking
- Try different LLM models

## Integration with Other Agents

The Ideation Agent integrates seamlessly with:
- **Deep Diver**: For detailed research on generated ideas
- **Paper Planner**: For literature review on ideated topics
- **Parallel React Agent**: For general parallel information gathering

## Contributing

To improve the Ideation Agent:
1. Edit prompts in `prompts.py` for better quality outputs
2. Refactor node logic in `nodes.py` for efficiency
3. Extend state in `state.py` for new output types
4. Add examples in `example/` directory

## License

See LICENSE file in the project root.
