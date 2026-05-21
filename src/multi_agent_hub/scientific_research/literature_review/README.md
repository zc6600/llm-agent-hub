# Literature Review Agent

A systematic literature review agent that comprehensively analyzes academic papers on a research topic using a 5-stage process.

## Overview

The Literature Review Agent implements a structured approach to conducting literature reviews:

1. **Query Generation**: Generates focused search queries from the research topic
2. **Parallel Paper Search**: Executes searches across multiple queries in parallel
3. **Paper Analysis**: Analyzes individual papers for key findings, methodology, and limitations
4. **Synthesis**: Identifies themes, trends, and research gaps
5. **Final Report**: Generates a comprehensive literature review document

## Features

- ✅ **Parallel Processing**: Uses `parallel_tool_agent` or `parallel_react_agent` for efficient multi-query searches
- ✅ **Two Modes**: "lite" mode for fast execution, "full" mode for comprehensive analysis
- ✅ **Structured Output**: Organized report with themes, trends, gaps, and references
- ✅ **Flexible Tools**: Compatible with any LangChain-style search tool
- ✅ **Verbose Logging**: Optional detailed progress tracking

## Usage

### Basic Example

```python
from langchain_openai import ChatOpenAI
from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
from llm_tool_hub.integrations.langchain_adapter import LangchainToolAdapter
from multi_agent_hub.scientific_research.literature_review import create_literature_review_agent

# Initialize LLM and tools
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
search_tool = LangchainToolAdapter.to_langchain_structured_tool(
    SearchSemanticScholar()
)

# Create agent
agent = create_literature_review_agent(
    llm=llm,
    tools=[search_tool],
    system_prompt="Focus on peer-reviewed papers from top conferences",
    verbose=True,
    mode="lite"  # or "full" for more detailed analysis
)

# Run literature review
result = agent.invoke({
    "review_topic": "transformer architectures for natural language processing"
})

# Access results
print(result["final_review_report"]["comprehensive_report"])
print(f"Papers analyzed: {result['final_review_report']['total_papers']}")
print(f"Themes identified: {result['final_review_report']['themes_identified']}")
```

### With Custom System Prompt

```python
custom_prompt = """
You are conducting a literature review for a PhD dissertation.
Focus on:
- Papers published in the last 5 years
- Highly cited works (>50 citations)
- Both theoretical foundations and practical applications
- Identify methodological innovations
"""

agent = create_literature_review_agent(
    llm=llm,
    tools=[search_tool],
    system_prompt=custom_prompt,
    verbose=True
)
```

## Modes

### Lite Mode (Default)
- Uses `parallel_tool_agent` 
- Faster execution (~60-90 seconds)
- Suitable for quick literature surveys
- Best for cost-conscious workflows

### Full Mode
- Uses `parallel_react_agent`
- More comprehensive analysis (~120-180 seconds)
- Detailed reasoning traces
- Best for thorough academic reviews

```python
# Lite mode (default)
agent_lite = create_literature_review_agent(llm, tools, mode="lite")

# Full mode
agent_full = create_literature_review_agent(llm, tools, mode="full")
```

## Output Structure

The agent returns a state dictionary with:

```python
{
    "review_topic": str,              # Original research topic
    "search_queries": List[str],       # Generated search queries
    "search_results": List[dict],      # Papers from each query
    "analyzed_papers": List[dict],     # Analyzed paper information
    "thematic_clusters": List[dict],   # Grouped papers by theme
    "synthesis": str,                  # Overall synthesis
    "research_trends": List[str],      # Identified trends
    "research_gaps": List[str],        # Identified gaps
    "final_review_report": {
        "comprehensive_report": str,   # Full literature review text
        "topic": str,
        "total_papers": int,
        "themes_identified": int,
        "word_count": int
    }
}
```

## Stage-by-Stage Breakdown

### Stage 1: Query Generation
- Analyzes the research topic
- Generates 3-5 focused search queries
- Uses simple, broad keywords for optimal search results

### Stage 2: Parallel Paper Search
- Executes all search queries in parallel
- Uses available search tools (e.g., Semantic Scholar, arXiv)
- Collects papers with metadata

### Stage 3: Paper Analysis
- Analyzes each paper for:
  - Summary and key findings
  - Methodology
  - Limitations
  - Relevance score

### Stage 4: Synthesis
- Groups papers into thematic clusters
- Identifies research trends
- Discovers research gaps
- Creates cross-paper connections

### Stage 5: Final Report
- Generates comprehensive literature review
- Structures: Introduction → Thematic Analysis → Trends → Gaps → Conclusion
- Includes full reference list
- Targets 2500-4000 words

## Integration with Existing Tools

Compatible with any LangChain-style tool:

```python
from llm_tool_hub.scientific_research_tool import (
    SearchSemanticScholar,
    SearchArxiv,
    SearchUnpaywall
)
from llm_tool_hub.integrations.langchain_adapter import LangchainToolAdapter

# Use multiple search sources
tools = [
    LangchainToolAdapter.to_langchain_structured_tool(SearchSemanticScholar()),
    LangchainToolAdapter.to_langchain_structured_tool(SearchArxiv()),
]

agent = create_literature_review_agent(llm, tools)
```

## Comparison with Ideation Agent

| Feature | Literature Review | Ideation |
|---------|------------------|----------|
| Purpose | Survey existing work | Generate novel ideas |
| Stages | 5 | 6 |
| Focus | Comprehensive analysis | Creative thinking |
| Output | Literature review report | Research ideas with hypotheses |
| Use Case | Understanding state-of-art | Proposing new research |

## API Reference

### `create_literature_review_agent`

```python
def create_literature_review_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    mode: str = "lite",
) -> CompiledGraph
```

**Parameters:**
- `llm`: Language model for all stages
- `tools`: List of search tools
- `system_prompt`: Optional context/instructions
- `verbose`: Print progress information
- `mode`: "lite" or "full"

**Returns:** Compiled LangGraph agent

### `get_compiled_graph`

```python
def get_compiled_graph(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    save_graph_image: bool = True,
    image_name: str = "literature_review_agent_graph.png",
    mode: str = "lite",
) -> CompiledGraph
```

Same as `create_literature_review_agent` but with graph visualization options.

## Requirements

- `langgraph>=0.0.20`
- `langchain-core>=0.1.0`
- `langchain-openai` (or other LLM provider)
- Search tools (e.g., `llm_tool_hub.scientific_research_tool`)

## See Also

- [Ideation Agent](../ideation/README.md) - For generating novel research ideas
- [Deep Diver Agent](../../../agent_blocks_hub/deep_diver/README.md) - For in-depth analysis
- [Parallel Tool Agent](../../../agent_blocks_hub/parallel_tool_agent/README.md) - Building block used internally
