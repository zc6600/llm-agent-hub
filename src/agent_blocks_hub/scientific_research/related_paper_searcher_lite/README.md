# Related Paper Searcher Litetools: Sematic Scholar



A LangGraph-based agent that searches for academic papers and uses an LLM to evaluate their relevance to a research query.Motivation: sematic scholar can generate unrelated result, which largely affect LLM, so we need a checker to choose and left comment for each result.



## Motivation1. query -> sematic scholar

2. LLM to judge with comment whether related: "High related and important" / "High related" "Medium related" "Not related"

Semantic Scholar can generate unrelated results, which negatively affects LLM performance. This tool filters search results by using an LLM to evaluate paper relevance before returning results.3. return only related paper



## Workflow

```python



1. **Search Papers** → Query Semantic Scholar to get candidate papers# default Sematic Scholar tool from llm_tool

2. **Evaluate Relevance** → Use LLM to classify each paper into:# you should check example to learn how to import semantic scholar tool from llm_tool_hub

   - "High related and important" - Directly addresses the research queryagent = create_related_paper_searcher_lite_agent(llm=llm)

   - "High related" - Closely related but not central

   - "Medium related" - Some relevanceresult = agent.invoke({"related_paper_searcher_lite_message": "query"})

   - "Not related" - Minimal or no relevance

3. **Filter Results** → Return only papers classified as related (exclude "Not related")

```

## Usage

```python
from langchain_openai import ChatOpenAI
from agent_blocks_hub.scientific_research.related_paper_searcher_lite import (
    create_related_paper_searcher_lite_agent
)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create agent
agent = create_related_paper_searcher_lite_agent(llm=llm)

# Run the agent
result = agent.invoke({
    "query": "transformer neural networks attention mechanism",
    "messages": []
})

# Access results
print(f"Total papers evaluated: {len(result['evaluation_results'])}")
print(f"Related papers: {len(result['related_papers'])}")

# Display related papers
for paper in result["related_papers"]:
    print(f"Title: {paper['paper_title']}")
    print(f"Relevance: {paper['relevance']}")
    print(f"Comment: {paper['comment']}")
```

## Input State

- `query` (str): The research query to search for papers
- `messages` (list): Conversation message history

## Output State

- `search_results` (str): Raw search results from Semantic Scholar
- `evaluation_results` (list): All evaluated papers with relevance scores
- `related_papers` (list): Filtered papers (excluding "Not related")

Each evaluation result contains:
- `paper_title`: Title of the paper
- `relevance`: Classification (High related and important | High related | Medium related | Not related)
- `comment`: Explanation of the classification

## Dependencies

- `langchain_core`: For LLM interfaces
- `langgraph`: For graph-based workflow
- `llm_tool_hub`: For Semantic Scholar search tool
- `semanticscholar`: For academic paper search

## Tools Used

This agent uses the `SearchSemanticScholar` tool from `llm_tool_hub` to search for academic papers.
