from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

from agent_blocks_hub.parallel_agent import create_parallel_agent


class PaperReviewState(TypedDict, total=False):
    paper_review_message: str
    llm: Any
    tools: List[BaseTool]
    system_prompt: str
    verbose: bool
    mode: str
    paper_search_result: str
    final_paper_review_report: Dict[str, Any]


PAPER_REVIEW_PROMPT = (
    "You are an expert peer reviewer. Review the paper based on the provided information. "
    "Return a COMPLETE JSON object with keys: "
    "title, identifiers, summary, novelty, strengths, weaknesses, methodology_soundness, "
    "reproducibility, significance, ethical_considerations, overall_rating, decision, "
    "comments_for_authors, comprehensive_report. "
    "Use a 0-10 scale for overall_rating. Decision must be one of: "
    "'Accept', 'Weak Accept', 'Borderline', 'Weak Reject', 'Reject'. "
    "Fill identifiers with available fields like doi, url, year, venue, authors."
)


def _initialize_state(state: PaperReviewState) -> Dict[str, Any]:
    verbose = state.get("verbose", False)
    if verbose:
        print("\n[INITIALIZE] Setting up Paper Review Agent")
    return {
        "paper_search_result": "",
        "final_paper_review_report": {},
    }


def _gather_paper_info(state: PaperReviewState) -> Dict[str, Any]:
    query = state.get("paper_review_message", "").strip()
    tools = state.get("tools", [])
    llm = state.get("llm")
    verbose = state.get("verbose", False)
    if not query:
        return {"paper_search_result": ""}
    if not tools:
        return {"paper_search_result": query}

    agent = create_parallel_agent(
        llm=llm,
        tools=tools,
        mode="direct",
        enable_summarization=False,
        enable_remark=True,
        remark_prompt=REMARK_PROMPT,
        verbose=verbose,
    )

    result = agent.invoke({
        "parallel_agent_message": [query],
        "verbose": verbose,
    })

    remarks = result.get("remark", {})
    text = remarks.get(0, "") or ""
    if not text:
        tool_result = result.get("tool_result", {})
        text = tool_result.get(0, "") or ""
    return {"paper_search_result": text or query}


def _generate_review(state: PaperReviewState) -> Dict[str, Any]:
    llm = state.get("llm")
    verbose = state.get("verbose", False)
    paper_text = state.get("paper_search_result", "")
    system_prompt = state.get("system_prompt", "")
    if llm is None:
        raise ValueError("LLM not configured")
    if verbose:
        print("\n[REVIEW] Generating paper review")

    prompt = f"{PAPER_REVIEW_PROMPT}\n\nPaper information:\n{paper_text}\n\n{system_prompt}"
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content

    review_data: Dict[str, Any] = {}
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            import json
            review_data = json.loads(content[start:end])
        else:
            review_data = {
                "title": "",
                "identifiers": {},
                "summary": content,
                "novelty": "",
                "strengths": [],
                "weaknesses": [],
                "methodology_soundness": "",
                "reproducibility": "",
                "significance": "",
                "ethical_considerations": "",
                "overall_rating": 0.0,
                "decision": "Borderline",
                "comments_for_authors": "",
                "comprehensive_report": content,
            }
    except Exception:
        review_data = {
            "title": "",
            "identifiers": {},
            "summary": content,
            "novelty": "",
            "strengths": [],
            "weaknesses": [],
            "methodology_soundness": "",
            "reproducibility": "",
            "significance": "",
            "ethical_considerations": "",
            "overall_rating": 0.0,
            "decision": "Borderline",
            "comments_for_authors": "",
            "comprehensive_report": content,
        }

    return {"final_paper_review_report": review_data}


def create_paper_review_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    mode: str = "lite",
) -> Any:
    if system_prompt is None:
        system_prompt = ""

    workflow = StateGraph(PaperReviewState)

    def init_node(state: PaperReviewState) -> Dict[str, Any]:
        return {
            "llm": llm,
            "tools": tools,
            "system_prompt": system_prompt,
            "verbose": verbose,
            "mode": mode,
            **_initialize_state(state),
        }

    def gather_node(state: PaperReviewState) -> Dict[str, Any]:
        return _gather_paper_info(state)

    def review_node(state: PaperReviewState) -> Dict[str, Any]:
        return _generate_review(state)

    workflow.add_node("initialize", init_node)
    workflow.add_node("gather", gather_node)
    workflow.add_node("review", review_node)

    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "gather")
    workflow.add_edge("gather", "review")
    workflow.add_edge("review", END)

    return workflow.compile()


def paper_review(
    llm: BaseChatModel,
    tools: List[BaseTool],
    paper_query: str,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    mode: str = "lite",
) -> Dict[str, Any]:
    agent = create_paper_review_agent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        verbose=verbose,
        mode=mode,
    )
    return agent.invoke({
        "paper_review_message": paper_query,
        "verbose": verbose,
    })
REMARK_PROMPT = (
    "You are a research synthesis expert. Analyze and synthesize the tool execution results for a single query in Markdown.\n\n"
    "Query: {query}\n"
    "Result:\n{result}\n\n"
    "Repeat central results, evaluate relevance/quality, and synthesize into bullet points with citations when available."
)