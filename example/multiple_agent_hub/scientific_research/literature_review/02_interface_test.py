import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multi_agent_hub.scientific_research.literature_review import create_literature_review_agent


class DummyLLM:
    def invoke(self, messages):
        from types import SimpleNamespace
        content = messages[0].content
        if "Generate simple, keyword-focused queries" in content:
            text = '["transformer attention", "bert model", "nlp deep learning"]'
        elif "Format as JSON:" in content and '"paper_id"' in content:
            text = '{"paper_id":"p1","title":"Dummy","summary":"s","key_findings":["f"],"methodology":"m","limitations":"l","relevance_score":0.9}'
        elif "Format as JSON array:" in content:
            text = '[{"theme":"Dummy Theme","papers":["p1"],"key_insights":["i"],"research_gaps":["g"]}]'
        elif "Format as JSON:" in content and '"synthesis"' in content:
            text = '{"synthesis":"s","research_trends":["t"],"research_gaps":["g"]}'
        elif "Generate the comprehensive literature review report" in content:
            text = "Comprehensive literature review"
        else:
            text = "OK"
        return SimpleNamespace(content=text)


class MockSearchTool:
    name = "mock_search"

    def invoke(self, args):
        q = args.get("query", "")
        return f"Papers for: {q}"


def main():
    llm = DummyLLM()
    tools = [MockSearchTool()]
    agent = create_literature_review_agent(
        llm=llm,
        tools=tools,
        system_prompt="",
        verbose=False,
        mode="lite",
    )
    result = agent.invoke({"review_topic": "test topic"})
    search_results = result.get("search_results", [])
    print(f"search_results_count={len(search_results)}")
    ok = all(r.get("success", True) is not False for r in search_results)
    print(f"search_results_success={ok}")
    print(f"final_review_report_present={'final_review_report' in result}")


if __name__ == "__main__":
    main()