"""Basic tests for the Literature Review Agent.

These tests focus on validating that:
- The graph compiles and can be invoked with a minimal mocked LLM.
- The final state contains a ``final_review`` with a ``raw_text`` field.
- The ``all_references`` list is correctly passed into the final_review.
"""

from typing import Any, Dict, List

import pytest

from multi_agent_hub.scientific_research.literature_review import (
    LiteratureReviewState,
    create_literature_review_agent,
)


class DummyLLM:
    """Very small stand-in for a chat model for unit tests.

    It returns simple, deterministic content that still satisfies the
    JSON parsing expectations in the nodes.
    """

    def invoke(self, messages: List[Any]):  # type: ignore[override]
        """Return deterministic content based on the prompt text."""

        from types import SimpleNamespace

        content = messages[0].content

        # Stage 1: sub-topic decomposition
        if "SUB_TOPIC_DECOMPOSITION_PROMPT" in content or "Sub-topic" in content:
            text = '["Background", "Methods", "Challenges"]'

        # Stage 2: search query generation
        elif "SEARCH_QUERY_GENERATION_PROMPT" in content or "search queries from sub-topics" in content:
            text = '["llm scientific agents", "ai literature review tools"]'

        # Stage 4: paper selection
        elif "PAPER_SELECTION_PROMPT" in content or "Candidate papers" in content:
            import json

            try:
                json_start = content.find("[")
                json_end = content.rfind("]") + 1
                json_str = content[json_start:json_end]
                papers = json.loads(json_str)
                ids = [p.get("paper_id", "") for p in papers]
                text = json.dumps(ids)
            except Exception:
                text = "[]"

        # Stage 5: clustering
        elif "CLUSTERING_PROMPT" in content or "Selected papers" in content:
            text = (
                "[{\"cluster_id\": 1, \"name\": \"All papers\", "
                "\"description\": \"Dummy cluster\", \"paper_ids\": []}]"
            )

        # Stage 6 or anything else: final review text
        else:
            text = "# Dummy Literature Review\n\nThis is a test review.\n\n## References\n\n[1] Dummy paper."

        return SimpleNamespace(content=text)


class DummyTool:
    """Placeholder tool; not used in tests but required by API."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        return None


@pytest.fixture
def dummy_agent() -> Any:
    llm = DummyLLM()
    tools: List[DummyTool] = [DummyTool()]
    return create_literature_review_agent(llm=llm, tools=tools, verbose=False)


def test_literature_review_agent_basic_run(dummy_agent: Any) -> None:
    """The agent should run end-to-end with the dummy LLM."""

    init_state: Dict[str, Any] = {
        "literature_review_message": "LLM-based scientific agents",
        # Pre-populate gathered_papers so that selection has something
        "gathered_papers": [
            {
                "paper_id": "p1",
                "title": "Dummy Paper 1",
                "abstract": "Test abstract 1",
                "authors": ["Author A"],
                "year": 2024,
                "venue": "TestConf",
                "url": "http://example.com/p1",
                "citation_count": 10,
                "source": "semantic_scholar",
            },
            {
                "paper_id": "p2",
                "title": "Dummy Paper 2",
                "abstract": "Test abstract 2",
                "authors": ["Author B"],
                "year": 2023,
                "venue": "TestConf",
                "url": "http://example.com/p2",
                "citation_count": 5,
                "source": "semantic_scholar",
            },
        ],
    }

    result: LiteratureReviewState = dummy_agent(init_state)

    assert "final_review" in result
    final = result["final_review"]
    assert isinstance(final, dict)
    assert "raw_text" in final
    assert isinstance(final["raw_text"], str)

    # Ensure that references were propagated into the final_review
    assert "references" in final
    references = final["references"]
    assert isinstance(references, list)
    assert len(references) >= 1
