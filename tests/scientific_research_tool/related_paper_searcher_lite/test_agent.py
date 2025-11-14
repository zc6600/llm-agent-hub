"""
Tests for Related Paper Searcher Lite Agent
"""

import pytest
from unittest.mock import Mock, MagicMock
from agent_blocks_hub.scientific_research.related_paper_searcher_lite.nodes import (
    search_papers_node, evaluate_papers_node, _parse_evaluation_results
)
from agent_blocks_hub.scientific_research.related_paper_searcher_lite.state import (
    RelatedPaperSearcherLiteState
)


def test_parse_evaluation_results():
    """Test parsing of evaluation results from LLM output"""
    
    evaluation_text = """Paper: Transformer Architecture for NLP
Relevance: High related and important
Comment: This paper introduces the transformer architecture which is fundamental to attention mechanisms.

---

Paper: BERT: Pre-training of Deep Bidirectional Transformers
Relevance: High related
Comment: Uses transformers for pre-training language models.

---

Paper: ResNet: Deep Residual Learning
Relevance: Not related
Comment: Computer vision architecture, not related to transformer attention mechanisms."""
    
    results = _parse_evaluation_results(evaluation_text)
    
    assert len(results) == 3
    assert results[0]["paper_title"] == "Transformer Architecture for NLP"
    assert results[0]["relevance"] == "High related and important"
    assert results[1]["relevance"] == "High related"
    assert results[2]["relevance"] == "Not related"


def test_search_papers_node():
    """Test search papers node"""
    
    # Mock search tool
    mock_tool = Mock()
    mock_tool.run.return_value = "Paper 1\n\nPaper 2\n\nPaper 3"
    
    # Create state
    state = {
        "query": "transformer attention",
        "messages": [],
        "search_results": None,
        "evaluation_results": [],
        "related_papers": []
    }
    
    # Run node
    result = search_papers_node(state, mock_tool)
    
    assert result["search_results"] == "Paper 1\n\nPaper 2\n\nPaper 3"
    assert mock_tool.run.called
    mock_tool.run.assert_called_with(query="transformer attention", limit=10)


def test_evaluate_papers_node():
    """Test evaluate papers node"""
    
    # Mock LLM
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = """Paper: Paper Title 1
Relevance: High related
Comment: Very relevant paper

---

Paper: Paper Title 2
Relevance: Not related
Comment: Not relevant"""
    
    mock_llm.invoke.return_value = mock_response
    
    # Create state
    state = {
        "query": "test query",
        "messages": [],
        "search_results": "Paper: Paper Title 1\n\nPaper: Paper Title 2",
        "evaluation_results": [],
        "related_papers": []
    }
    
    # Run node
    result = evaluate_papers_node(state, mock_llm)
    
    # Should have 1 related paper (excluding "Not related")
    assert len(result["related_papers"]) == 1
    assert len(result["evaluation_results"]) == 2
    assert result["related_papers"][0]["relevance"] == "High related"


def test_evaluate_papers_empty_results():
    """Test evaluate papers node with empty search results"""
    
    mock_llm = Mock()
    
    state = {
        "query": "test query",
        "messages": [],
        "search_results": "",
        "evaluation_results": [],
        "related_papers": []
    }
    
    result = evaluate_papers_node(state, mock_llm)
    
    assert result["evaluation_results"] == []
    assert result["related_papers"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
