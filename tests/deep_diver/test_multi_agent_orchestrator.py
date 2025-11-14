"""
Unit tests for the MultiAgentOrchestrator.

Tests the multi-agent problem decomposition and parallel execution.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent_blocks_hub.deep_diver.multi_agent_orchestrator import (
    MultiAgentOrchestrator,
    SubTaskAgent
)


class TestSubTaskAgent(unittest.TestCase):
    """Test SubTaskAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.mock_agent_graph = Mock()
        self.task_id = 1
        self.task_description = "What is LangGraph?"
    
    def test_initialization(self):
        """Test SubTaskAgent initialization."""
        agent = SubTaskAgent(
            task_id=self.task_id,
            task_description=self.task_description,
            agent_graph=self.mock_agent_graph,
            llm=self.mock_llm,
            system_prompt="Test prompt"
        )
        
        self.assertEqual(agent.task_id, self.task_id)
        self.assertEqual(agent.task_description, self.task_description)
        self.assertIsNone(agent.result)
        self.assertIsNone(agent.error)
    
    def test_execute_success(self):
        """Test successful execution of a sub-task agent."""
        # Mock the agent graph invoke method
        self.mock_agent_graph.invoke.return_value = {
            "gathered_information": [
                {"tool": "search", "result": "info1"}
            ],
            "final_answer": "Test answer",
            "hypotheses": ["hyp1"],
            "decomposed_problems": ["prob1"]
        }
        
        agent = SubTaskAgent(
            task_id=1,
            task_description="Test task",
            agent_graph=self.mock_agent_graph,
            llm=self.mock_llm
        )
        
        result = agent.execute()
        
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["task_id"], 1)
        self.assertEqual(len(result["gathered_information"]), 1)
        self.assertEqual(result["final_answer"], "Test answer")
    
    def test_execute_failure(self):
        """Test execution failure handling."""
        # Mock the agent graph to raise an exception
        self.mock_agent_graph.invoke.side_effect = Exception("Test error")
        
        agent = SubTaskAgent(
            task_id=1,
            task_description="Test task",
            agent_graph=self.mock_agent_graph,
            llm=self.mock_llm
        )
        
        result = agent.execute()
        
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["task_id"], 1)
        self.assertIn("error", result)
        self.assertEqual(len(result["gathered_information"]), 0)


class TestMultiAgentOrchestrator(unittest.TestCase):
    """Test MultiAgentOrchestrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.mock_tool = Mock()
        self.mock_tool.name = "search"
        
        self.orchestrator = MultiAgentOrchestrator(
            llm=self.mock_llm,
            tools=[self.mock_tool],
            system_prompt="Test prompt",
            max_workers=2,
            max_iterations=2
        )
    
    def test_initialization(self):
        """Test MultiAgentOrchestrator initialization."""
        self.assertEqual(self.orchestrator.llm, self.mock_llm)
        self.assertEqual(len(self.orchestrator.tools), 1)
        self.assertEqual(self.orchestrator.max_workers, 2)
        self.assertEqual(self.orchestrator.max_iterations, 2)
    
    def test_parse_decomposed_problems_json(self):
        """Test parsing JSON-formatted decomposed problems."""
        json_response = '{"sub_problems": [{"problem": "Problem 1"}, {"problem": "Problem 2"}]}'
        
        result = self.orchestrator._parse_decomposed_problems(json_response)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Problem 1")
        self.assertEqual(result[1], "Problem 2")
    
    def test_parse_decomposed_problems_numbered_list(self):
        """Test parsing numbered list format."""
        list_response = "1. Problem 1\n2. Problem 2\n3. Problem 3"
        
        result = self.orchestrator._parse_decomposed_problems(list_response)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "Problem 1")
        self.assertEqual(result[1], "Problem 2")
        self.assertEqual(result[2], "Problem 3")
    
    def test_parse_decomposed_problems_fallback(self):
        """Test fallback parsing when format is unclear."""
        text_response = "This is line 1 with enough text\nThis is line 2 with enough text"
        
        result = self.orchestrator._parse_decomposed_problems(text_response)
        
        self.assertGreater(len(result), 0)
    
    @patch('multi_agent_hub.deep_diver.multi_agent_orchestrator.create_deepdiver_agent')
    def test_create_sub_task_agents(self, mock_create_agent):
        """Test creation of sub-task agents."""
        mock_create_agent.return_value = Mock()
        
        sub_tasks = ["Task 1", "Task 2", "Task 3"]
        agents = self.orchestrator.create_sub_task_agents(sub_tasks)
        
        self.assertEqual(len(agents), 3)
        self.assertEqual(agents[0].task_id, 1)
        self.assertEqual(agents[1].task_id, 2)
        self.assertEqual(agents[2].task_id, 3)
        self.assertEqual(mock_create_agent.call_count, 3)
    
    def test_aggregate_results_all_success(self):
        """Test aggregation when all sub-tasks succeed."""
        sub_results = [
            {
                "task_id": 1,
                "task_description": "Task 1",
                "status": "completed",
                "gathered_information": [{"tool": "search", "result": "info1"}],
                "final_answer": "Answer 1",
                "hypotheses": ["hyp1"]
            },
            {
                "task_id": 2,
                "task_description": "Task 2",
                "status": "completed",
                "gathered_information": [{"tool": "search", "result": "info2"}],
                "final_answer": "Answer 2",
                "hypotheses": ["hyp2"]
            }
        ]
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Final synthesized answer"
        self.mock_llm.invoke.return_value = mock_response
        
        result = self.orchestrator.aggregate_results("Test question", sub_results)
        
        self.assertEqual(result["num_sub_tasks"], 2)
        self.assertEqual(result["total_gathered_sources"], 2)
        self.assertEqual(result["total_hypotheses"], 2)
        self.assertEqual(result["final_answer"], "Final synthesized answer")
    
    def test_aggregate_results_with_failures(self):
        """Test aggregation when some sub-tasks fail."""
        sub_results = [
            {
                "task_id": 1,
                "task_description": "Task 1",
                "status": "completed",
                "gathered_information": [{"tool": "search", "result": "info1"}],
                "final_answer": "Answer 1",
                "hypotheses": ["hyp1"]
            },
            {
                "task_id": 2,
                "task_description": "Task 2",
                "status": "failed",
                "error": "Test error",
                "gathered_information": [],
                "final_answer": ""
            }
        ]
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Partial answer"
        self.mock_llm.invoke.return_value = mock_response
        
        result = self.orchestrator.aggregate_results("Test question", sub_results)
        
        # Should still have 2 sub-tasks
        self.assertEqual(result["num_sub_tasks"], 2)
        # But only 1 successful source
        self.assertEqual(result["total_gathered_sources"], 1)
        # And 1 hypothesis
        self.assertEqual(result["total_hypotheses"], 1)


class TestMultiAgentWorkflow(unittest.TestCase):
    """Test the complete multi-agent workflow."""
    
    @patch('multi_agent_hub.deep_diver.multi_agent_orchestrator.create_deepdiver_agent')
    def test_execute_workflow(self, mock_create_agent):
        """Test the complete execute workflow."""
        # Set up mocks
        mock_llm = Mock()
        mock_tool = Mock()
        mock_tool.name = "search"
        
        # Mock the decomposition
        decompose_response = Mock()
        decompose_response.content = "1. Problem 1\n2. Problem 2"
        
        # Mock the aggregation
        aggregate_response = Mock()
        aggregate_response.content = "Final answer"
        
        mock_llm.invoke.side_effect = [
            decompose_response,
            aggregate_response
        ]
        
        # Mock agent graph
        mock_agent_graph = Mock()
        mock_agent_graph.invoke.return_value = {
            "gathered_information": [{"tool": "search", "result": "info"}],
            "final_answer": "Sub answer",
            "hypotheses": ["hyp"],
            "decomposed_problems": []
        }
        mock_create_agent.return_value = mock_agent_graph
        
        orchestrator = MultiAgentOrchestrator(
            llm=mock_llm,
            tools=[mock_tool],
            max_workers=2
        )
        
        # Execute
        result = orchestrator.execute("Test question")
        
        # Verify
        self.assertIn("original_question", result)
        self.assertIn("final_answer", result)
        self.assertGreater(result["num_sub_tasks"], 0)


if __name__ == "__main__":
    unittest.main()
