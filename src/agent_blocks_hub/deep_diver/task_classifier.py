"""
Task classifier for determining workflow complexity.

Classifies user queries to determine whether they need full scientific
method iteration (hypothesis-verification) or just factual lookup.
Considers available tools when making classification decisions.
"""

from typing import Dict, Any, Optional, Literal, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool


TaskType = Literal["simple", "complex", "auto"]


class TaskClassifier:
    """
    Classifies tasks to determine appropriate workflow path.
    
    Task types:
    - "simple": Factual queries that need info gathering but not hypothesis testing
      Examples: "What is LangGraph?", "How do I install X?", "Who is Y?"
      
    - "complex": Research/analysis tasks requiring hypothesis generation and testing
      Examples: "How should I architect X?", "What's the best approach for Y?",
      "Analyze and compare X vs Y", "Design a system that does X"
    """
    
    CLASSIFICATION_PROMPT = """Analyze the following user question and classify whether it's a "simple" or "complex" task.

USER QUESTION: {question}

AVAILABLE TOOLS:
{available_tools}

Classification Guidelines:

**SIMPLE Tasks** (factual/definitional):
- Asking for definitions or explanations
- Requesting instructions or how-to information
- Asking for factual information about existing things
- Can be resolved with available tools and information gathering
- Examples: "What is X?", "How do I use X?", "Tell me about X"
- Resolution: Information gathering + summarization

**COMPLEX Tasks** (analytical/research):
- Asking for design/architecture decisions
- Comparing multiple options with trade-offs
- Solving novel problems
- Understanding deep concepts with practical implications
- Requires analysis, synthesis, and hypothesis testing
- Cannot be directly answered by available tools
- Examples: "How should I architect X?", "What's better: X or Y?", "How do I solve X?"
- Resolution: Information gathering + hypothesis generation + verification

TOOL CONSIDERATION:
- If strong search tools are available (e.g., web search, API access), simple queries can be resolved quickly
- If limited tools, even simple queries may need deeper reasoning
- Consider both tool capability and query complexity

Return your classification as valid JSON with this exact format:
{{"task_type": "simple" or "complex", "confidence": 0.0-1.0, "reasoning": "explanation", "tool_consideration": "how tools affected this classification"}}

Ensure the JSON is complete and valid."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None
    ):
        """
        Initialize classifier.
        
        Args:
            llm: Language model for classification. If None, uses simple heuristics.
            tools: List of available tools that can affect classification
        """
        self.llm = llm
        self.tools = tools or []
    
    def classify(
        self,
        question: str,
        task_type: TaskType = "auto",
        tools: Optional[List[BaseTool]] = None
    ) -> Dict[str, Any]:
        """
        Classify a task.
        
        Args:
            question: The user's question/task
            task_type: "auto" (use LLM), "simple", or "complex" (force type)
            tools: Optional list of available tools. If provided, overrides init tools.
            
        Returns:
            Dict with keys:
            - task_type: "simple" or "complex"
            - confidence: float between 0 and 1
            - reasoning: explanation of classification
            - tool_consideration: how tools affected the decision
            - use_hypothesis: bool (whether to include hypothesis generation)
        """
        if task_type in ["simple", "complex"]:
            # Use forced type with high confidence
            return {
                "task_type": task_type,
                "confidence": 1.0,
                "reasoning": f"Forced to {task_type} task type",
                "tool_consideration": "Classification forced by user override",
                "use_hypothesis": task_type == "complex"
            }
        
        # Use provided tools or fall back to init tools
        available_tools = tools if tools is not None else self.tools
        
        # Use LLM classification if available
        if self.llm:
            return self._classify_with_llm(question, available_tools)
        else:
            return self._classify_with_heuristics(question, available_tools)
    
    def _classify_with_llm(
        self,
        question: str,
        tools: Optional[List[BaseTool]] = None
    ) -> Dict[str, Any]:
        """Classify using LLM."""
        # Format available tools
        available_tools = tools or []
        if available_tools:
            tools_str = "\n".join([
                f"- {tool.name}: {tool.description or 'No description'}"
                for tool in available_tools
            ])
        else:
            tools_str = "No tools available"
        
        prompt = self.CLASSIFICATION_PROMPT.format(
            question=question,
            available_tools=tools_str
        )
        
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        # Parse JSON response
        try:
            import json
            
            content = response.content.strip()
            
            # Extract JSON
            if "{" in content and "}" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                json_str = content[start:end]
                result = json.loads(json_str)
                
                task_type = result.get("task_type", "simple")
                confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "")
                tool_consideration = result.get("tool_consideration", "")
                
                return {
                    "task_type": task_type,
                    "confidence": float(confidence),
                    "reasoning": reasoning,
                    "tool_consideration": tool_consideration,
                    "use_hypothesis": task_type == "complex"
                }
        except Exception as e:
            print(f"[TaskClassifier] Error parsing LLM response: {e}")
        
        # Fallback to heuristics
        return self._classify_with_heuristics(question, tools)
    
    def _classify_with_heuristics(
        self,
        question: str,
        tools: Optional[List[BaseTool]] = None
    ) -> Dict[str, Any]:
        """Classify using simple heuristic rules with tool consideration."""
        available_tools = tools or []
        question_lower = question.lower()
        
        # Check if we have powerful search/retrieval tools
        has_search_tool = any(
            "search" in tool.name.lower() or 
            "retrieval" in tool.name.lower() or
            "query" in tool.name.lower()
            for tool in available_tools
        )
        
        # Keywords that indicate complex tasks
        complex_keywords = [
            "how should", "how can i", "best way", "architecture", "design",
            "compare", "vs", "advantage", "disadvantage", "trade-off",
            "analyze", "why", "solve", "problem", "challenge",
            "strategy", "approach", "implement", "build", "create",
            "optimize", "improve", "recommend", "suggest"
        ]
        
        # Keywords that indicate simple tasks
        simple_keywords = [
            "what is", "what are", "how do i", "how to", "tell me about",
            "explain", "definition", "install", "setup", "configure",
            "tutorial", "guide", "example", "documentation", "help"
        ]
        
        # Count keyword matches
        complex_score = sum(1 for kw in complex_keywords if kw in question_lower)
        simple_score = sum(1 for kw in simple_keywords if kw in question_lower)
        
        # Tool consideration
        tool_consideration = ""
        tool_impact = 0.0  # Bonus/penalty for task type
        
        if has_search_tool:
            tool_consideration = "Search tools available - simple queries can be resolved faster"
            # With search tools, favor simple path for factual queries
            if simple_score > 0:
                tool_impact = 0.15  # Boost simple classification
        else:
            tool_consideration = "Limited tools available - may need deeper reasoning"
            # Without tools, even simple queries might need more reasoning
            if simple_score > 0:
                tool_impact = -0.1  # Slightly reduce confidence in simple path
        
        # Decide based on scores
        if complex_score > simple_score:
            task_type = "complex"
            confidence = min(0.95, 0.6 + complex_score * 0.1)
        elif simple_score > complex_score:
            task_type = "simple"
            confidence = min(0.95, 0.6 + simple_score * 0.1 + tool_impact)
        else:
            # Default: if question is long and complex-looking, likely complex
            task_type = "complex" if len(question) > 80 else "simple"
            confidence = 0.5 + tool_impact
        
        # Ensure confidence stays in valid range
        confidence = max(0.3, min(0.95, confidence))
        
        return {
            "task_type": task_type,
            "confidence": confidence,
            "reasoning": f"Heuristic classification based on keywords (complex: {complex_score}, simple: {simple_score})",
            "tool_consideration": tool_consideration,
            "use_hypothesis": task_type == "complex"
        }
