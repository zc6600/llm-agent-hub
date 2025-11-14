"""
System prompts for the Parallel React Agent.

Defines prompts for individual ReAct agents and the summarizing agent.
"""

# Prompt for individual ReAct agents
PARALLEL_REACT_AGENT_SYSTEM_PROMPT = """You are a research assistant that provides deep and comprehensive answers to questions.

When answering the question, follow these steps:
1. Break down the question into key components
2. Use available tools to gather relevant information
3. Synthesize the information to provide a well-reasoned answer
4. Be thorough, evidence-based, and critical in your analysis"""

# Prompt for summarizing agent
SUMMARIZING_AGENT_SYSTEM_PROMPT = """You are a summarizing assistant specialized in synthesizing research results.

Your task is to:
1. Analyze responses from different research agents exploring various angles
2. Identify the internal structure and logical connections between different hypotheses
3. Integrate responses from different aspects logically and coherently
4. Synthesize all information into a unified, well-organized summary

Focus on creating a cohesive narrative that connects all the pieces of information
into a comprehensive and insightful summary."""


def get_combined_system_prompt(
    agent_type: str,
    user_system_prompt: str = ""
) -> str:
    """
    Combine architecture system prompt with user-provided system prompt.
    
    Args:
        agent_type: "react" for individual agents or "summarizing" for summary agent
        user_system_prompt: Optional user-provided system prompt
        
    Returns:
        Combined system prompt
    """
    if agent_type == "react":
        base_prompt = PARALLEL_REACT_AGENT_SYSTEM_PROMPT
    elif agent_type == "summarizing":
        base_prompt = SUMMARIZING_AGENT_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")
    
    if user_system_prompt:
        # Combine prompts
        combined = f"{base_prompt}\n\nAdditional instructions:\n{user_system_prompt}"
        return combined
    
    return base_prompt
