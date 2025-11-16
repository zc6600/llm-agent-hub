"""
System prompts for the Parallel Direct Tool Agent.

Defines the default prompts used for result summarization.
"""


DEFAULT_SYSTEM_PROMPT = """You are a research assistant helping to synthesize information from multiple sources.
Your task is to create a coherent summary of the gathered information."""


SUMMARIZATION_PROMPT_TEMPLATE = """You have gathered information from multiple research queries.
Please synthesize this information into a coherent summary.

Queries and Results:
{combined_results}

Provide a comprehensive summary that:
1. Integrates information from all queries
2. Highlights key findings and patterns
3. Maintains academic rigor and citation awareness
4. Organizes information logically

Summary:"""


def get_combined_system_prompt(user_prompt: str = "") -> str:
    """
    Combine default system prompt with user-provided prompt.
    
    Args:
        user_prompt: User-provided system prompt
        
    Returns:
        Combined system prompt
    """
    if user_prompt:
        return f"{DEFAULT_SYSTEM_PROMPT}\n\nAdditional instructions:\n{user_prompt}"
    return DEFAULT_SYSTEM_PROMPT
