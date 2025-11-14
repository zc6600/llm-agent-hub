"""
LLM Provider Module

This module provides a centralized way to initialize and configure LLM instances
with support for environment variables and LangSmith monitoring.
"""

import os
from typing import Optional
from pathlib import Path
from langchain_openai import ChatOpenAI


# Load environment variables from .env file
def _load_env():
    """Load .env file from project root or current directory."""
    # Try multiple locations for .env file
    possible_paths = [
        Path(__file__).resolve().parent.parent / ".env",  # src/.. = project root
        Path.cwd() / ".env",  # Current working directory
    ]
    
    for env_path in possible_paths:
        if env_path.exists():
            _load_env_file(env_path)
            return env_path
    
    return None


def _load_env_file(env_path: Path):
    """Manually load environment variables from .env file."""
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Only set if not already set
                    if key not in os.environ:
                        os.environ[key] = value
    except Exception as e:
        print(f"[Warning] Failed to load .env file from {env_path}: {e}")


# Load env on module import
_load_env()


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 100000,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    enable_langsmith: bool = True,
) -> ChatOpenAI:
    """
    Initialize and return a configured ChatOpenAI LLM instance.
    
    Args:
        model: The model to use. If None, uses OPENROUTER_MODEL from .env
        temperature: Temperature parameter for the model (0-1)
        max_tokens: Maximum tokens in response
        api_key: API key. If None, uses OPENROUTER_API_KEY or OPENAI_API_KEY from .env
        base_url: Base URL for API. If None, uses OpenRouter default for OpenRouter API key
        enable_langsmith: Whether to enable LangSmith monitoring
        
    Returns:
        Configured ChatOpenAI instance
    """
    
    # Ensure environment variables are loaded
    _load_env()
    
    # Configure LangSmith if enabled
    if enable_langsmith:
        _configure_langsmith()
    
    # Get API key from parameter, .env, or raise error
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key found. Please set OPENROUTER_API_KEY or OPENAI_API_KEY in .env file"
            )
    
    # Get model from parameter or .env
    if model is None:
        model = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
    
    # Set base URL for OpenRouter if using OpenRouter API key and no custom base_url
    if base_url is None and "sk-or-v1-" in api_key:
        base_url = "https://openrouter.ai/api/v1"
    
    # Log configuration
    print(f"[LLM Provider] Initializing LLM:")
    print(f"  - Model: {model}")
    print(f"  - Temperature: {temperature}")
    print(f"  - Max Tokens: {max_tokens}")
    if base_url:
        print(f"  - Base URL: {base_url}")
    print(f"  - LangSmith Monitoring: {'Enabled' if enable_langsmith else 'Disabled'}")
    
    # Initialize and return ChatOpenAI
    llm_kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    if base_url:
        llm_kwargs["base_url"] = base_url
    
    return ChatOpenAI(**llm_kwargs)


def _configure_langsmith() -> None:
    """Configure LangSmith monitoring from environment variables."""
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    langsmith_project = os.getenv("LANGSMITH_PROJECT", "llm-tool-hub")
    
    if langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        print(f"[LangSmith] ✓ Configured successfully")
        print(f"[LangSmith] Project: {langsmith_project}")
        print(f"[LangSmith] Endpoint: https://api.smith.langchain.com")
        print(f"[LangSmith] API Key: {'***' + langsmith_api_key[-10:] if len(langsmith_api_key) > 10 else '***'}")
    else:
        print("[LangSmith] ⚠ API key not found in .env, LangSmith monitoring disabled")
        print("[LangSmith] To enable, set LANGSMITH_API_KEY in .env file")


def get_llm_with_custom_config(
    config: dict,
    enable_langsmith: bool = True,
) -> ChatOpenAI:
    """
    Initialize LLM with custom configuration dictionary.
    
    Args:
        config: Dictionary with keys: model, temperature, max_tokens, api_key, base_url
        enable_langsmith: Whether to enable LangSmith monitoring
        
    Returns:
        Configured ChatOpenAI instance
    """
    return get_llm(
        model=config.get("model"),
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 100000),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        enable_langsmith=enable_langsmith,
    )


__all__ = ["get_llm", "get_llm_with_custom_config"]
