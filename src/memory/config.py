"""
Configuration management for F-Mem.

This module provides centralized configuration for all memory components.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import logging


@dataclass
class FMemConfig:
    """
    Configuration for F-Mem memory system.
    
    Attributes:
        buffer_size: Number of messages to buffer before triggering automatic write.
        summary_length: Length of file summaries for search indexing (in characters).
        max_recent_messages: Maximum number of recent messages to include in context.
        memory_categories: List of top-level memory directory categories.
        max_file_size: Maximum file size to read (in bytes). Files larger than this
                      will be skipped to prevent memory issues.
        log_level: Logging level for F-Mem components.
        enable_validation: Whether to validate LLM-generated tool calls before execution.
    
    Examples:
        >>> config = FMemConfig(buffer_size=20, log_level=logging.DEBUG)
        >>> config.buffer_size
        20
    """
    
    # Memory writing settings
    buffer_size: int = 10
    
    # Memory search settings
    summary_length: int = 500
    
    # Context building settings
    max_recent_messages: int = 10
    memory_categories: List[str] = field(default_factory=lambda: [
        "rules", 
        "preference", 
        "state", 
        "knowledge"
    ])
    
    # File system settings
    max_file_size: int = 1024 * 1024  # 1MB
    
    # Logging settings
    log_level: int = logging.INFO
    
    # Validation settings
    enable_validation: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        if self.summary_length < 100:
            raise ValueError("summary_length must be at least 100")
        if self.max_recent_messages < 1:
            raise ValueError("max_recent_messages must be at least 1")
        if self.max_file_size < 1024:
            raise ValueError("max_file_size must be at least 1024 bytes (1KB)")
        if not self.memory_categories:
            raise ValueError("memory_categories cannot be empty")


# Default configuration instance
DEFAULT_CONFIG = FMemConfig()
