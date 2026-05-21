from .protocols import MemoryProvider, MessageLike
from .utils import normalize_message, normalize_messages, format_messages_as_text
from .config import FMemConfig, DEFAULT_CONFIG
from .fs_mem.client import FMemClient
from .fs_mem.core.context_builder import ContextBuilder
from .fs_mem.core.memory_searcher import MemorySearcher
from .fs_mem.core.memory_writer import MemoryWriter
from .fs_mem.core.directory_manager import DirectoryManager

# Keep FMem for backward compatibility
FMem = FMemClient

__all__ = [
    # Protocols and utilities
    "MemoryProvider",
    "MessageLike",
    "normalize_message",
    "normalize_messages",
    "format_messages_as_text",
    
    # Configuration
    "FMemConfig",
    "DEFAULT_CONFIG",
    
    # Main client
    "FMemClient",
    "FMem",  # Backward compatibility alias
    
    # Core components
    "ContextBuilder",
    "MemorySearcher",
    "MemoryWriter",
    "DirectoryManager",
]