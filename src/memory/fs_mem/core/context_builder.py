import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ContextBuilder:
    def __init__(self, task_dir: str, config=None):
        self.task_dir = task_dir
        self.config = config
        if config:
            logger.setLevel(config.log_level)

    def _read_dir(self, name: str) -> str:
        p = Path(self.task_dir, name)
        if not p.exists():
            logger.debug(f"Directory does not exist: {p}")
            return ""
        
        texts = []
        max_file_size = self.config.max_file_size if self.config else 1024 * 1024  # 1MB default
        
        for f in sorted(p.glob("**/*")):
            if not f.is_file():
                continue
            
            try:
                # Check file size
                file_size = f.stat().st_size
                if file_size > max_file_size:
                    logger.warning(
                        f"Skipping large file: {f.name} ({file_size} bytes, "
                        f"limit: {max_file_size} bytes)"
                    )
                    texts.append(f"[File too large, skipped: {f.name}]")
                    continue
                
                # Try to read as text
                content = f.read_text(encoding="utf-8")
                texts.append(content)
                logger.debug(f"Read file: {f.name} ({file_size} bytes)")
                
            except UnicodeDecodeError:
                logger.warning(f"Skipping non-text file: {f.name}")
                texts.append(f"[Binary file, skipped: {f.name}]")
            except Exception as e:
                logger.error(f"Failed to read file: {f.name}, error: {e}")
                texts.append(f"[Read error: {f.name}]")
        
        return "\n".join(texts)

    def _load_current_state(self) -> str:
        p = Path(self.task_dir, "state", "current_state.txt")
        if p.exists():
            try:
                content = p.read_text(encoding="utf-8")
                logger.debug(f"Loaded current state: {len(content)} characters")
                return content
            except Exception as e:
                logger.error(f"Failed to load current state: {e}")
                return ""
        return ""

    def build_context(self, query: Optional[str] = None) -> str:
        """
        Build context from memory directories.
        
        Args:
            query: Optional query (currently unused, for future enhancement).
        
        Returns:
            Formatted context string.
        """
        rules = self._read_dir("rules")
        state = self._read_dir("state")
        preference = self._read_dir("preference")
        knowledge = self._read_dir("knowledge")
        
        parts = []
        if rules:
            parts.append("=== RULES ===\n" + rules)
        if state:
            parts.append("=== STATE ===\n" + state)
        if preference:
            parts.append("=== PREFERENCE ===\n" + preference)
        if knowledge:
            parts.append("=== KNOWLEDGE ===\n" + knowledge)
        
        context = "\n\n".join(parts)
        logger.debug(f"Built context: {len(context)} characters from {len(parts)} categories")
        
        return context