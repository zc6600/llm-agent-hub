"""Compatibility patches for Deep Diver agent dependencies."""

from __future__ import annotations

try:
    from langchain_core.messages import RemoveMessage  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    # Some langchain-core releases (>=1.0) removed RemoveMessage from public API.
    # LangGraph still imports it, so we provide a lightweight stub to keep
    # backward compatibility. The stub only retains the interface used by
    # LangGraph (it inherits from object and stores data payloads).
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class RemoveMessage:  # type: ignore
        """Fallback implementation mirroring legacy RemoveMessage contract."""

        id: str | None = None
        metadata: dict[str, Any] | None = None

    import langchain_core.messages as _lc_messages  # type: ignore

    if not hasattr(_lc_messages, "RemoveMessage"):
        _lc_messages.RemoveMessage = RemoveMessage  # type: ignore[attr-defined]
