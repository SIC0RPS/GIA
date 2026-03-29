# src/gia/__init__.py
"""Public API for GIA with lazy exports to avoid import-time side effects."""

from importlib import import_module
from typing import Any

__all__ = [
    "generate",
    "update_database",
    "clear_vram",
    "save_database",
    "load_database",
    "get_system_info",
    "append_to_chatbot",
    "state_manager",
    "load_state",
    "logger",
    "log_banner",
]

_UTIL_EXPORTS = {
    "generate",
    "update_database",
    "clear_vram",
    "save_database",
    "load_database",
    "get_system_info",
    "append_to_chatbot",
}
_STATE_EXPORTS = {"state_manager", "load_state"}
_LOGGER_EXPORTS = {"logger", "log_banner"}


def __getattr__(name: str) -> Any:
    if name in _UTIL_EXPORTS:
        module = import_module(".core.utils", __name__)
        return getattr(module, name)
    if name in _STATE_EXPORTS:
        module = import_module(".core.state", __name__)
        return getattr(module, name)
    if name in _LOGGER_EXPORTS:
        module = import_module(".core.logger", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
