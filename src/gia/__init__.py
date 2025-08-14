# src/gia/__init__.py
"""PUBLIC API FOR GIA. EXPLICIT RE-EXPORTS; PEP 8/257, BLACK, FLAKE8 COMPLIANT."""

from .core.utils import (
    generate,
    update_database,
    clear_vram,
    save_database,
    load_database,
    get_system_info,
    append_to_chatbot,
    PresencePenaltyLogitsProcessor,  # ADD: USED BY CALL SITES
)
from .core.state import state_manager
from .core.state import load_state
from .core.logger import logger, log_banner

__all__ = [
    "generate",
    "update_database",
    "clear_vram",
    "save_database",
    "load_database",
    "get_system_info",
    "append_to_chatbot",
    "PresencePenaltyLogitsProcessor",
    "state_manager",
    "load_state",
    "logger",
    "log_banner",
]
