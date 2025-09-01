# src/gia/core/state.py
"""State management for GIA project.
This module provides a singleton StateManager class to manage project-wide state
with thread safety. It uses a dataclass for structured state access and ensures
that state values can be accessed and modified safely across different modules.
"""

from dataclasses import dataclass, fields
from typing import Any, Optional, List
import threading
from gia.config import CONFIG
from gia.core.logger import log_banner

log_banner(__file__)


@dataclass
class ProjectState:
    """Dataclass for dot access to state values with fallbacks."""

    CHROMA_COLLECTION: Any = None
    EMBED_MODEL: Any = None
    LLM: Any = None
    QUERY_ENGINE: Any = None
    MODEL_NAME: Any = None
    MODEL_PATH: Any = None
    EMBED_MODEL_PATH: Any = None
    DATABASE_LOADED: bool = False
    INDEX: Any = None
    MODEL: Any = None
    TOKENIZER: Any = None
    STREAMER: Any = None
    USE_CHAT: bool = True  # FLAG FOR CHAT VS COMPLETE
    QA_PROMPT: List[str] | None = None
    MODEL_TYPE: str | None = None
    MODE: str | None = None  # EXPLICIT MODE: "Local" OR "Online"; NONE INITIAL
    COLLECTION_NAME: str | None = None


class StateManager:
    """Manage project-wide state safely across modules (singleton)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # run once
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Validate collection name coming from CONFIG (which already reads config.toml)
        collection_name = CONFIG.get("COLLECTION_NAME")
        if not isinstance(collection_name, str) or not (
            3 <= len(collection_name) <= 63
        ):
            collection_name = "GIA_db"

        self._state = {
            "CHROMA_COLLECTION": None,
            "EMBED_MODEL": None,
            "LLM": None,
            "QUERY_ENGINE": None,
            "MODEL_NAME": None,
            "MODEL_PATH": CONFIG.get("MODEL_PATH"),
            "EMBED_MODEL_PATH": CONFIG.get("EMBED_MODEL_PATH"),
            "DATABASE_LOADED": False,
            "INDEX": None,
            "MODEL": None,
            "TOKENIZER": None,
            "STREAMER": None,
            "USE_CHAT": True,
            "QA_PROMPT": CONFIG.get("QA_PROMPT"),
            "MODEL_TYPE": CONFIG.get("MODEL_TYPE"),
            "MODE": None,  # "Local" or "Online"
            "COLLECTION_NAME": collection_name,
        }
        self._lock = threading.Lock()
        self._initialized = True

    def set_state(self, key: str, value: Any) -> None:
        """Thread-safe set."""
        with self._lock:
            self._state[key] = value

    def get_state(self, key: str, default: Optional[Any] = None) -> Any:
        """Thread-safe get with default."""
        with self._lock:
            return self._state.get(key, default)


state_manager = StateManager()


def load_state() -> ProjectState:
    """Snapshot current state into a dataclass (dot-access)."""
    state_dict = {
        f.name: state_manager.get_state(f.name, f.default) for f in fields(ProjectState)
    }
    return ProjectState(**state_dict)
