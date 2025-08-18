# In state.py (confirmed get_state signature with default to prevent TypeError)

# src/gia/core/state_manager.py
"""State management for GIA project.
This module provides a singleton StateManager class to manage project-wide state
with thread safety. It uses a dataclass for structured state access and ensures
that state values can be accessed and modified safely across different modules.
"""

from dataclasses import dataclass, fields
from typing import Any, Optional
import threading
from gia.config import CONFIG, PROJECT_ROOT

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

def load_state() -> ProjectState:
    """Load or initialize the current project state."""
    state_dict = {
        f.name: state_manager.get_state(f.name, f.default) or f.default
        for f in fields(ProjectState)
    }
    return ProjectState(**state_dict)

class StateManager:
    """Manage project-wide state safely across modules."""

    _instance = None

    def __new__(cls):
        # TO ENSURE SINGLETON INSTANCE
        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def __init__(self):
        # TO INITIALIZE STATE AND LOCK ONCE
        if not hasattr(self, "_state"):
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
            }
            self._lock = threading.Lock()

    def set_state(self, key: str, value: Any) -> None:
        """Set a state value with thread safety.

        Args:
            key (str): The key to set in the state dictionary.
            value (Any): The value to associate with the key.
        """
        # TO THREAD-SAFE SET OPERATION
        with self._lock:
            self._state[key] = value

    def get_state(self, key: str, default: Optional[Any] = None) -> Any:
        """Retrieve a state value with optional default.

        Args:
            key (str): The key to retrieve from the state dictionary.
            default (Optional[Any]): Default value if key not found.

        Returns:
            Any: The value associated with the key, or default if not found.
        """
        # TO THREAD-SAFE GET OPERATION WITH DEFAULT
        with self._lock:
            return self._state.get(key, default)

# TO CREATE SINGLETON INSTANCE FOR PROJECT-WIDE USE
state_manager = StateManager()
"""Global StateManager instance for thread-safe state management across modules."""