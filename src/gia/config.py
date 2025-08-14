# src/gia/config.py

# IMPORT STANDARD LIBRARIES FOR CONFIG HANDLING
import os
from pathlib import Path
from typing import Dict, Optional, Any
import json  # FOR POTENTIAL JSON CONFIG EXPANSION

# PROJECT ROOT FOR ALL PATH RESOLUTIONS - TO ANCHOR ALL RELATIVE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# LOAD ENVIRONMENT VARIABLES WITH ERROR HANDLING - TO CATCH MISSING ENV
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError as e:
    raise ImportError("dotenv not found; install via pip install python-dotenv") from e
except Exception as e:
    raise RuntimeError(f"Failed to load .env: {e}") from e

CONFIG: Dict[str, Optional[Any]] = {
    # LLM CONFIGURATION - FOR CONSISTENT USER SETTINGS
    "CONTEXT_WINDOW": 32768,
    "MAX_NEW_TOKENS": 4096,
    "TEMPERATURE": 0.7,
    "TOP_P": 0.8,
    "TOP_K": 20,
    "REPETITION_PENALTY": 1.1,
    "NO_REPEAT_NGRAM_SIZE": 4,
    # DEVICE AND DEBUG - FOR RUNTIME CONTROLS
    "DEBUG": False,
    # PATHS - UPPERCASE NAMING FOR STANDARDIZATION
    "MODEL_PATH": os.getenv("MODEL_PATH", str(Path.home() / "models" / "quantized")),
    "EMBED_MODEL_PATH": os.getenv(
        "EMBED_MODEL_PATH", str(Path.home() / "models" / "bge-large-en-v1.5")
    ),
    "DATA_PATH": os.getenv(
        "DATA_PATH", str(PROJECT_ROOT / "gia" / "downloads" / "extracted")
    ),
    "DB_PATH": os.getenv("DB_PATH", str(PROJECT_ROOT / "gia" / "db")),
    "RULES_PATH": os.getenv(
        "RULES_PATH", str(PROJECT_ROOT / "gia" / "db" / "rules.json")
    ),
}

# VALIDATE PATHS EXIST OR CREATE IF NEEDED - FOR ROBUST STARTUP
for key, path in CONFIG.items():
    if isinstance(path, str) and ("PATH" in key):
        try:
            path_obj = Path(path)
            if "RULES_PATH" in key:  # CHECK EXISTENCE FOR FILES
                if not path_obj.exists():
                    raise FileNotFoundError(f"Required file not found: {path}")
            else:
                path_obj.mkdir(parents=True, exist_ok=True)
        except (OSError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to create/validate {key}: {path} - {e}") from e


def get_config(key: str, default: Optional[Any] = None) -> Optional[Any]:
    """Retrieve config value or default with error handling.

    Args:
        key (str): Configuration key to retrieve.
        default (Any, optional): Default value if key is not found.

    Returns:
        Any: Configuration value or default.

    Raises:
        KeyError: If key is missing and no default is provided.
    """
    # TO SAFELY ACCESS CONFIG WITH FALLBACK
    try:
        return CONFIG.get(key, default)
    except KeyError as e:
        raise KeyError(f"Missing config key: {key}") from e
