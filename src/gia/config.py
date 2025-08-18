# src/gia/config.py
# IMPORT STANDARD LIBRARIES FOR CONFIG HANDLING
import os
from pathlib import Path
from typing import Dict, Optional, Any

# PROJECT ROOT FOR ALL PATH RESOLUTIONS
PROJECT_ROOT = Path(__file__).resolve().parent
# User-editable configuration section
CONTEXT_WINDOW = 32768
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY = 1.1
NO_REPEAT_NGRAM_SIZE = 4
DEBUG = True

MODEL_PATH = "~/models/quantized"
EMBED_MODEL_PATH = "~/models/bge-large-en-v1.5"
DATA_PATH = "MyData"  # RELATIVE TO PROJECT_ROOT/src/gia
DB_PATH = "db"  # RELATIVE TO PROJECT_ROOT/src/gia
RULES_PATH = "db/rules.json"  # RELATIVE TO PROJECT_ROOT/src/gia

# LOAD ENVIRONMENT VARIABLES WITH ERROR HANDLING
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError as e:
    raise ImportError("dotenv not found; install via pip install python-dotenv") from e
except Exception as e:
    raise RuntimeError(f"Failed to load .env: {e}") from e

# Internal logic to populate CONFIG dictionary
CONFIG: Dict[str, Optional[Any]] = {
    "CONTEXT_WINDOW": CONTEXT_WINDOW,
    "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
    "TEMPERATURE": TEMPERATURE,
    "TOP_P": TOP_P,
    "TOP_K": TOP_K,
    "REPETITION_PENALTY": REPETITION_PENALTY,
    "NO_REPEAT_NGRAM_SIZE": NO_REPEAT_NGRAM_SIZE,
    "DEBUG": DEBUG,
    "MODEL_PATH": os.getenv("MODEL_PATH", os.path.expanduser(MODEL_PATH)),
    "EMBED_MODEL_PATH": os.getenv("EMBED_MODEL_PATH", os.path.expanduser(EMBED_MODEL_PATH)),
    "RULES_PATH": os.getenv("RULES_PATH", str(PROJECT_ROOT / RULES_PATH)),
    "DATA_PATH": os.getenv("DATA_PATH", str(PROJECT_ROOT / DATA_PATH)),
    "DB_PATH": os.getenv("DB_PATH", str(PROJECT_ROOT / DB_PATH)),
}

# VALIDATE PATHS EXIST OR CREATE IF NEEDED
for key, path in CONFIG.items():
    if isinstance(path, str) and ("PATH" in key):
        try:
            path_obj = Path(path)
            if "RULES_PATH" in key:  # CHECK EXISTENCE FOR FILES
                if not path_obj.exists():
                    # TO LOG WARNING INSTEAD OF RAISE FOR NON-BLOCKING
                    print(f"Warning: Required file not found: {path}. Using fallback in system_prefix.")
            else:
                path_obj.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create/validate {key}: {path} - {e}") from e

# Function to retrieve config values
def get_config(key: str, default: Optional[Any] = None) -> Optional[Any]:
    """Retrieve config value or default with error handling."""
    try:
        return CONFIG.get(key, default)
    except KeyError as e:
        raise KeyError(f"Missing config key: {key}") from e
    