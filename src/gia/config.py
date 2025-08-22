# src/gia/config.py
# IMPORT STANDARD LIBRARIES AND TYPING
import os
import json  # For loading rules.json
from pathlib import Path
from typing import Dict, Optional, Any

# PROJECT ROOT FOR RELATIVE PATH RESOLUTION
PROJECT_ROOT = Path(__file__).resolve().parent

# DEFAULT CONFIGURATION VALUES (USER-EDITABLE)
CONTEXT_WINDOW = 32768
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY = 1.1
NO_REPEAT_NGRAM_SIZE = 4
DEBUG = True
DEFAULT_SYSTEM_PROMPT = "You are an expert assistant. Provide accurate, reliable, and evidence-based answers."

MODEL_PATH = "~/models/quantized"
EMBED_MODEL_PATH = "~/models/bge-large-en-v1.5"
DATA_PATH = "MyData"        # RELATIVE TO PROJECT_ROOT/src/gia
DB_PATH = "db"              
RULES_PATH = "db/rules.json"  

# LOAD ENVIRONMENT VARIABLES FROM .env IF AVAILABLE (NON-FATAL ON FAILURE)
try:
    import dotenv  # type: ignore
    from dotenv import load_dotenv
    load_dotenv(override=False) # Do not override existing environment variables
except ImportError:
    print("Warning: python-dotenv not installed; skipping .env loading.")
except FileNotFoundError as e:
    print(f"Warning: .env file not found: {e}")
except ValueError as e:
    print(f"Warning: Failed to parse .env file: {e}")

# BUILD CONFIGURATION DICTIONARY (NO API KEYS STORED; FETCH ON-DEMAND VIA OS.GETENV)
CONFIG: Dict[str, Optional[Any]] = {
    "CONTEXT_WINDOW": CONTEXT_WINDOW,
    "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
    "TEMPERATURE": TEMPERATURE,
    "TOP_P": TOP_P,
    "TOP_K": TOP_K,
    "REPETITION_PENALTY": REPETITION_PENALTY,
    "NO_REPEAT_NGRAM_SIZE": NO_REPEAT_NGRAM_SIZE,
    "DEBUG": DEBUG,
    "MODEL_PATH": os.path.expanduser(MODEL_PATH),
    "EMBED_MODEL_PATH": os.path.expanduser(EMBED_MODEL_PATH),
    "RULES_PATH": str(PROJECT_ROOT / RULES_PATH),
    "DATA_PATH": str(PROJECT_ROOT / DATA_PATH),
    "DB_PATH": str(PROJECT_ROOT / DB_PATH),
}

# VALIDATE FILE/DIRECTORY PATHS AND ENSURE THEY EXIST
for key, path in CONFIG.items():
    if isinstance(path, str) and ("PATH" in key):
        try:
            path_obj = Path(path)
            if "RULES_PATH" in key:
                if not path_obj.exists():
                    # WARN IF RULES FILE IS MISSING (WILL FALL BACK TO DEFAULT PROMPT)
                    print(f"Warning: Required file not found: {path}. Falling back to default system prompt.")
            else:
                path_obj.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create/validate {key}: {path} - {e}") from e

def get_config(key: str, default: Optional[Any] = None) -> Optional[Any]:
    """Retrieve a configuration value or return the default if not set."""
    return CONFIG.get(key, default)

def system_prefix() -> str:
    """
    Load system prompt from CONFIG or rules.json file, handling all edge cases.
    Priority:
        1. CONFIG["SYSTEM_PROMPT"] if defined.
        2. RULES_PATH file if exists and valid.
        3. DEFAULT_SYSTEM_PROMPT as fallback.
    """
    try:
        system_prompt = CONFIG.get("SYSTEM_PROMPT", None)
        rules_path_obj = Path(CONFIG["RULES_PATH"]) if CONFIG.get("RULES_PATH") else None

        # Use config prompt directly if defined
        if system_prompt:
            return str(system_prompt)

        # Attempt rules.json if available
        if rules_path_obj and rules_path_obj.exists():
            try:
                with rules_path_obj.open("r", encoding="utf-8") as f:
                    rules = json.load(f)

                prompt_data = rules.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
                if isinstance(prompt_data, str):
                    return prompt_data
                return "\n".join(prompt_data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading/parsing RULES_PATH '{rules_path_obj}': {e}")
                return DEFAULT_SYSTEM_PROMPT

        # Warn if rules file missing
        if rules_path_obj and not rules_path_obj.exists():
            print(f"Warning: RULES_PATH not found at {rules_path_obj}. Falling back to DEFAULT_SYSTEM_PROMPT.")

    except Exception as e:
        print(f"Unexpected error in system_prefix: {e}")

    return DEFAULT_SYSTEM_PROMPT
