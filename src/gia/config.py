# src/gia/config.py
import os
import tomllib
from pathlib import Path
from typing import Dict, Optional, Any, Union
from transformers import PretrainedConfig

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULTS: Dict[str, Any] = {
    "CONTEXT_WINDOW": 32768,
    "MAX_NEW_TOKENS": 2048,
    "TEMPERATURE": 0.7,
    "TOP_P": 0.8,
    "TOP_K": 20,
    "REPETITION_PENALTY": 1.1,
    "NO_REPEAT_NGRAM_SIZE": 4,
    "DEBUG": True,
    "PLUGIN_IDLE_TIMEOUT_SECONDS": 0,
    "MODEL_PATH": "~/models/quantized",
    "EMBED_MODEL_PATH": "~/models/bge-large-en-v1.5",
    "DATA_PATH": "MyData",
    "DB_PATH": "db",
    "QA_PROMPT": [
        "Context information is below.",
        "###",
        "{context_str}",
        "###",
        "{query_str}",
    ],
    "COLLECTION_NAME": "GIA_db",
}

DEFAULT_SYSTEM_PROMPT = "You are an expert assistant. Provide accurate, reliable, and evidence-based answers."

config_path = PROJECT_ROOT.parent.parent / "config.toml"
config_dict: Dict[str, Any] = {}
if config_path.exists():
    try:
        with config_path.open("rb") as f:
            config_dict = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        from gia.core.logger import logger

        logger.info(
            f"Warning: Failed to parse config.toml: {e}. Falling back to defaults."
        )
    except OSError as e:
        from gia.core.logger import logger

        logger.info(
            f"Warning: Failed to read config.toml: {e}. Falling back to defaults."
        )
else:
    from gia.core.logger import logger

    logger.info("Warning: config.toml not found. Falling back to defaults.")

CONFIG: Dict[str, Optional[Any]] = {}
CONFIG["MODEL_PATH"] = str(
    Path(
        os.path.expanduser(
            config_dict.get("paths", {}).get("model_path", DEFAULTS["MODEL_PATH"])
        )
    ).resolve()
)
CONFIG["EMBED_MODEL_PATH"] = str(
    Path(
        os.path.expanduser(
            config_dict.get("paths", {}).get(
                "embed_model_path", DEFAULTS["EMBED_MODEL_PATH"]
            )
        )
    ).resolve()
)
CONFIG["DATA_PATH"] = str(
    (
        PROJECT_ROOT
        / config_dict.get("paths", {}).get("data_path", DEFAULTS["DATA_PATH"])
    ).resolve()
)
CONFIG["DB_PATH"] = str(
    (
        PROJECT_ROOT / config_dict.get("paths", {}).get("db_path", DEFAULTS["DB_PATH"])
    ).resolve()
)


def get_default_generation_settings(
    model_path: str,
) -> Dict[str, Optional[Union[int, float, str]]]:
    """Extract selected defaults from model config.json & generation_config.json."""
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("model_path must be a non-empty string.")
    settings: Dict[str, Optional[Union[int, float, str]]] = {
        "context_window": None,
        "max_new_tokens": None,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "repetition_penalty": None,
        "no_repeat_ngram_size": None,
        "model_type": None,
    }
    try:
        cfg, _ = PretrainedConfig.get_config_dict(model_path)
    except EnvironmentError as e:
        raise ValueError(f"Failed to load config from '{model_path}': {str(e)}") from e
    if cfg:
        for key in ("max_position_embeddings", "n_positions", "model_max_length"):
            if key in cfg and isinstance(cfg[key], int):
                settings["context_window"] = cfg[key]
                break
        if isinstance(cfg.get("model_type"), str):
            settings["model_type"] = cfg["model_type"]
    try:
        gen_cfg, _ = PretrainedConfig.get_config_dict(
            model_path, _filename="generation_config.json"
        )
    except EnvironmentError:
        gen_cfg = {}
    if gen_cfg:
        if isinstance(gen_cfg.get("max_new_tokens"), int):
            settings["max_new_tokens"] = gen_cfg["max_new_tokens"]
        if isinstance(gen_cfg.get("temperature"), (int, float)):
            settings["temperature"] = float(gen_cfg["temperature"])
        if isinstance(gen_cfg.get("top_p"), (int, float)):
            settings["top_p"] = float(gen_cfg["top_p"])
        if isinstance(gen_cfg.get("top_k"), int):
            settings["top_k"] = gen_cfg["top_k"]
        if isinstance(gen_cfg.get("repetition_penalty"), (int, float)):
            settings["repetition_penalty"] = float(gen_cfg["repetition_penalty"])
        if isinstance(gen_cfg.get("no_repeat_ngram_size"), int):
            settings["no_repeat_ngram_size"] = gen_cfg["no_repeat_ngram_size"]
    return settings


try:
    model_defaults = get_default_generation_settings(CONFIG["MODEL_PATH"])
except ValueError:
    model_defaults = {}
    from gia.core.logger import logger

    logger.info(
        f"Warning: Failed to load model defaults from '{CONFIG['MODEL_PATH']}'. Falling back to hardcoded defaults."
    )

generation_keys = [
    "context_window",
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "no_repeat_ngram_size",
]
for k in generation_keys:
    upper_k = k.upper()
    val = config_dict.get("generation", {}).get(
        k, model_defaults.get(k, DEFAULTS.get(upper_k))
    )
    CONFIG[upper_k] = val
CONFIG["MODEL_TYPE"] = model_defaults.get("model_type", None)

# ----- GENERAL / PROMPTS -----
CONFIG["DEBUG"] = config_dict.get("general", {}).get("debug", DEFAULTS["DEBUG"])
CONFIG["PLUGIN_IDLE_TIMEOUT_SECONDS"] = config_dict.get("general", {}).get(
    "plugin_idle_timeout_seconds",
    DEFAULTS["PLUGIN_IDLE_TIMEOUT_SECONDS"],
)

qa_prompt_data = config_dict.get("prompt", {}).get("qa_prompt", DEFAULTS["QA_PROMPT"])
if isinstance(qa_prompt_data, str):
    CONFIG["QA_PROMPT"] = [qa_prompt_data]
elif isinstance(qa_prompt_data, list) and all(
    isinstance(item, str) for item in qa_prompt_data
):
    CONFIG["QA_PROMPT"] = qa_prompt_data
else:
    from gia.core.logger import logger

    logger.info(
        "Warning: Invalid 'prompt.qa_prompt' in config.toml (must be str or list[str]). Falling back to default."
    )
    CONFIG["QA_PROMPT"] = DEFAULTS["QA_PROMPT"]


def _validate_collection_name(name: Any, default: str = "GIA_db") -> str:
    """Ensure a non-empty 3..63 char string for Chroma collection names."""
    if not isinstance(name, str):
        return default
    n = len(name)
    if n < 3 or n > 63:
        return default
    return name


_raw_name = config_dict.get("database", {}).get(
    "collection_name", DEFAULTS["COLLECTION_NAME"]
)
CONFIG["COLLECTION_NAME"] = _validate_collection_name(
    _raw_name, DEFAULTS["COLLECTION_NAME"]
)

for key, value in CONFIG.items():
    if isinstance(value, str) and "PATH" in key:
        try:
            path_obj = Path(value)
            path_obj.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create/validate {key}: {value} - {e}") from e


def get_config(key: str, default: Optional[Any] = None) -> Optional[Any]:
    """Retrieve a configuration value or return the default if not set."""
    return CONFIG.get(key, default)


def system_prefix() -> str:
    """
    Load system prompt dynamically from config.toml.
    Priority:
      1. prompt.system_prompt (string or list of strings)
      2. DEFAULT_SYSTEM_PROMPT
    """
    from gia.core.logger import logger

    config_path = PROJECT_ROOT.parent.parent / "config.toml"
    try:
        if config_path.exists():
            with config_path.open("rb") as f:
                _cfg = tomllib.load(f)
            prompt_data = _cfg.get("prompt", {}).get("system_prompt", None)
            if isinstance(prompt_data, str):
                return prompt_data
            if isinstance(prompt_data, list) and all(
                isinstance(item, str) for item in prompt_data
            ):
                return "\n".join(prompt_data)
            logger.info(
                "Warning: Invalid 'prompt.system_prompt' in config.toml (must be str or list[str]). Falling back."
            )
        else:
            logger.info(
                f"Warning: config.toml not found at {config_path}. Falling back to default system prompt."
            )
    except (tomllib.TOMLDecodeError, OSError) as e:
        logger.info(
            f"Error reading/parsing config.toml: {e}. Falling back to default system prompt."
        )
    except Exception as e:
        logger.info(f"Unexpected error in system_prefix: {e}")
    return DEFAULT_SYSTEM_PROMPT
