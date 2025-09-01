# src/gia/__main__.py
"""Entry point for launching the GIA application."""
import argparse
import os

# NOTE: We import logger/config first; levels/handlers were set at import time.
# We will adjust levels at runtime below if --debug is passed.
from gia.config import CONFIG  # ACCESS DEBUG SETTING
from gia.core.logger import logger  # App logger (already configured)
from .GIA import launch_app


def _apply_debug_runtime_switch() -> None:
    """Ensure logger + env reflect a late DEBUG override."""
    # WHAT: RAISE LOGGING VERBOSITY GLOBALLY; WHY: CONFIG["DEBUG"] CHANGED POST-IMPORT
    # SET APP LOGGER
    logger.setLevel(10)  # logging.DEBUG == 10

    # SET ROOT + COMMON THIRD-PARTY NOISY LOGGERS TO RESPECT DEBUG (OPTIONAL TUNE)
    import logging

    logging.getLogger().setLevel(10)
    for name in (
        "app",
        "gptqmodel",
        "uvicorn",
        "httpx",
        "httpcore",
        "urllib3.connectionpool",
        "gradio",
    ):
        logging.getLogger(name).setLevel(10)

    # BYTECODE OFF IN DEBUG (helpful during dev)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def main() -> None:
    """Launch the GIA application (with optional CLI-only mode and debug override)."""
    # ENSURE UNBUFFERED STDOUT/ERR (STABLE IN WSL/THREADS)
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = argparse.ArgumentParser(description="Launch the GIA application")
    parser.add_argument(
        "--cli", action="store_true", help="Run in CLI-only mode without Gradio UI"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Override DEBUG mode to True"
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: Ignoring unknown arguments: {unknown}")

    # APPLY --debug BEFORE USING CONFIG["DEBUG"] FOR ANY BEHAVIOR
    if args.debug:
        CONFIG["DEBUG"] = True
        _apply_debug_runtime_switch()
    else:
        # HONOR EXISTING CONFIG: if already True, disable bytecode to mirror previous behavior
        if CONFIG.get("DEBUG"):
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    logger.info(f"Launching with args: {args}")
    launch_app(args)


if __name__ == "__main__":
    main()
