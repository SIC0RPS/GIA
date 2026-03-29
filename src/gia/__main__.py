# src/gia/__main__.py
"""Entry point for launching the GIA application."""
import argparse
import os
import sys

def _apply_debug_runtime_switch(logger) -> None:
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
    parser.add_argument(
        "--mode",
        choices=["Local", "HuggingFace", "OpenAI", "OpenRouter"],
        help="Select the model backend for CLI usage.",
    )
    parser.add_argument(
        "--model",
        help="Model identifier or local model path, depending on --mode.",
    )
    parser.add_argument(
        "--prompt",
        help="Run one non-streaming prompt and exit.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run one non-streaming prompt from stdin when --prompt is not provided.",
    )
    parser.add_argument(
        "--system-prompt",
        help="Override the configured system prompt for one-shot CLI runs.",
    )
    parser.add_argument(
        "--load-db",
        action="store_true",
        help="Load the configured database before running the CLI model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override generation token budget for one-shot CLI runs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override generation temperature for one-shot CLI runs.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Override nucleus sampling top-p for one-shot CLI runs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Override top-k sampling for one-shot CLI runs.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        help="Override repetition penalty for one-shot CLI runs.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        help="Override no-repeat ngram size for one-shot CLI runs.",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable /think control tag for compatible local models in one-shot CLI runs.",
    )
    parser.add_argument(
        "--gen-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Pass an extra generation kwarg through to generate(). Repeat as needed.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: Ignoring unknown arguments: {unknown}")

    wants_model = bool(args.mode or args.model)
    wants_one_shot = bool(args.prompt is not None) or bool(args.run)
    if wants_model and not (args.mode and args.model):
        parser.error("--mode and --model must be provided together.")
    if wants_one_shot and not wants_model:
        parser.error("One-shot CLI usage requires both --mode and --model.")
    if args.cli:
        # Disable external debug terminal viewers for CLI runs unless explicitly re-enabled.
        os.environ.setdefault("GIA_OPEN_LOG_TERMINAL", "0")
    if args.run and args.prompt is None:
        args.stdin_prompt = sys.stdin.read()
    else:
        args.stdin_prompt = None

    # NOTE: Delay app imports until after CLI parsing/stdin capture so one-shot
    # runs can consume piped input before the full GIA stack loads.
    from gia.config import CONFIG  # ACCESS DEBUG SETTING
    from gia.core.logger import logger  # App logger (already configured)
    from .GIA import launch_app

    # APPLY --debug BEFORE USING CONFIG["DEBUG"] FOR ANY BEHAVIOR
    if args.debug:
        CONFIG["DEBUG"] = True
        _apply_debug_runtime_switch(logger)
    else:
        # HONOR EXISTING CONFIG: if already True, disable bytecode to mirror previous behavior
        if CONFIG.get("DEBUG"):
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    logger.info(f"Launching with args: {args}")
    launch_app(args)


if __name__ == "__main__":
    main()
