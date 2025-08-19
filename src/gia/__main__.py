"""Entry point for GIA."""
import argparse
import os
from .GIA import launch_app
from gia.config import CONFIG  # TO ACCESS DEBUG SETTING

def main() -> None:
    """Launch the GIA application with optional CLI-only mode and bytecode cache control."""
    # TO SET UNBUFFERED OUTPUT FOR RELIABLE PRINTS IN THREADED WSL ENV
    os.environ["PYTHONUNBUFFERED"] = "1"

    # TO DISABLE BYTECODE CACHE WHEN DEBUG IS TRUE
    if CONFIG["DEBUG"]:
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    parser = argparse.ArgumentParser(description="Launch GIA application")
    parser.add_argument('--cli', action='store_true', help="Run in CLI-only mode without Gradio UI")
    # TO IGNORE UNKNOWN ARGS TO PREVENT EARLY EXIT
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: Ignoring unknown arguments: {unknown}")
    launch_app(args)

if __name__ == "__main__":
    main()