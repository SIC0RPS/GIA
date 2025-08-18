# src/gia/__main__.py
"""Entry point for GIA"""
# TO IMPORT MAIN ENTRY POINT
from .GIA import launch_app
import argparse
import os

def main() -> None:
    """Launch the gia application with optional CLI-only mode."""
    # TO SET UNBUFFERED OUTPUT FOR RELIABLE PRINTS IN THREADED WSL ENV
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = argparse.ArgumentParser(description="Launch GIA application")
    parser.add_argument('--cli', action='store_true', help="Run in CLI-only mode without Gradio UI")
    args = parser.parse_args()
    launch_app(args)

# TO RUN APPLICATION WHEN EXECUTED AS python -m gia
if __name__ == "__main__":
    main()