"""Entry point for GIA"""
# TO IMPORT MAIN ENTRY POINT
from .GIA import launch_app

def main() -> None:
    """Launch the gia application."""
    # TO RUN THE GRADIO APPLICATION
    launch_app()

# TO RUN APPLICATION WHEN EXECUTED AS python -m gia
if __name__ == "__main__":
    main()