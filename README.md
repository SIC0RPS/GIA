# GIA: General Intelligence Assistant

GIA is a local-first, extensible AI assistant. It’s designed for people who want a smart chatbot that runs on their own hardware, can use their own data, and is easy to extend with plugins. This project is for developers, power users, and anyone who wants to build or customize their own AI tools.

## What is GIA?
GIA is a Python application that combines large language models (LLMs), retrieval-augmented generation (RAG), and a plugin system. It supports both a command-line interface and a web UI (Gradio). All data and models stay on your machine unless you add a plugin that does otherwise.

## Features
- Local LLM inference (no cloud required)
- Retrieval-augmented generation using your own files and databases
- Plugin system: add new features by dropping in a Python file
- CLI and web UI, always in sync
- Efficient resource management for long-running use

## Quick Start

1. **Install Python 3.11+** (and CUDA if you want GPU support).
2. **Install GIA**:
   ```bash
   pip install -e . --no-deps --no-build-isolation
   ```
3. **Check the install**:
   ```bash
   pip show gia
   pip check
   ```
4. **Run GIA**:
   ```bash
   python -m gia
   ```
   The CLI will start. The web UI is at http://localhost:7860.

## How it Works

- On startup, GIA loads your model, database, and plugins.
- You can chat via CLI or web UI. Both interfaces are always in sync.
- Commands starting with a dot (like `.genmaster`) call plugins. Everything else is a chat or query.
- Plugins run in their own threads, so they can’t crash the main app.
- All data stays local unless you add a plugin that changes that.

### Typical Workflow

1. **Enter the path** where your models (LLM and embedding model) are located in the UI.
2. **Scan the directory** to list available models.
3. **Confirm model** selection. No manual config or coding required.
4. **Create a database** with one click (“Create Database”).  
   *Tip: After creating the database, close and restart GIA for best results.*
5. **Next time you start GIA**, load your chosen model and then load the database to use the query engine.
6. **Loaded plugins** will show at the bottom of the Gradio UI.  
   You can launch plugins by clicking their name (no arguments needed for now).

## Creating Plugins

You can add new features to GIA by writing a plugin. Here’s how:

1. Create a folder: `src/gia/plugins/myplugin/`
2. Add a Python file: `src/gia/plugins/myplugin/myplugin.py`
3. Write a function named after your plugin (e.g., `def myplugin(...)`).
   - The function can take arguments like `llm`, `query_engine`, `embed_model`, and `chat_history`.
   - Yield strings to send messages to the UI/CLI.

Example:
```python
from gia.core.logger import logger, log_banner
log_banner(__file__)
from gia.core.utils import generate

def myplugin(llm, query_engine, embed_model, chat_history, arg=None):
    yield "MyPlugin started."
    try:
        prompt = "What's the weather like on Mars?"
        response = generate(prompt, "System prompt", llm)
        yield f"LLM says: {response}"
    except Exception as e:
        logger.error(f"Error in myplugin: {e}")
        yield f"Error: {str(e)}"
```
- Edit your plugin file and GIA will pick up changes automatically.
- Only load plugins you trust. Plugins run as Python code.

## Project Structure
- `src/gia/GIA.py`: Main app (CLI, UI, plugin loader)
- `src/gia/core/`: Core logic (state, utils, logging, etc.)
- `src/gia/plugins/`: Drop-in plugins (each in its own folder)
- `src/gia/db/`, `src/gia/datasets/`: Your data

## Philosophy
- Local-first: your data, your models, your rules
- Extensible: add features with a single Python file
- Transparent: no black boxes—read and change the code

## Contributing
This is a personal project, but suggestions and bug reports are welcome. If you have an idea or find a problem, open an issue.

---

For more details, see the code and comments. If you get stuck, just ask.