# **GIA: General Intelligence Assistant**

GIA is a local-first, extensible AI assistant. It’s designed for people who want a smart chatbot that runs on their own hardware (if you can) its also possible to use openai and hugginfaces, xai api will be added soon, use it with your own data, and is easy to extend with plugins. This project is for developers, power users, and anyone who wants to use their own model or any .safetensors locally with your own documents. Quantized GPTQModel / [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel). you can also build your own plugins easily.

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
- Commands starting with a dot (like `.example`) call plugins. Everything else is a chat or query.
- Plugins run in their own threads, so they can’t crash the main app.
- All data stays local unless you add a plugin that does otherwise.

### Typical Workflow

1. Enter the path where your models (LLM and embedding model) are located in the UI.
2. Scan the directory to list available models.
3. Confirm model selection. No manual config or coding required.
4. Create a database with one click (“Create Database”).  
   _Tip: After creating the database, close and restart GIA for best results._
5. Next time you start GIA, load your chosen model and then load the database to use the query engine.
6. Loaded plugins will show at the bottom of the Gradio UI. You can launch plugins by clicking their name.


## Creating Plugins

You can add new features to GIA by writing a plugin. Here’s how:

1. Create a folder for your plugin, e.g.:
   ```bash
   mkdir -p src/gia/plugins/myplugin/
   ```
2. Add a Python file named after your plugin, e.g.:
   ```bash
   touch src/gia/plugins/myplugin/myplugin.py
   ```
3. Define a function with the same name as your plugin folder. The function signature should be:
   ```python
   def myplugin(llm, query_engine, embed_model, chat_history, arg=None):
       ...
   ```
   - `llm`: The loaded language model instance.
   - `query_engine`: The retrieval/query engine (if available).
   - `embed_model`: The embedding model instance.
   - `chat_history`: List of previous chat messages.
   - `arg`: (Optional) Any argument passed from the UI/CLI.
   - The function should yield strings to send messages to the UI/CLI.

**Example Plugin:**
```python
from gia.core.logger import logger, log_banner
from gia.core import generate

log_banner(__file__)

def myplugin(llm, query_engine, embed_model, chat_history, arg=None):
    yield "MyPlugin started."
    try:
        prompt = "What's the weather like on Mars?"
        system_prompt = "You are a helpful assistant."
        response = generate(prompt, system_prompt, llm, max_new_tokens=256)
        yield f"LLM says: {response}"
    except Exception as e:
        logger.error(f"Error in myplugin: {e}")
        yield f"Error: {str(e)}"
```
- Edit your plugin file and GIA will pick up changes automatically.
- Only load plugins you trust. Plugins run as Python code and have access to your system.

This is a personal project, but suggestions and bug reports are welcome. If you have an idea or find a problem, open an issue.

---

For more details, see the code and comments. If you get stuck, just ask.