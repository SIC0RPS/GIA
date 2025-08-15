# **GIA: General Intelligence Assistant**

GIA is a local first extensible AI assistant and not just another wrapper but a platform for anyone to experiment with AI. It can run on your own hardware or connect to OpenAI, HuggingFace, and soon XAI APIs. It works with your own data, supports local models including any .safetensors, and can run quantized GPTQ models for faster performance. Built for developers, power users, and anyone who wants full control. Plugins are simple to create.


## What is GIA?
GIA is a Python application that combines large language models (LLMs), retrieval-augmented generation (RAG), and a plugin system. It supports both a command-line interface and a web UI (Gradio).

## Features
- Local LLM inference (no cloud required)
- Retrieval-augmented generation using your own files and databases
- Plugin system: add new features by dropping in a Python file
- CLI and web UI, always in sync
- Efficient resource management for long-running use
- HuggingFace Online: enter any valid model name manually or select from dropdown
- OpenAI: dynamic dropdown of valid models from API (no manual entry)

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
4. For HuggingFace Online: you can enter any valid model name manually or pick from the dropdown.
5. For OpenAI: select from the dynamic dropdown (models are fetched from the API).
6. Create a database with one click (“Create Database”).  
   _Tip: After creating the database, close and restart GIA for best results._
7. Next time you start GIA, load your chosen model and then load the database to use the query engine.
8. If database is not loaded, LLM will still answer directly (no retrieval).
9. Loaded plugins will show at the bottom of the Gradio UI. You can launch plugins by clicking their name.


## Creating Plugins

Add new features by writing a plugin. Anyone can do it:

1. Create a folder for your plugin:
   ```bash
   mkdir -p src/gia/plugins/my_plugins/
   ```
2. Add a Python file with the same name:
   ```bash
   touch src/gia/plugins/my_plugins/my_plugins.py
   ```
3. Define a function with the same name as the folder/file:
   ```python
   def my_plugins(llm, query_engine, embed_model, chat_history, arg=None):
       yield "Plugin started."
       # Your logic here
   ```

**How plugins work:**
- Function name must match folder and file name.
- Arguments:
  - `llm`: Loaded language model instance
  - `query_engine`: Retrieval/query engine (if available)
  - `embed_model`: Embedding model instance
  - `chat_history`: List of previous chat messages
  - `arg`: (Optional) Argument from UI/CLI
- Yield strings to send output to UI/CLI (can yield multiple times for streaming)
- Plugins are hot-reloaded, edit and rerun, no restart needed
- Only load trusted code (plugins run as Python)

**Minimal working example:**
```python
def my_plugins(llm, query_engine, embed_model, chat_history, arg=None):
    if arg:
        yield f"You passed: {arg}"
    else:
        yield "No argument provided."
```

**Example with LLM call:**
```python
from gia.core import generate

def my_plugins(llm, query_engine, embed_model, chat_history, arg=None):
    prompt = arg or "Say hello."
    system_prompt = "You are a helpful assistant."
    response = generate(prompt, system_prompt, llm, max_new_tokens=128)
    yield f"LLM says: {response}"
```

- Edit your plugin file and GIA will pick up changes automatically.
- Plugins run in their own threads; errors do not crash the main app.

---

For more details, see the code and comments.