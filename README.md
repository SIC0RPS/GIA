# GIA: General Intelligence Assistant

GIA is a local-first, extensible AI assistant designed for developers and power users. You can run it on your own hardware using local models or connect to online APIs like OpenAI (with dynamic model selection) and HuggingFace providers. It allows you to use your own data, build custom plugins, and run quantized models via GPTQModel.

## What is GIA?

GIA is a Python-based tool that combines large language models (LLMs), Retrieval-Augmented Generation (RAG), and plugins. It supports both a command-line interface (CLI) and a Gradio web UI. Your data remains local unless you choose to use online APIs or custom plugins that require external access. For security, API keys are managed exclusively through environment variables.

## Features

- **LLM Inference Options**: Run models locally (using GPTQ for pre-quantized models) or online via HuggingFace (serverless with explicit providers or automatic curated order), OpenAI (all models supported with api_base), or OpenRouter.
- **RAG Support**: Integrate your own files and databases for context-aware responses.
- **Plugin System**: Easily add new features by creating Python files.
- **Synced Interfaces**: Use CLI or web UI interchangeably—sessions stay in sync.
- **Resource Management**: Optimized for long-running operations without excessive resource consumption.

## Quick Start

Follow these steps to install and run GIA.

1. **Create Environment**

   Choose one option. If Conda is not installed, follow the official installation guides: [Windows](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html), [Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html), [macOS](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html).

   - **Option A: Conda**
     ```
     conda create -n GIA python=3.11 -y
     conda activate GIA
     ```

   - **Option B: venv**
     ```
     python3.11 -m venv .venv && source .venv/bin/activate
     ```

2. **Install PyTorch (CUDA 12.8 or CPU)**

   ```
   python -m pip install --upgrade pip
   python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.7.1+cu128"
   # or for CPU only:
   python -m pip install torch==2.7.1
   ```

3. **Install Dependencies + GIA**

   Choose one method (pip or uv) to install dependencies from requirements.lock and set up GIA in editable mode. If using uv and it's not installed, run `pip install uv` (see [PyPI page](https://pypi.org/project/uv/) for details).

   - **Using pip**:
     ```
     pip install -r requirements.lock
     pip install -e .
     ```

   - **Using uv** (faster alternative):
     ```
     pip install uv
     uv pip sync requirements.lock
     pip install -e .
     ```

4. **Verify + Run**

   ```
   pip show gia
   pip check
   python -m gia
   ```

   Access the web UI at [http://localhost:7860](http://localhost:7860).

   Note: If cloning models (e.g., for embeddings) requires Git and it's not installed, follow the [official Git installation guide](https://github.com/git-guides/install-git).

## Setting API Keys Securely

To use online models, set API keys as environment variables. This avoids hardcoding sensitive information in files, reducing security risks. Set them temporarily for a session or persistently for ongoing use. Always restart your terminal or application after changes.

### Linux

- **Temporary (current session)**:
  ```
  export HF_TOKEN="your_token_here"
  export OPENAI_API_KEY="your_key_here"
  export OPENROUTER_API_KEY="your_key_here"
  ```

- **Persistent (user-level)**:
  Add the following lines to `~/.bashrc` or `~/.profile`:
  ```
  export HF_TOKEN="your_token_here"
  export OPENAI_API_KEY="your_key_here"
  export OPENROUTER_API_KEY="your_key_here"
  ```
  Then run:
  ```
  source ~/.bashrc
  ```

  For system-wide access (requires sudo): Edit `/etc/environment` and reboot.

### Windows

- **Temporary (current Command Prompt session)**:
  ```
  set HF_TOKEN="your_token_here"
  set OPENAI_API_KEY="your_key_here"
  set OPENROUTER_API_KEY="your_key_here"
  ```

- **Persistent (user or system)**:
  Search for "Environment Variables" in the Start menu, then select "Edit the system environment variables" > "Environment Variables" button.
  - Under "User variables" (for your user) or "System variables" (for all users), click "New".
  - Enter the variable name (e.g., HF_TOKEN) and value (your_token_here).
  - Click OK to save. Restart your Command Prompt or application.

In PowerShell (temporary):
```
$env:HF_TOKEN="your_token_here"
$env:OPENAI_API_KEY="your_key_here"
$env:OPENROUTER_API_KEY="your_key_here"
```

For persistent PowerShell setup, edit your profile file (`$PROFILE`).

## How it Works

GIA starts by loading your selected model (local or online), any RAG database, and plugins. You can interact via CLI or UI, with both interfaces staying synchronized.

- Commands starting with a dot (e.g., `.myplugin`) trigger plugins; all other inputs are treated as queries to the LLM.
- Plugins execute in separate threads to prevent crashes from affecting the main application.
- Data remains local by default, except when using online APIs or plugins that explicitly send data externally.
- **OpenAI**: Uses OPENAI_API_KEY for authentication; supports dynamic selection from all available models and api_base for custom endpoints.
- **HuggingFace**: Uses HF_TOKEN for authentication; provides serverless inference without local downloads—select an explicit provider or let it automatically try a curated list.
- **Local Models**: Employs GPTQModel for pre-quantized models, wrapped as a HuggingFaceLLM.
- **Security**: API keys are read only from environment variables (HF_TOKEN, OPENAI_API_KEY, OPENROUTER_API_KEY); no logging of secrets.
- **Database**: For RAG, place text files in the MyData folder, then use the `.create` command in chat (or upcoming UI button) to build a ChromaDB database.

  **Recommended Embedding Models**
For optimal performance with ChromaDB persistent databases, use one of these HuggingFace models (clone via Git for local use):
  - BAAI/bge-large-en-v1.5: Higher accuracy, suitable for detailed queries. [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5). 

    ```
    git clone https://huggingface.co/BAAI/bge-large-en-v1.5
    ```

  - BAAI/bge-small-en-v1.5: More efficient for resource-constrained setups. [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).

    ```
    git clone https://huggingface.co/BAAI/bge-small-en-v1.5
    ```

### Typical Workflow

1. **Select Mode in UI**: Choose Local, HuggingFace Online, OpenAI, or OpenRouter.
   - **Local**: Enter the model directory path; GIA scans and lets you select a folder.
   - **HuggingFace**: Set HF_TOKEN, then choose a model from the dropdown.
   - **OpenAI**: Set OPENAI_API_KEY, then select from the dynamic list of all available models in the dropdown.
   - **OpenRouter**: Set OPENROUTER_API_KEY, then enter the model name (e.g., "x-ai/grok-4") in the textbox.

2. **Confirm Model**: No additional configuration files are required.

3. **Create Database**: Add files to the MyData folder, then use the "Create Database" button (or `.create` in chat) to generate the ChromaDB.
   - Tip: Restart GIA after creation to load the new database.

4. **Query and Interact**: On restart, the model and database load automatically for use in queries.

5. **Use Plugins**: View available plugins in the UI and click to run them.

## Creating Plugins

Extend GIA by adding custom plugins:

1. **Create Folder**:
   ```
   mkdir -p src/gia/plugins/myplugin/
   ```

2. **Add File**:
   ```
   touch src/gia/plugins/myplugin/myplugin.py
   ```

3. **Define Function**: The function name must match the folder name.
   ```python
   def myplugin(llm, query_engine, embed_model, chat_history, arg=None):
       ...
   ```
   - **Arguments**:
     - `llm`: The loaded model.
     - `query_engine`: RAG engine (if a database is loaded).
     - `embed_model`: Embedding model for RAG.
     - `chat_history`: List of previous messages.
     - `arg`: Optional argument from UI/CLI.
   - **Output**: Use `yield` to send strings back to the UI/CLI.

**Example Plugin**:
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

- Edit the file as needed; changes are auto-detected on reload.
- **Warning**: Only load plugins from trusted sources, as they execute with full system access.