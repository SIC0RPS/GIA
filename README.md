# **GIA: General Intelligence Assistant**

GIA is a local-first, extensible AI assistant. Run it on your hardware with local models, or use online APIs like OpenAI (dynamic model selection from list) and HuggingFace Providers. Use your own data, build plugins, and run quantized models via GPTQModel. For developers and power users.

## What is GIA?
GIA combines LLMs, RAG, and plugins in Python. Supports CLI and Gradio web UI. Data stays local unless using online APIs or custom plugins. API keys handled via environment variables for security.

## Features
- Local or online LLM inference: Local (GPTQ pre-quantized), HuggingFace Online (serverless with explicit provider; auto tries curated order), OpenAI (all models with api_base support)
- RAG with your files and databases
- Plugin system: add features via Python files
- CLI and web UI in sync
- Resource management for sustained operation

## Quick Start

1. **Create environment**

Option A: Conda

```bash
conda create -n GIA python=3.11 -y
conda activate GIA
```

Option B: venv

```bash
python3.11 -m venv .venv && source .venv/bin/activate
```

2. **Install PyTorch (CUDA 12.8 or CPU)**

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.7.1+cu128"
# or CPU only:
python -m pip install torch==2.7.1
```

3. **Install dependencies + GIA (pip)**

```bash
pip install -r requirements.lock
pip install -e .
```

4. **Install dependencies + GIA (uv)**

```bash
pip install uv
uv pip sync requirements.lock
pip install -e .
```

5. **Verify + Run**

```bash
pip show gia
pip check
python -m gia
```

Web UI: [http://localhost:7860](http://localhost:7860)


## Setting API Keys Securely

Use environment variables to store API keys. This prevents hardcoding secrets in files. Set them via terminal for temporary use or configuration files/system settings for persistence. Restart your terminal or application after setting.

### Linux
- Temporary (current session):
  ```bash
  export HF_TOKEN=your_token_here
  export OPENAI_API_KEY=your_key_here
  export OPENROUTER_API_KEY=your_key_here
  ```
- Persistent (user-level):
  Add lines to `~/.bashrc` or `~/.profile`:
  ```bash
  export HF_TOKEN=your_token_here
  export OPENAI_API_KEY=your_key_here
  export OPENROUTER_API_KEY=your_key_here
  ```
  Then run:
  ```bash
  source ~/.bashrc
  ```
  For system-wide (requires sudo): Edit `/etc/environment` and reboot.

### Windows
- Temporary (current Command Prompt session):
  ```cmd
  set HF_TOKEN=your_token_here
  set OPENAI_API_KEY=your_key_here
  set OPENROUTER_API_KEY=your_key_here
  ```
- Persistent (user or system):
  Search "Environment Variables" in Start menu > Edit the system environment variables > Environment Variables button.
  - Under "User variables" (user-specific) or "System variables" (all users), click New.
  - Variable name: e.g., HF_TOKEN
  - Variable value: your_token_here
  - OK to save. Restart Command Prompt or application.

In PowerShell (temporary):
```powershell
$env:HF_TOKEN="your_token_here"
$env:OPENAI_API_KEY="your_key_here"
$env:OPENROUTER_API_KEY="your_key_here"
```

For persistence in PowerShell, edit your profile file (`$PROFILE`).

## How it Works

- Loads model (local/online), database, plugins on start.
- Chat via CLI or UI. Interfaces sync.
- Dot commands (e.g., `.myplugin`) trigger plugins. Others are queries.
- Plugins run in threads to avoid crashes.
- Data local except for online APIs or plugins.
- OpenAI: Supports dynamic model selection from list via OPENAI_API_KEY env var; api_base passthrough.
- HuggingFace: Supports provider models via HF_TOKEN env var; serverless, no local downloads; explicit provider or auto-curated order.
- Local: Uses GPTQModel for pre-quantized models, wraps to HuggingFaceLLM; no bitsandbytes/auto_gptq.
- Security: Reads only from env vars HF_TOKEN, OPENAI_API_KEY; no secrets logged.
- Database: Use MyData folder for simpler ChromaDB creation. Drop text files there and run .create command in chat to create DB (button coming soon).

### Typical Workflow

1. Select mode in UI: Local, HuggingFace Online, OpenAI, or OpenRouter.
   - Local: Enter model directory, scan, select folder.
   - HuggingFace: Set HF_TOKEN env var for security, choose model from dropdown.
   - OpenAI: Set OPENAI_API_KEY env var for security, dynamically select from list of all available models in dropdown.
   - OpenRouter: Set OPENROUTER_API_KEY env var for security, enter model name in textbox.
2. Confirm model. No config files needed.
3. Create database with "Create Database" button or .create in chat after adding files to MyData.
   - Tip: Restart GIA after creation.
4. On restart, load model then database for queries.
5. Plugins listed in UI. Click to run.

## Creating Plugins

Add features with plugins:

1. Create folder:
   ```bash
   mkdir -p src/gia/plugins/myplugin/
   ```
2. Add file:
   ```bash
   touch src/gia/plugins/myplugin/myplugin.py
   ```
3. Define function matching folder name:
   ```python
   def myplugin(llm, query_engine, embed_model, chat_history, arg=None):
       ...
   ```
   - Args: llm (model), query_engine (RAG if loaded), embed_model, chat_history (messages), arg (optional from UI/CLI).
   - Yield strings for UI/CLI output.

**Example:**
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
- Edit file; changes auto-detected.
- Load only trusted plugins. They run as code with system access.

Personal project. Issues and suggestions welcome via GitHub.