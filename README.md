# GIA
It runs on your own hardware with local models or connects to APIs like OpenAI HuggingFace and OpenRouter with dynamic model selection. You can work with your own data build custom plugins run quantized models via GPTQModel and load `.safetensors` models directly through the Gradio UI no coding required.

> ### Chat with AI in your browser or terminal, using local or online models, your data, and your own plugins.

- Web chat UI (Gradio) and terminal interface, always in sync.  
- Load models stored on your machine, or connect to APIs like OpenAI, Hugging Face, or OpenRouter.  
- Give it files (PDFs, text) — it uses them for answers.  
- Add your own tools with simple Python plugins.  

## Features

- **Models**: Local GPTQ (via GPTQModel) or remote (HuggingFace Inference, OpenAI, OpenRouter).
- **RAG Support**: Use your .pdf files or .txt for context-aware responses.
- **UI**: Gradio chat; optional CLI mode mirrors chat behavior.
- **Plugins**: Drop a file in `src/gia/plugins/<name>/<name>.py` and call it via `.<name>`.
- **Keys**: Read from environment variables only.
- **Synced Interfaces**: Use CLI or web UI interchangeably—sessions stay in sync.
- **Resource Management**: Optimized for long-running operations without excessive resource consumption.

![UI Screenshot](https://sicorps.com/content/4039-9696-294ba61e2a9f.png)

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

   - **Using pip**:
     ```
      python -m pip install -r requirements.lock
      python -m pip install -e . --no-deps
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

- Commands starting with a dot (e.g., `.myplugins_example`) trigger plugins; all other inputs are treated as queries to the LLM.
- Plugins execute in separate threads to prevent crashes from affecting the main application.
- Data remains local by default, except when using online APIs or plugins that explicitly send data externally.
- **OpenAI**: Uses OPENAI_API_KEY for authentication; supports dynamic selection from all available models and api_base for custom endpoints.
- **HuggingFace**: Uses HF_TOKEN for authentication; provides serverless inference without local downloads, select an explicit provider or let it automatically try a curated list.
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
   mkdir -p src/gia/plugins/myplugins_example/
   ```

2. **Add File**:
   ```
   touch src/gia/plugins/myplugins_example/myplugins_example.py
   ```

3. **Define Function**: The function name must match the folder name and accept a `ProjectState` snapshot (this is enforced by the loader at runtime).
   ```python
   from typing import Dict, List, Optional
   from gia.core.state import ProjectState

   def myplugins_example(
       state: ProjectState,
       chat_history: Optional[List[Dict[str, str]]] = None,
       arg: Optional[str] = None,
   ) -> str:
       ...
   ```
   - **Arguments**:
     - `state`: read-only access to handles such as `state.LLM`, `state.QUERY_ENGINE`, and `state.EMBED_MODEL`.
     - `chat_history`: list of prior chat messages (optional).
     - `arg`: optional argument from UI/CLI (e.g., `.myplugins_example topic`).
   - **Output**: Return a string for simple results, or `yield` chunks (strings or `{role, content, metadata}` dicts) to stream progress back into the UI.

See `src/gia/plugins/README.md` for a deeper dive into plugin architecture, streaming patterns, and development tips.

**Example Plugin**:
```python
from typing import Dict, List, Optional
from gia.config import system_prefix
from gia.core.utils import generate
from gia.core.logger import log_banner
from gia.core.state import ProjectState

log_banner(__file__)

def myplugins_example(
    state: ProjectState,
    chat_history: Optional[List[Dict[str, str]]] = None,
    arg: Optional[str] = None,
):
    llm = state.LLM
    if llm is None:
        yield "Error: Load a model first (.load)."
        return

    topic = arg or "quick status update"
    yield f"Starting run for: {topic}"

    response = generate(
        query=f"Summarize: {topic}",
        system_prompt=system_prefix(),
        llm=llm,
        max_new_tokens=256,
    )
    yield response
```

- Edit the file as needed; changes are auto-detected on reload.
- **Warning**: Only load plugins from trusted sources, as they execute with full system access.
