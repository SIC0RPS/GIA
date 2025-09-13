# GIA Plugin System (Quick Guide)

## Placement
- Path: `src/gia/plugins/<name>/<name>.py`
- Function name = folder name = file name

## Loading
- Auto-loaded at startup by `load_plugins()`
- Run with `.name` in CLI or click in UI

## Function Signature
```python
def myplugin(llm, query_engine, embed_model, chat_history, arg=None):
    ...
````

## Core Usage

* Use `generate()` for LLM calls
* Use `yield` to stream output
* Catch exceptions → `yield "Error: ..."`

## State Variables

* `llm` → current model (`state.LLM`)
* `query_engine` → RAG engine (`state.QUERY_ENGINE`)
* `embed_model` → embedding model
* `chat_history` → full conversation list

## Example

```python
from gia.core import generate
from gia.core.logger import logger, log_banner

log_banner(__file__)

def myplugin(llm, query_engine, embed_model, chat_history, arg=None):
    yield "MyPlugin started."
    try:
        prompt = arg or "What's the weather on Mars?"
        resp = generate(prompt, "You are a helpful assistant.", llm, max_new_tokens=100)
        yield f"LLM says: {resp}"
    except Exception as e:
        logger.error(f"Error in myplugin: {e}")
        yield f"Error: {str(e)}"
```

## Key Points
- Correct path and naming are mandatory  
- Plugins run in isolated threads (safe)  
- UI auto-lists plugins (clickable)  
- `arg` is optional string passed by user  
# GIA Plugin System — Quick Guide

## Placement & Naming
- Put each plugin in `src/gia/plugins/<name>/<name>.py`.
- The exported function must be named exactly `<name>`.
- Example: `src/gia/plugins/minidemo/minidemo.py` exports `def minidemo(...):`.

## Loading & Running
- GIA auto-discovers plugins at startup; no manual registration required.
- Run via chat with dot-commands (e.g., `.minidemo topic here`) or click in the UI list.

## Standard Signature
```python
from typing import List, Dict, Optional, Generator
from gia.core.state import ProjectState

def myplugin(
    state: ProjectState,
    chat_history: Optional[List[Dict[str, str]]] = None,
    arg: Optional[str] = None,
) -> Generator[List[Dict[str, str]], None, None] | str:
    ...
```
- `state`: read-only snapshot with `LLM`, `QUERY_ENGINE`, `EMBED_MODEL`, `MODEL_NAME`, etc.
- `chat_history`: the full session history (list of `{role, content, metadata}` dicts).
- `arg`: optional single CLI argument string passed by the user.
- Return a generator and `yield` message batches to stream into the UI, or return a single `str`.

## Message Shape for Yields
Yield lists of chat messages compatible with `gr.Chatbot(type="messages")`:
```python
[{"role": "assistant", "content": "text", "metadata": {"plugin": "myplugin"}}]
```
- Roles: `assistant`, `user`, or `system`.
- Metadata is optional; include `{"plugin": "<name>"}` if helpful.

## Minimal Streaming Example
```python
from typing import List, Dict, Optional, Generator
from gia.core.state import ProjectState
from gia.core.logger import logger, log_banner

log_banner(__file__)

def minidemo(
    state: ProjectState,
    chat_history: Optional[List[Dict[str, str]]] = None,
    arg: Optional[str] = None,
) -> Generator[List[Dict[str, str]], None, None]:
    yield [{"role": "assistant", "content": "Starting demo…", "metadata": {"plugin": "minidemo"}}]
    for i in range(3):
        yield [{"role": "assistant", "content": f"Chunk {i+1}/3", "metadata": {"plugin": "minidemo"}}]
```

## One‑Shot Example Using `generate`
```python
from typing import Optional
from gia.core.state import ProjectState
from gia.core.utils import generate
from gia.config import system_prefix

def oneshot(state: ProjectState, chat_history=None, arg: Optional[str] = None) -> str:
    llm = state.LLM
    if llm is None:
        return "Error: Load a model first (.load)."
    user = arg or "Explain diffusion models simply."
    system = system_prefix()
    return generate(query=user, system_prompt=system, llm=llm, max_new_tokens=512, think=False)
```

## Logger Integration
```python
from gia.core.logger import logger, log_banner
log_banner(__file__)
logger.info("Plugin loaded")
logger.debug("Details…")
logger.error("Something went wrong")
```
All plugin logs go to the main application log pipeline.

## Safe State Usage
- Access `state.*` inside your function only; do not reference `state` at import time.
- Use `state.LLM`, `state.QUERY_ENGINE`, `state.EMBED_MODEL` read‑only.
- For cross‑plugin flags, use `state_manager.get_state/set_state` sparingly; avoid mutating core handles.

## System Prompts
- Global default comes from `config.toml` `[prompt].system_prompt` (string or list of strings).
- Use `from gia.config import system_prefix` to fetch the default at runtime.
- You can override per‑call via helper function parameters (see below).

## Generation Helpers (utils.py)
Two helper functions manage token budgets, device safety, and provider quirks.

```python
from gia.core.utils import generate, stream_generate
from gia.config import system_prefix

system = system_prefix()  # default system prompt from config.toml

# Non‑streaming (retries on OOM by halving budget)
text = generate(
    query="Your question here",
    system_prompt=system,
    llm=state.LLM,
    max_new_tokens=512,      # optional; defaults to CONFIG["MAX_NEW_TOKENS"]
    think=False,             # Qwen3/CyberSic: set True to add /think, False adds /no_think
)

# Stream‑first (falls back to non‑stream if needed), returns final text
text_streamed = stream_generate(
    query="Stream a short poem",
    system_prompt=system,
    llm=state.LLM,
    max_new_tokens=256,
    think=False,
)
```

Notes:
- `max_new_tokens` is respected per provider; helpers sanitize wrapper kwargs and restore caps.
- `think=True` automatically adds a control tag for Qwen3/CyberSic models.

## Filtered RAG Query Engine
Use `filtered_query_engine` to query a category‑filtered vector index. Your nodes must have `{"category": "<value>"}` metadata at ingestion time.

```python
from gia.core.utils import filtered_query_engine
from gia.config import system_prefix

def rag_demo(state: ProjectState, chat_history=None, arg=None):
    llm = state.LLM
    if llm is None or state.QUERY_ENGINE is None:
        return "Error: Load a model (.load) and database (.load) first."

    category = "python"  # must match node metadata
    user_query = arg or "Explain list comprehensions."
    system = system_prefix()

    qe = filtered_query_engine(
        llm=llm,
        query_str=user_query,
        category=category,
        system_prompt=system,  # optional; prepends to QA template
    )
    resp = qe.query(user_query)
    return getattr(resp, "response", str(resp))
```

To ingest with categories, call `update_database(filepaths, query, metadata={"category": "python", "tags": ["..."]})`.

## Yielding vs Returning
- For streaming UI updates, implement a generator and `yield` lists of messages.
- For simple, immediate results, return a single `str`.
- GIA normalizes plugin outputs and displays them in the chat.

## Security & Best Practices
- Treat `arg` as untrusted input; validate before use.
- Prefer pure‑Python logic; avoid `exec`/`eval` and unsafe shell calls.
- Catch exceptions and return/readably yield error messages.
- Keep outputs concise and incremental to maintain a responsive UI.

## Troubleshooting
- Plugin not found: folder/file/function names must match (`plugins/<name>/<name>.py` with `def <name>(...)`).
- No output in UI: ensure you return a `str` or yield lists of message dicts.
- “Error: No model loaded”: run `.load` to initialize an LLM first.
- Filtered RAG returns nothing: verify your DB nodes contain `{"category": "..."}` metadata that matches your filter.

## Minimal Template (Copy/Paste)
```python
# src/gia/plugins/_template/_template.py
from typing import List, Dict, Optional, Generator
from gia.core.state import ProjectState
from gia.core.utils import generate
from gia.core.logger import logger, log_banner
from gia.config import system_prefix

log_banner(__file__)

def _template(
    state: ProjectState,
    chat_history: Optional[List[Dict[str, str]]] = None,
    arg: Optional[str] = None,
) -> Generator[List[Dict[str, str]], None, None]:
    llm = state.LLM
    if llm is None:
        yield [{"role": "assistant", "content": "Load a model first (.load)"}]
        return

    topic = (arg or "Hello from a GIA plugin").strip()
    system = system_prefix()

    yield [{"role": "assistant", "content": f"Generating about: {topic}"}]
    text = generate(query=topic, system_prompt=system, llm=llm, max_new_tokens=256, think=False)
    yield [{"role": "assistant", "content": text, "metadata": {"plugin": "_template"}}]
```

