# GIA Plugin System (Quick Guide)

## Placement
- Path: `src/gia/plugins/<name>/<name>.py`
- Folder name, module name, and exported function must match (`myplugin/myplugin.py` → `def myplugin(...)`).

## Loading
- Auto-loaded at startup by `gia.GIA.load_plugins()`.
- Invoke from chat with `.name` or click the plugin button in the UI.

## Function Signature
```python
from typing import Dict, List, Optional
from gia.core.state import ProjectState

def myplugin(
    state: ProjectState,
    chat_history: Optional[List[Dict[str, str]]] = None,
    arg: Optional[str] = None,
):
    ...
```
- `state`: read-only snapshot with handles such as `state.LLM`, `state.QUERY_ENGINE`, `state.EMBED_MODEL`, `state.MODEL_NAME`.
- `chat_history`: list of prior messages (`[{"role": str, "content": str, "metadata": dict}]`) or `None`.
- `arg`: optional string captured from `.myplugin value`.

GIA binds these arguments by keyword. Missing or renamed parameters trigger a validation error before execution.

## Core Usage
- Call `generate()` (or `stream_generate()`) for LLM work.
- Use `yield` to stream progress; return a string/list for one-shot results.
- Catch exceptions and surface readable errors with `yield "Error: ..."` or `return "Error: ..."`.

## State Handles
Access runtime components through `state` only:
- `state.LLM`
- `state.QUERY_ENGINE`
- `state.EMBED_MODEL`
- `state.MODEL_NAME`
- `state.DATABASE_LOADED`

Treat them as read-only. Avoid storing the snapshot for reuse outside the function.

## Example
```python
from typing import Dict, List, Optional
from gia.core.logger import logger, log_banner
from gia.core.state import ProjectState
from gia.core.utils import generate
from gia.config import system_prefix

log_banner(__file__)

def myplugin(
    state: ProjectState,
    chat_history: Optional[List[Dict[str, str]]] = None,
    arg: Optional[str] = None,
):
    llm = state.LLM
    if llm is None:
        yield "Error: Load a model first (.load)."
        return

    topic = (arg or "What's the weather on Mars?").strip()
    yield f"Working on: {topic}"

    response = generate(
        query=topic,
        system_prompt=system_prefix(),
        llm=llm,
        max_new_tokens=256,
    )
    yield f"LLM says: {response}"

    logger.info("myplugin completed for topic=%s", topic)
```

## Key Points
- Path + naming must align or the loader skips the plugin.
- Plugins run inside sandbox threads; always stream status for long jobs.
- The UI lists available plugins automatically.
- `arg` is optional; validate it before use.

# GIA Plugin System — Developer Guide

## Placement & Naming
- Each plugin lives in `src/gia/plugins/<name>/`.
- Entry file: `<name>.py` with `def <name>(...)`.
- Optional helpers can live beside the entry file; they must not conflict with other plugin names.

## Loading & Running
- Plugins load when GIA starts or when `.reload` (if implemented) runs.
- Users trigger plugins via dot-commands (`.name optional-arg`).
- The loader inspects signatures with `inspect.signature(...).bind_partial(state=..., chat_history=..., arg=...)` and raises clear errors on mismatch.

## Standard Signature Recap
```python
from typing import Dict, List, Optional, Generator
from gia.core.state import ProjectState

def myplugin(
    state: ProjectState,
    chat_history: Optional[List[Dict[str, str]]] = None,
    arg: Optional[str] = None,
) -> Generator[List[Dict[str, str]], None, None] | str:
    ...
```
- Return a generator to stream (`yield` lists of messages), or return a single `str`/list/dict for immediate results.
- When yielding dicts, include at least `role` and `content`; the sandbox stamps missing metadata (`plugin`, `title`).

## Message Shape for Yields
```python
[
    {
        "role": "assistant",
        "content": "Chunk text",
        "metadata": {"plugin": "myplugin"}
    }
]
```
- Roles: `assistant`, `user`, `system`.
- `metadata` is optional; avoid non-serializable types.

## Minimal Streaming Example
```python
from typing import Dict, List, Optional, Generator
from gia.core.state import ProjectState
from gia.core.logger import log_banner

log_banner(__file__)

def minidemo(
    state: ProjectState,
    chat_history: Optional[List[Dict[str, str]]] = None,
    arg: Optional[str] = None,
) -> Generator[List[Dict[str, str]], None, None]:
    yield [{"role": "assistant", "content": "Starting demo…", "metadata": {"plugin": "minidemo"}}]
    for i in range(3):
        yield [{"role": "assistant", "content": f"Chunk {i + 1}/3", "metadata": {"plugin": "minidemo"}}]
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
All plugin logs flow into the main application log pipeline.

## Safe State Usage
- Touch `state.*` only inside the entrypoint; never reference it at import time.
- Treat `state.LLM`, `state.QUERY_ENGINE`, `state.EMBED_MODEL`, etc. as read-only.
- For cross-plugin coordination use `state_manager` sparingly; avoid mutating core handles.

## System Prompts
- Global default comes from `config.toml` `[prompt].system_prompt`.
- Call `system_prefix()` to fetch it at runtime.
- Override per call as needed.

## Generation Helpers (utils.py)
```python
from gia.core.utils import generate, stream_generate
from gia.config import system_prefix

system = system_prefix()

text = generate(
    query="Your question here",
    system_prompt=system,
    llm=state.LLM,
    max_new_tokens=512,
    think=False,
)

text_streamed = stream_generate(
    query="Stream a short poem",
    system_prompt=system,
    llm=state.LLM,
    max_new_tokens=256,
    think=False,
)
```
Notes:
- Helpers manage provider quirks and retry on OOM.
- `think=True` injects `/think` control tags for supported models; `False` uses `/no_think`.

## Filtered RAG Query Engine
```python
from gia.core.utils import filtered_query_engine
from gia.config import system_prefix

def rag_demo(state: ProjectState, chat_history=None, arg=None):
    llm = state.LLM
    if llm is None or state.QUERY_ENGINE is None:
        return "Error: Load a model (.load) and database (.load) first."

    category = "python"
    user_query = arg or "Explain list comprehensions."
    system = system_prefix()

    qe = filtered_query_engine(
        llm=llm,
        query_str=user_query,
        category=category,
        system_prompt=system,
    )
    resp = qe.query(user_query)
    return getattr(resp, "response", str(resp))
```
To ingest with categories, call `update_database(filepaths, query, metadata={"category": "python"})`.

## Yielding vs Returning
- Streaming UI updates → generator + `yield` (strings or message lists).
- Simple result → return a string/list/dict.
- The sandbox normalizes outputs and appends metadata when missing.

## Security & Best Practices
- Treat `arg` and file inputs as untrusted; validate before use.
- Prefer pure Python helpers; avoid `exec`, `eval`, and unsafe shell calls.
- Catch exceptions and surface readable messages to the user.
- Keep outputs concise to maintain responsive UI updates.

## Troubleshooting
- Plugin missing: folder/file/function names must match exactly.
- No output: ensure you yield lists or return a string.
- “Error: Load a model first” → run `.load` to initialize the LLM.
- Filtered RAG empty: confirm your database nodes have matching `category` metadata.

## Minimal Template (Copy/Paste)
```python
# src/gia/plugins/_template/_template.py
from typing import Dict, List, Optional, Generator
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

    logger.info("_template finished topic=%s", topic)
```

Stick to these conventions and your plugins will integrate cleanly with the GIA runtime.
