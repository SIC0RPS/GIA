# GIA Plugin Notes

## Required Layout

- Put each plugin in `src/gia/plugins/<name>/`.
- Entry file must be `src/gia/plugins/<name>/<name>.py`.
- Exported function name must exactly match `<name>`.

If folder, module, and function names do not match, the loader will skip or reject the plugin.

## Required Signature

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

GIA binds these by keyword. Renaming parameters or changing the call shape breaks loading.

## Runtime Contract

- `state` is a read-only snapshot of app state.
- `chat_history` may be `None`.
- `arg` is the optional string from `.myplugin value`.

Useful state handles include:

- `state.LLM`
- `state.QUERY_ENGINE`
- `state.EMBED_MODEL`
- `state.MODEL_NAME`
- `state.DATABASE_LOADED`

## Output Patterns

- Return a string for a simple one-shot result.
- Yield strings or message dict/list payloads to stream progress.
- Keep output serializable and human-readable.

Typical guard:

```python
llm = state.LLM
if llm is None:
    yield "Error: Load a model first (.load)."
    return
```

## Preferred Helpers

Use existing runtime helpers instead of rolling your own:

- `generate(...)` for normal LLM output
- `stream_generate(...)` for streaming text generation
- `system_prefix()` for the current system prompt

This keeps plugins aligned with provider quirks and active repo config.

## Plugin Loader Behavior

The plugin loader lives in `src/gia/GIA.py`. It auto-loads plugins at startup and supports reload behavior through `.reload` and UI controls.

When changing plugin infrastructure:

- inspect the loader
- inspect `handle_command()`
- inspect UI plugin button wiring
- verify both command-line and UI-triggered paths still work

## Good Plugin Habits

- Call `log_banner(__file__)` at module import if you want consistent logging style.
- Avoid mutating shared global runtime state from plugins unless the task explicitly requires it.
- Validate `arg` before using it.
- Catch exceptions and return or yield readable `Error: ...` messages.
- Stream intermediate status for long-running work.

## Common Mistakes

- Wrong folder/file/function naming
- Import-time access to `state`
- Returning non-serializable objects
- Reimplementing model loading inside the plugin
- Assuming chat history is always present
- Forgetting that plugins may be triggered from either CLI or UI
