# GIA Runtime Notes

## Key Files

- `src/gia/__main__.py`: console entrypoint; parses `--cli` and `--debug`, then calls `launch_app(args)`.
- `src/gia/GIA.py`: main app orchestration, UI wiring, CLI loop, command handling, plugin loading, and chat dispatch.
- `src/gia/core/utils.py`: model creation/loading, generation helpers, database helpers, and provider-specific behavior.
- `src/gia/config.py`: loads `config.toml`, resolves paths, and exposes `system_prefix()`.
- `src/gia/core/state.py`: singleton state manager and `ProjectState` snapshot.
- `config.toml`: generation defaults, prompt text, and path defaults.

## Architecture

GIA is not split into small service modules. The main runtime behavior is concentrated in `src/gia/GIA.py`, with supporting helpers in `core/utils.py`. Expect cross-cutting state dependencies.

The usual flow is:

1. `python -m gia` or installed `gia` command enters `src/gia/__main__.py`.
2. `main()` parses args and calls `launch_app(args)`.
3. `launch_app()` starts Gradio unless `--cli` is set, then enters the shared CLI loop.
4. User input goes through `process_input()`.
5. Dot-commands route through `handle_command()`.
6. Normal prompts route to query-engine or LLM paths.

## Current CLI Reality

- The CLI loop is interactive and reads from stdin.
- The current CLI contract is not a full one-shot API for selecting provider + model + prompt from flags.
- Another repo can shell into the interactive CLI, but model/provider selection still depends on existing state or additional code changes.
- If asked to add CLI model selection, implement it at the entrypoint/startup layer, not by overloading `generate()` alone.

## State Model

`state_manager` holds long-lived process state such as:

- `LLM`
- `QUERY_ENGINE`
- `MODEL_NAME`
- `MODEL_PATH`
- `DATABASE_LOADED`
- `MODE`
- `USE_CHAT`
- `COLLECTION_NAME`

The UI also keeps a separate mutable state dict with keys like:

- `mode`
- `model_path_gr`
- `hf_model_name`
- `openai_model_name`
- `openrouter_model_name`
- `database_loaded_gr`
- `use_query_engine_gr`

Many bugs come from changing one state path but not the other.

## Model Loading

`load_llm(mode, config)` delegates to `create_llm(mode, model_name)`.

Supported modes:

- `Local`
- `HuggingFace`
- `OpenAI`
- `OpenRouter`

Important details:

- Local mode uses `GPTQModel.load(...)`, then wraps it in `HuggingFaceLLM`.
- HuggingFace online mode uses `HuggingFaceInferenceAPI`.
- OpenAI mode uses `LlamaOpenAI`.
- OpenRouter mode uses `OpenRouter`.
- API keys are taken from environment variables, not config files.
- `create_llm()` is duplicated in `src/gia/core/utils.py`. Do not patch only one copy.

## Generation

Use the helpers from `core/utils.py`:

- `generate(...)`: primary non-stream helper with provider-aware caps and fallback logic.
- `stream_generate(...)`: streaming helper.

Do not reimplement provider-specific token caps, chat-vs-complete logic, or retry rules in plugins or CLI glue unless you are intentionally replacing that behavior.

## Config

`config.toml` controls:

- generation defaults
- data/model/db paths
- debug flag
- system prompt
- QA prompt
- Chroma collection name

`system_prefix()` reloads prompt text dynamically from `config.toml`. Use it when the task depends on the active system prompt.

## Environment Constraints

- The package requires Python 3.11+.
- Imports fail on Python 3.10 because `tomllib` is used directly.
- If validation fails before app startup, confirm interpreter version first.

## Editing Guidance

- For CLI changes, inspect `__main__.py`, `launch_app()`, `cli_loop()`, and any helper that initializes model state.
- For command behavior, inspect both `handle_command()` and all UI callbacks that reuse or mirror it.
- For database work, inspect wrapper functions around Chroma and the load/create/delete paths.
- For provider/model issues, inspect `load_llm()`, both `create_llm()` definitions, and `generate()`.

