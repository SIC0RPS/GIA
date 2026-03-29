---
name: gia-senior-dev
description: Work effectively as a senior developer inside the GIA repository. Use when Codex needs to inspect, modify, debug, or extend GIA runtime code, model-loading logic, CLI/UI behavior, RAG/database flow, config handling, or plugins under `src/gia/**`. Also use when creating or fixing GIA plugins in `src/gia/plugins/name/name.py`, when tracing how commands like `.load` and `.reload` behave, or when another repo wants to drive GIA through its current CLI entrypoints.
---

# GIA Senior Dev

## Overview

Use the real repo shape, not assumptions from the README. GIA is a Python 3.11+ package with a Gradio UI, an interactive CLI, shared in-memory state, model backends for Local/HuggingFace/OpenAI/OpenRouter, Chroma-backed RAG, and a plugin system that dispatches dot-commands into Python functions.

## Start Here

- Read [references/runtime.md](./references/runtime.md) before editing runtime behavior, model loading, CLI/UI flows, config usage, or shared state.
- Read [references/plugins.md](./references/plugins.md) before creating or changing anything under `src/gia/plugins/` or the plugin loader.
- Prefer `rg` for code search.
- Respect the current worktree. This repo may already contain uncommitted changes unrelated to your task.

## Working Rules

- Use Python 3.11+ for any import or runtime validation. The repo imports `tomllib`, so Python 3.10 will fail immediately.
- Treat README guidance as secondary to code. Confirm behavior in `src/gia/__main__.py`, `src/gia/GIA.py`, `src/gia/core/utils.py`, `src/gia/config.py`, and `src/gia/core/state.py`.
- Keep changes narrow. GIA has large multi-purpose files, so broad refactors raise regression risk quickly.
- Preserve both interfaces unless the task explicitly narrows scope. UI and CLI share state and helper paths.
- When changing commands or plugin behavior, check both the command handler and the UI wiring that calls the same path.

## Workflow

1. Identify the layer first.
   Runtime entrypoint, app orchestration, generation helper, config/state, database helper, or plugin.
2. Trace the live call path.
   For user input this usually means `__main__.py` -> `launch_app()` -> UI/CLI handler -> `process_input()` / `handle_command()` / model helpers.
3. Confirm state dependencies.
   GIA stores core handles in `state_manager`; bugs often come from missing or stale `LLM`, `QUERY_ENGINE`, `MODEL_PATH`, `MODE`, or UI state dict keys.
4. Edit the smallest correct layer.
   Do not push CLI concerns into `generate()` if the real missing piece is model initialization or argument parsing.
5. Validate with the closest realistic command.
   If imports or full app launch are too heavy, validate the specific helper path you changed. State clearly what you could not run.

## High-Value Repo Facts

- Console entrypoint is `gia = "gia.__main__:main"`.
- CLI flags are minimal today: `--cli` and `--debug`.
- CLI is interactive; it is not a complete one-shot model-selection interface by default.
- Built-in dot-commands currently include `.load`, `.create`, `.info`, `.delete`, `.unload`, and `.reload`.
- Model/provider selection is largely driven by UI state and shared helper functions, not a full standalone CLI contract.
- `create_llm()` appears twice in `src/gia/core/utils.py`. When changing model creation, update both definitions or consolidate them first.

## Plugin Work

- Follow the exact naming rule: `src/gia/plugins/<name>/<name>.py` with `def <name>(...)`.
- Match the required signature exactly: `state`, optional `chat_history`, optional `arg`.
- Treat `state` as a read-only snapshot. Do not cache it beyond the plugin call.
- Use `generate()` or `stream_generate()` for LLM work instead of reimplementing provider logic.
- For long jobs, yield progress. For simple jobs, return a string.
- Surface readable errors such as `Error: Load a model first (.load).`

## Validation

- For package/runtime checks, prefer commands that use the repo’s intended Python version and environment.
- Validate skill or docs changes with the smallest relevant command.
- If a task touches plugin loading, command parsing, or UI dispatch, inspect both direct command execution and UI button wiring.
