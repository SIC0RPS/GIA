# src/gia/GIA.py
import sys
import os
import importlib
import threading
import inspect
import json
import logging
import gc
import hashlib
import traceback
import argparse
import time
import sys
import asyncio
import shlex
import threading
import torch
import gradio as gr
from functools import partial
from queue import Empty
from types import GeneratorType
from functools import partial
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from pathlib import Path
from colorama import Style, Fore
from logging.handlers import QueueHandler, QueueListener
from queue import Queue, Empty
from typing import Dict, Callable, List, Generator, Optional, Tuple, Any, Union
from gptqmodel import GPTQModel
from colorama import init
init(autoreset=True)

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import Settings

from gia.config import CONFIG, PROJECT_ROOT, system_prefix
from gia.core.logger import logger, log_banner
from gia.core.state import ( load_state, ProjectState, state_manager )
from gia.core.utils import (
    generate,
    clear_vram,
    save_database,
    load_database,
    get_system_info,
    append_to_chatbot,
    unload_model,
    load_llm,
    fetch_openai_models,
)


CONTEXT_WINDOW = CONFIG["CONTEXT_WINDOW"]
MAX_NEW_TOKENS = CONFIG["MAX_NEW_TOKENS"]
TEMPERATURE = CONFIG["TEMPERATURE"]
TOP_P = CONFIG["TOP_P"]
TOP_K = CONFIG["TOP_K"]
REPETITION_PENALTY = CONFIG["REPETITION_PENALTY"]
NO_REPEAT_NGRAM_SIZE = CONFIG["NO_REPEAT_NGRAM_SIZE"]
MODEL_PATH = CONFIG["MODEL_PATH"]
EMBED_MODEL_PATH = CONFIG["EMBED_MODEL_PATH"]
DB_PATH = CONFIG["DB_PATH"]
DEBUG = CONFIG["DEBUG"]
chat_history = []
log_banner(__file__)

class PluginSandbox:
    """Isolated sandbox for plugin execution using threading with cooperative stop.

    MIL PURPOSE:
    - RUN PLUGINS IN THEIR OWN THREADS WITH COOPERATIVE STOP CONTROL.
    - FORWARD PLUGIN LOGS VIA A QueueHandler → QueueListener PIPELINE.
    - STREAM RESULTS BACK TO CALLER USING AN OUTPUT QUEUE.

    SECURITY:
    - NO eval/exec. TRUSTS ONLY THE CALLABLE PASSED BY THE APPLICATION.
    - TUNNELS EXCEPTIONS BACK AS OBJECTS; AVOIDS LEAKING TRACEBACKS IN UI.

    EDGE CASES:
    - GENERATOR PLUGINS: STREAM CHUNKS UNTIL STOP EVENT OR COMMAND.
    - NON-GENERATOR PLUGINS: SINGLE PUT OF RESULT, THEN SENTINEL None.
    """

    def __init__(
        self,
        plugin_name: str,
        func: Callable,
        args: Tuple,
        kwargs: Dict,
    ) -> None:
        """Initialize sandbox with queues for communication.

        Args:
            plugin_name: Name of the plugin.
            func: Function to execute.
            args: Positional arguments for the function.
            kwargs: Keyword arguments for the function.
        """
        # MIL: ASSIGN METADATA AND IPC QUEUES
        self.plugin_name = plugin_name
        self.func = func
        self.args = args
        self.kwargs = kwargs

        # MIL: THREAD HANDLE + IPC
        self.thread: Optional[threading.Thread] = None
        self.output_queue: Queue = Queue()  # RESULTS/STREAMING
        self.input_queue: Queue = Queue()  # CONTROL COMMANDS (E.G., "stop")
        self.logger_queue: Queue = Queue()  # LOG RECORDS FROM WORKER
        self.stop_event = threading.Event()  # COOPERATIVE STOP SIGNAL

        # MIL: PER-SANDBOX QueueListener (TERMINAL/STDOUT ONLY)
        self.listener: Optional[QueueListener] = None

    def start(self) -> None:
        """Start the sandbox thread and logger listener.

        WHAT:
            - SPAWN DAEMON THREAD TO EXECUTE THE PLUGIN.
            - START A QueueListener TO FORWARD WORKER LOGS TO STDOUT.
        WHY:
            - NON-BLOCKING UI + TERMINAL VISIBILITY.
        """
        # MIL: CREATE AND START WORKER THREAD
        self.thread = threading.Thread(
            target=self._run_func,
            args=(
                self.func,
                self.args,
                self.kwargs,
                self.output_queue,
                self.input_queue,
                self.logger_queue,
                self.stop_event,
            ),
            daemon=True,
        )
        self.thread.start()

        # MIL: LISTENER WRITES TO STDOUT; USE logging.MODULE API
        handlers = [logging.StreamHandler(sys.stdout)]
        self.listener = QueueListener(self.logger_queue, *handlers)
        self.listener.start()

    def stop(self) -> None:
        """Stop the sandbox thread cooperatively and clean up resources.

        WHAT:
            - SET STOP FLAG AND QUEUE COMMAND.
            - JOIN WITH TIMEOUT TO AVOID HANGS.
            - STOP LISTENER AND CLEAR STATE.
        WHY:
            - GRACEFUL SHUTDOWN; PREVENT RESOURCE LEAKS.
        """
        # MIL: SIGNAL STOP
        self.stop_event.set()
        self.input_queue.put("stop")

        # MIL: JOIN WITH TIMEOUT FOR RELIABILITY
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                # NOTE: USE GLOBAL PROJECT LOGGER IF AVAILABLE
                try:
                    logger.warning(
                        "Timeout waiting for plugin '%s' thread to stop.",
                        self.plugin_name,
                    )
                except Exception:
                    pass

        # MIL: TEAR DOWN LISTENER
        if self.listener:
            try:
                self.listener.stop()
            finally:
                self.listener = None

    @staticmethod
    def _run_func(
        func: Callable,
        args: Tuple,
        kwargs: Dict,
        output_queue: Queue,
        input_queue: Queue,
        logger_queue: Queue,
        stop_event: threading.Event,
    ) -> None:
        """Target function for the thread: execute plugin with logger and stop check.

        LOGGING PIPELINE:
            worker thread -> QueueHandler(logger_queue)
                          -> QueueListener (in main thread) -> stdout

        RISKS MITIGATED:
            - DUPLICATE LOGS: HANDLER IS ALWAYS REMOVED IN FINALLY.
            - UI CRASH: EXCEPTIONS ARE SENT VIA QUEUE; THREAD DOES NOT HARD-CRASH.
        """
        # MIL: OBTAIN ROOT LOGGER VIA MODULE-LEVEL API (FIXES AttributeError)
        root_logger = logging.getLogger()

        # MIL: ATTACH A QueueHandler TO ROUTE RECORDS TO THIS SANDBOX'S QUEUE
        queue_handler: QueueHandler = QueueHandler(logger_queue)
        root_logger.addHandler(queue_handler)

        try:
            # MIL: DIAGNOSTIC — CONFIRM THREAD START
            try:
                logger.debug(
                    "Plugin thread started; executing '%s'.",
                    getattr(func, "__name__", "<anonymous>"),
                )
            except Exception:
                pass

            # MIL: EXECUTE PLUGIN (GENERATOR OR REGULAR)
            result = func(*args, **kwargs)

            if inspect.isgenerator(result):
                # MIL: STREAM CHUNKS WITH COOPERATIVE-STOP + COMMAND CHECK
                for chunk in result:
                    if stop_event.is_set():
                        try:
                            logger.info("Plugin stopped by event")
                        except Exception:
                            pass
                        break
                    try:
                        cmd = input_queue.get_nowait()
                        if cmd == "stop":
                            try:
                                logger.info("Plugin stopped by command")
                            except Exception:
                                pass
                            break
                    except Empty:
                        pass
                    # MIL: FORWARD OUTPUT
                    output_queue.put(chunk)
            else:
                # MIL: NON-GENERATOR RESULT PATH
                if stop_event.is_set():
                    try:
                        logger.info("Plugin stopped by event")
                    except Exception:
                        pass
                else:
                    output_queue.put(result)

            # MIL: SENTINEL — SIGNAL END OF STREAM
            output_queue.put(None)

        except Exception as exc:
            # MIL: SAFE ERROR TUNNEL — NO STACK TRACE LEAK TO UI BY DEFAULT
            try:
                logger.error("Plugin error: %s", exc)
            except Exception:
                pass
            output_queue.put(exc)
            output_queue.put(None)

        finally:
            # MIL: CRITICAL — ALWAYS REMOVE HANDLER TO PREVENT DUPLICATE LOGS/LEAKS
            try:
                root_logger.removeHandler(queue_handler)
                try:
                    queue_handler.close()
                except Exception:
                    pass
            except Exception:
                # MIL: NEVER RAISE ON CLEANUP
                pass

plugins: Dict[str, Callable] = {}
module_mtimes: Dict[str, float] = {}
plugins_dir_path: Path = PROJECT_ROOT / "plugins"
active_sandboxes: Dict[str, PluginSandbox] = {}
sandboxes_lock = threading.Lock()

def format_json_for_chat(item: Any) -> str:
    """Pretty-print dict/list for chat display without looking like a 'history' blob."""
    try:
        # Make JSON readable and safely delimited for the UI
        return "```json\n" + json.dumps(item, ensure_ascii=False, indent=2) + "\n```"
    except Exception:
        return str(item)

def load_plugins() -> Dict[str, Callable]:
    """Load plugins from plugins/ directory for dynamic extensibility.

    Returns:
        Dict[str, callable]: Dictionary of plugin functions.
    """
    global plugins, plugins_dir_path
    plugins = {}
    if not plugins_dir_path.is_dir():
        logger.warning(f"Plugins directory not found: {plugins_dir_path}")
        return plugins

    # TO SCAN DIRECT SUBDIRECTORIES ONLY
    for subdir in os.listdir(plugins_dir_path):
        subdir_path = plugins_dir_path / subdir
        if not subdir_path.is_dir():
            continue
        plugin_file = f"{subdir}.py"
        plugin_path = subdir_path / plugin_file
        if not plugin_path.is_file():
            logger.debug(f"No matching plugin file {plugin_file} in {subdir}")
            continue

        # TO IMPORT SUBMODULE
        module_name = f"gia.plugins.{subdir}.{subdir}"
        try:
            module = importlib.import_module(module_name)
            # TO COLLECT ONLY MATCHING LOCAL FUNCTION
            if subdir in dir(module):
                attr = getattr(module, subdir)
                if inspect.isfunction(attr) and attr.__module__ == module.__name__:
                    plugins[subdir] = attr
                    logger.debug(f"Loaded plugin function: {subdir} from {module_name}")
        except ImportError as e:
            logger.error(f"Failed to load plugin {module_name}: {e}")
        except Exception as e:
            logger.error(f"Error processing plugin {module_name}: {e}")

    logger.info(f"(L.P) Loaded {len(plugins)} plugins: {list(plugins.keys())}")
    return plugins

plugins = load_plugins()
logger.info(f"(L.P) Loaded {len(plugins)} plugins: {list(plugins.keys())}")

METADATA_TITLE = f"Generated by {PROJECT_ROOT.name.upper()}-{hash(PROJECT_ROOT) % 10000}"
dark_gray = Style.DIM + Fore.LIGHTBLACK_EX
reset_style = Style.RESET_ALL
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.INFO)
    
def m_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file.

    Args:
        filepath (str): Path to the file.

    Returns:
        str: SHA256 hash.
    """
    # TO COMPUTE HASH
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def handle_command(
    cmd: str,
    chat_history: List[Dict[str, str]],
    state: Optional[Dict[str, Any]] = None,
    is_gradio: bool = False,
) -> Union[str, Generator[Union[str, List[Dict[str, str]]], None, None]]:
    """
    Handle ONLY built-in & plugin commands (strings starting with '.').
    Returns:
      - str for immediate results (built-ins)
      - generator yielding text chunks or a full history list (plugins)
    No history mutation here; wrappers/process_input handle appends.
    """
    app_state = load_state()
    state_dict: Dict[str, Any] = state if isinstance(state, dict) else {}

    if not isinstance(chat_history, list):
        raise ValueError("chat_history must be a list[dict].")
    if not isinstance(cmd, str) or not cmd.strip():
        return "Error: Empty command."

    cmd_lower = cmd.strip().lower()
    llm_free = {".load", ".create", ".info", ".delete", ".unload"}
    if app_state.LLM is None and cmd_lower not in llm_free:
        return "Error: No model loaded. Please confirm a model first."

    if cmd_lower in llm_free:
        try:
            if cmd_lower == ".create":
                save_database()
                state_dict["use_query_engine_gr"] = True
                state_dict["database_loaded_gr"] = True
                state_dict["status_gr"] = "Database created"
                return "Database created. Query engine re-enabled."

            if cmd_lower == ".load":
                return load_database_wrapper(state_dict)

            if cmd_lower == ".info":
                cpu, mem, gpus = get_system_info()
                model_name = app_state.MODEL_NAME or "None"
                db_loaded = bool(app_state.DATABASE_LOADED)
                state_dict["status_gr"] = "System info retrieved"
                return (
                    "System Info:\n"
                    f"CPU Usage: {cpu}%\n"
                    f"Memory Usage: {mem}%\n"
                    f"GPU Usage: {gpus}%\n"
                    f"Model: {model_name}\n"
                    f"Database Loaded: {db_loaded}"
                )

            if cmd_lower == ".delete":
                db_path = Path(DB_PATH)
                if not db_path.exists():
                    state_dict["status_gr"] = "No database to delete"
                    return "No database to delete."
                proceed = True
                if not is_gradio:
                    try:
                        confirm = input("Confirm delete database? (y/n): ").strip().lower()
                        proceed = (confirm == "y")
                    except Exception:
                        proceed = False
                if not proceed:
                    state_dict["status_gr"] = "Delete canceled"
                    return "Delete canceled."
                try:
                    import shutil
                    shutil.rmtree(db_path)
                    state_manager.set_state("DATABASE_LOADED", False)
                    state_manager.set_state("INDEX", None)
                    state_manager.set_state("QUERY_ENGINE", None)
                    state_dict["use_query_engine_gr"] = False
                    state_dict["database_loaded_gr"] = False
                    state_dict["status_gr"] = "Database deleted"
                    return "Database deleted."
                except Exception as e:
                    state_dict["status_gr"] = "Delete failed"
                    return f"Error deleting database: {e}"

            if cmd_lower == ".unload":
                msg = unload_model()
                state_dict["status_gr"] = "Model unloaded" if "success" in msg.lower() else "Unload attempted"
                return msg

        except Exception as e:
            logger.error("Built-in command failed: %s", e)
            return f"Command failed: {e}"

    if not cmd_lower.startswith(".") or len(cmd_lower) <= 1:
        return "Error: Invalid command. Commands must start with '.'"

    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        return f"Error parsing command: {e}"

    if not parts or not parts[0].startswith("."):
        return "Error: Invalid plugin syntax."
    if len(parts) > 2:
        return "Error: Plugin command accepts at most one argument."

    plugin_name = parts[0][1:].lower()
    optional_arg = parts[1] if len(parts) == 2 else None
    if plugin_name not in plugins:
        return f"Error: Plugin '{plugin_name}' not found."

    plugin_func = plugins[plugin_name]
    available_kwargs = {
        "llm": app_state.LLM,
        "query_engine": app_state.QUERY_ENGINE,
        "embed_model": app_state.EMBED_MODEL,
        "chat_history": chat_history,
    }
    sig = inspect.signature(plugin_func)
    kwargs_to_pass: Dict[str, Any] = {k: v for k, v in available_kwargs.items() if k in sig.parameters}
    if optional_arg is not None:
        if "arg" in sig.parameters:
            kwargs_to_pass["arg"] = optional_arg
        elif "query" in sig.parameters:
            kwargs_to_pass["query"] = optional_arg

    try:
        sig.bind_partial(**kwargs_to_pass)
        missing = [
            p for p, param in sig.parameters.items()
            if (param.default is inspect._empty and
                param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY) and
                p not in kwargs_to_pass)
        ]
        if missing:
            return f"Error: Plugin '{plugin_name}' missing required args: {', '.join(missing)}"
    except TypeError as e:
        return f"Error: Cannot call plugin '{plugin_name}': {e}"

    sandbox = PluginSandbox(plugin_name, plugin_func, (), kwargs_to_pass)
    sandbox.start()
    with sandboxes_lock:
        active_sandboxes[plugin_name] = sandbox

    def _stream() -> Generator[Union[str, List[Dict[str, str]]], None, None]:
        try:
            start = time.time()
            max_seconds = 3600
            while True:
                try:
                    item = sandbox.output_queue.get(timeout=0.05)
                except Empty:
                    if (time.time() - start) > max_seconds:
                        raise TimeoutError(f"Plugin '{plugin_name}' timed out after {max_seconds}s")
                    continue
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                if isinstance(item, list) and all(isinstance(d, dict) for d in item):
                    yield item
                    continue
                try:
                    text = format_json_for_chat(item) if isinstance(item, (dict, list)) else str(item)
                except Exception:
                    text = str(item)
                if text:
                    yield text
        finally:
            sandbox.stop()
            with sandboxes_lock:
                active_sandboxes.pop(plugin_name, None)

    return _stream()

def process_input(
    user_text: str,
    state: Optional[Dict[str, Any]],
    chat_history: List[Dict[str, str]],
) -> Generator[List[Dict[str, str]], None, None]:
    """
    Minimal, reliable streaming for gr.Chatbot(type="messages").

    Emits: only full `chat_history` lists on each yield.
    Order: QueryEngine (stream if available) → LLM (stream if available)
           → single-shot complete() fallback.
    Context: session-only; model sees system_prefix + current user_text.
    Commands: caller must not append a user bubble for dot-commands;
              this function produces assistant-only output for them.
    """
    # MIL: INPUT VALIDATION – PREVENT TYPE/SHAPE ERRORS EARLY.
    text = (user_text or "").strip()
    if not text:
        return
    if not isinstance(chat_history, list):
        raise TypeError("chat_history must be List[Dict[str, str]]")
    state = state if isinstance(state, dict) else {}

    # MIL: LOCAL EMITTER – APPEND/EXTEND ASSISTANT CHUNK AND YIELD.
    def _emit_chunk(chunk: str) -> Generator[List[Dict[str, str]], None, None]:
        piece = (chunk or "").rstrip("\x00")
        if not piece:
            return
        if not chat_history or chat_history[-1].get("role") != "assistant":
            chat_history.append({"role": "assistant", "content": piece})
        else:
            chat_history[-1]["content"] += piece
        print(piece, end="", flush=True)
        yield list(chat_history)

    # MIL: DOT-COMMANDS – ASSISTANT-ONLY OUTPUT; NO USER BUBBLE HERE.
    if text.startswith("."):
        try:
            result = handle_command(text, chat_history, state=state, is_gradio=True)
        except Exception as exc:
            for out in _emit_chunk(f"Command error: {exc}"):
                yield out
            return

        # MIL: APPEND MESSAGES INSTEAD OF REPLACING HISTORY TO AVOID DATA LOSS.
        if isinstance(result, list) and all(isinstance(d, dict) for d in result):
            for m in result:
                role = (m.get("role") or "assistant").strip().lower()
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                append_to_chatbot(chat_history, content, metadata={"role": role})
                yield list(chat_history)
            return

        if isinstance(result, str):
            for out in _emit_chunk(result):
                yield out
            return

        for item in result:
            if isinstance(item, list) and all(isinstance(d, dict) for d in item):
                for m in item:
                    role = (m.get("role") or "assistant").strip().lower()
                    content = (m.get("content") or "").strip()
                    if not content:
                        continue
                    append_to_chatbot(chat_history, content, metadata={"role": role})
                    yield list(chat_history)
            else:
                for out in _emit_chunk(str(item)):
                    yield out
        return

    # MIL: QUERY ENGINE – PREFER STREAM; SAFE FALLBACK TO LLM.
    app_state = load_state()
    use_qe = bool(state.get("use_query_engine_gr", True))
    qe = getattr(app_state, "QUERY_ENGINE", None)
    if use_qe and qe is not None:
        try:
            resp = qe.query(text)
            gen = getattr(resp, "response_gen", None)
            if gen is not None:
                streamed = False
                for token in gen:
                    tok = str(token)
                    if not tok:
                        continue
                    streamed = True
                    for out in _emit_chunk(tok):
                        yield out
                if streamed:
                    return  # ONLY RETURN IF WE ACTUALLY STREAMED TOKENS
            # Non-stream or empty-stream fallback
            answer = getattr(resp, "response", None) or str(resp)
            if answer and answer.strip().lower() != "empty response":
                for out in _emit_chunk(answer):
                    yield out
                return
        except Exception as exc:
            logger.warning(f"[PI] Query engine failed; fallback to LLM. err={exc}")

    # === LLM – stream if Local/HF; else .complete(); else report ==============
    llm = getattr(app_state, "LLM", None)
    if llm is None:
        for out in _emit_chunk("Error: No model loaded."):
            yield out
        return

    sys_pfx = system_prefix().strip()
    prompt = f"{sys_pfx}\n{text}".strip() or " "

    # Local/HuggingFaceLLM: do safe streaming with enforced int dtype
    try:
        from llama_index.llms.huggingface import HuggingFaceLLM as _HFL_TYPE  # type: ignore[attr-defined]
    except Exception:
        _HFL_TYPE = tuple()  # type: ignore[assignment]

    if isinstance(llm, _HFL_TYPE) and hasattr(llm, "_model") and hasattr(llm, "_tokenizer"):
        import torch, threading
        from transformers import TextIteratorStreamer

        model = llm._model
        tok = llm._tokenizer
        device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
        ids = enc.get("input_ids")
        if ids is None or ids.numel() == 0 or ids.shape[-1] == 0:
            # MIL: NON-EMPTY GUARD — PREVENTS CACHE-POSITION BUG PATH.
            fid = (
                int(getattr(tok, "bos_token_id"))
                if getattr(tok, "bos_token_id", None) is not None
                else int(getattr(tok, "eos_token_id", 0))
            )
            ids = torch.tensor([[fid]], dtype=torch.long)
            attn = torch.ones_like(ids, dtype=torch.long)
        else:
            attn = enc.get("attention_mask")
            if attn is None:
                attn = torch.ones_like(ids, dtype=torch.long)

        # MIL: MOVE TO DEVICE W/O DTYPE CAST (EMBEDDING EXPECTS LONG)
        ids = ids.to(device)
        attn = attn.to(device)
        if ids.dtype is not torch.long:
            ids = ids.long()
        if attn.dtype is not torch.long:
            attn = attn.long()

        # Pad id (no global mutation)
        try:
            if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                tok.pad_token_id = int(tok.eos_token_id)
        except Exception:
            pass

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

        # MIL: ENFORCE CONFIGURED TOKEN BUDGET (NO 256 FALLBACK) + SYNC WRAPPER CAPS
        target_tokens = int(MAX_NEW_TOKENS)
        try:
            if hasattr(llm, "max_new_tokens"):
                llm.max_new_tokens = target_tokens
            if hasattr(llm, "generate_kwargs") and isinstance(llm.generate_kwargs, dict):
                llm.generate_kwargs["max_new_tokens"] = target_tokens
                llm.generate_kwargs.pop("max_length", None)
        except Exception:
            # MIL: SOFT-FAIL; DO NOT BLOCK GENERATION IF WRAPPER ATTRS DIFFER.
            pass

        gen_kwargs = {
            "input_ids": ids,
            "attention_mask": attn,
            "max_new_tokens": target_tokens,  # MIL: CONFIG-DRIVEN; FIXES EARLY CUT-OFF.
            "do_sample": True,
            "top_p": float(TOP_P),
            "temperature": float(TEMPERATURE),
            # break 4.55 bug path:
            "use_cache": False,  # MIL: SAFETY—BYPASS cache_position PATH; PERF: SMALL HIT, STABILITY WIN.
            # or: "cache_implementation": "static",
            "pad_token_id": int(getattr(tok, "pad_token_id", getattr(tok, "eos_token_id", 0))),
            "streamer": streamer,
        }

        # kick generation in a thread
        def _run():
            # MIL: INFERENCE MODE FOR PERF; DISABLES GRAD.
            with torch.inference_mode():
                model.generate(**gen_kwargs)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        streamed_any = False
        for piece in streamer:
            s = str(piece)
            if not s:
                continue
            streamed_any = True
            for out in _emit_chunk(s):
                yield out

        if streamed_any:
            return

        # No tokens streamed -> fall back to single shot
        from gia.core.utils import generate as _gen_safe
        text_out = _gen_safe(text, sys_pfx, llm)
        for out in _emit_chunk(text_out):
            yield out
        return

    # Non-Local providers: use LlamaIndex interface (no extra kwargs)
    stream_complete = getattr(llm, "stream_complete", None)
    if callable(stream_complete):
        try:
            streamed = False
            for ev in stream_complete(prompt):
                delta = ev if isinstance(ev, str) else (getattr(ev, "delta", None) or getattr(ev, "text", ""))
                if not delta:
                    continue
                streamed = True
                for out in _emit_chunk(delta):
                    yield out
            if streamed:
                return
        except Exception as exc:
            logger.warning(f"[PI] stream_complete failed; switching to complete(). err={exc}")

    complete = getattr(llm, "complete", None)
    if callable(complete):
        try:
            # DO NOT pass extra kwargs like 'use_cache' here
            resp = llm.complete(prompt)
            text_out = getattr(resp, "text", None) or str(resp)
            for out in _emit_chunk(text_out):
                yield out
            return
        except Exception as exc:
            logger.error(f"[PI] complete() failed: {exc}")
            for out in _emit_chunk(f"Generation error: {type(exc).__name__}"):
                yield out
            return

    for out in _emit_chunk("LLM backend does not support streaming or complete()."):
        yield out


def list_all_folders(folder: str) -> List[str]:
    """Return a list of all subdirectories in the given folder.

    Args:
        folder (str): Directory to scan.

    Returns:
        List[str]: List of subdirectory paths.
    """
    # TO SCAN DIRECTORIES
    return [
        os.path.join(folder, d)
        for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d))
    ]

def update_model_dropdown(directory: str) -> Tuple[gr.Dropdown, List[Dict[str, str]]]:
    """Update model dropdown with available folders.

    Args:
        directory (str): Directory to scan for models.

    Returns:
        Tuple[gr.Dropdown, List[Dict[str, str]]]: Updated dropdown and chat message.
    """
    # TO VALIDATE DIRECTORY
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        return gr.update(choices=[]), [
            {
                "role": "assistant",
                "content": f"Error: Directory '{directory}' does not exist.",
            }
        ]
    folders = list_all_folders(directory)
    return gr.update(choices=folders, value=folders[0] if folders else None), [
        {"role": "assistant", "content": f"Found {len(folders)} model folders."}
    ]

def finalize_model_selection(
    mode: str,
    state: Dict,
) -> Tuple[str, Dict]:
    """Finalize model selection based on mode.

    Args:
        mode: Selected mode (Local, HuggingFace, OpenAI, OpenRouter).
        state: Gradio state dictionary.

    Returns:
        Tuple[str, Dict]: Status message and updated state.
    """
    # UNLOAD PREVIOUS MODEL
    try:
        unload_model()
        logger.info("Previous model unloaded successfully")
    except Exception as e:
        logger.warning(f"Unload previous model failed: {str(e)}. Proceeding with load.")

    # LOAD STATE
    app_state = load_state()
    if not isinstance(state, dict):
        logger.error("Invalid state: expected dictionary")
        state = {}
        return "Error: Invalid state provided", state

    # SET MODE IN STATE
    state_manager.set_state("MODE", mode)
    logger.info(f"Model mode set to {mode} globally")

    # CONFIGURE MODEL PARAMETERS
    config = {}
    if mode == "Local":
        config["model_path"] = state.get("model_path_gr", CONFIG["MODEL_PATH"])
    elif mode == "HuggingFace":
        config["model_name"] = state.get("hf_model_name", "mistralai/Mistral-7B-Instruct-v0.2")
    elif mode == "OpenAI":
        config["model_name"] = state.get("openai_model_name", "gpt-4o")
    elif mode == "OpenRouter":
        config["model_name"] = state.get("openrouter_model_name", "x-ai/grok-3-mini")
    else:
        error_msg = f"Invalid mode: {mode}"
        logger.error(error_msg)
        state["status_gr"] = error_msg
        return error_msg, state

    # LOAD MODEL
    try:
        msg, llm = load_llm(mode, config)
        if llm:
            state_manager.set_state("LLM", llm)
            state_manager.set_state("MODEL_NAME", config.get("model_name", config.get("model_path")))
            state_manager.set_state("MODEL_PATH", config.get("model_path", ""))
            state["status_gr"] = f"{mode} model loaded: {state_manager.get_state('MODEL_NAME')}"
            state["model_loaded_gr"] = True
            logger.info(f"Loaded LLM type: {type(llm).__name__} for mode {mode}")
            return state["status_gr"], state
        else:
            raise RuntimeError(msg)
    except Exception as e:
        error_msg = f"Failed to load {mode} model: {str(e)}"
        logger.error(error_msg)
        state["status_gr"] = error_msg
        state["model_loaded_gr"] = False
        return error_msg, state

def chat_fn(
    message: str,
    chat_history: List[Dict[str, str]],
    state: Dict[str, Any],
) -> Generator[Tuple[str, List[Dict[str, str]], Dict[str, Any]], None, None]:
    """
    GRADIO CHAT BRIDGE (MINIMAL, RELIABLE).

    - APPENDS USER BUBBLE (EXCEPT DOT-COMMANDS), CLEARS TEXTBOX.
    - FORWARDS process_input() STREAM AS Chatbot UPDATES.
    - RETURNS TRIPLES: (textbox_value, chatbot_value, state).
    """
    # MIL: DEFENSIVE NORMALIZATION — STABLE TYPES FOR UI PIPELINE.
    chat_history = chat_history if isinstance(chat_history, list) else []
    state = state if isinstance(state, dict) else {}

    msg = (message or "").strip()
    if not msg:
        yield "", gr.update(value=list(chat_history)), state
        return

    # MIL: AVOID HISTORY POLLUTION FOR COMMANDS.
    is_command = msg.startswith(".")
    if not is_command:
        append_to_chatbot(chat_history, msg, metadata={"role": "user"})
        # MIL: IMMEDIATE UI UPDATE + CLEAR TEXTBOX.
        yield "", gr.update(value=list(chat_history)), state

    # MIL: STREAM ASSISTANT OUTPUT; COMMANDS ALSO EMIT FIRST UPDATE HERE.
    for updated_history in process_input(msg, state, chat_history):
        yield "", gr.update(value=updated_history), state

def handle_load_button(state: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Handle load button action with early initialization and deduplication.

    Args:
        state: Application state dictionary.

    Returns:
        Tuple[List[Dict[str, str]], Dict[str, Any]]: Updated chat history, updated state dict.
    """
    # TO INITIALIZE CHAT HISTORY EARLY FOR ALL RETURNS
    chat_history: List[Dict[str, str]] = []

    # TO LOAD STATE
    app_state = load_state()
    state_dict = state if isinstance(state, dict) else {
        "use_query_engine_gr": True,
        "status_gr": "",
        "model_path_gr": None,
        "database_loaded_gr": False,
    }
    if state_dict.get("database_loaded_gr", False):
        message = "Database already loaded."
        if not chat_history or chat_history[-1].get("content") != message:
            append_to_chatbot(chat_history, message, metadata={"role": "assistant"})
        return chat_history, state_dict
    model_path_gr = state_dict.get("model_path_gr")
    # For online modes, model_path_gr may be None, but database can still be loaded
    try:
        if app_state.LLM is not None and (app_state.MODEL_PATH == model_path_gr or model_path_gr is None):
            logger.info("(LOAD) LLM already loaded, skipping reload")
        else:
            logger.info(f"(LOAD) Loading model from {model_path_gr} as LLM is None or path mismatch")
            # Use mode from state_dict
            mode = state_dict.get("mode", "Local")
            config = {"model_path": model_path_gr} if mode == "Local" else {}
            msg, llm = load_llm(mode, config)
            if llm is None:
                message = f"Error: Failed to load model: {msg}"
                append_to_chatbot(chat_history, message, metadata={"role": "assistant"})
                return chat_history, state_dict
            state_manager.set_state("LLM", llm)
        result = load_database_wrapper(state_dict)
        message = str(result).strip()
        if message and (not chat_history or chat_history[-1].get("content") != message):
            append_to_chatbot(chat_history, message, metadata={"role": "assistant"})
        state_dict["use_query_engine_gr"] = True
        state_dict["status_gr"] = "Database loaded"
        state_dict["database_loaded_gr"] = True
    except Exception as e:
        logger.error(f"Error during load button handling: {e}")
        message = f"Error during loading: {str(e)}"
        append_to_chatbot(chat_history, message, metadata={"role": "assistant"})

    return chat_history, state_dict

def launch_app(args: argparse.Namespace) -> None:
    """Launch Gradio application in a background thread (if not CLI-only) and run CLI loop in main thread.

    Args:
        args: Parsed command-line arguments, including --cli flag.
    """
    # EVENT FOR SIGNALING GRADIO LAUNCH COMPLETE
    launch_event = threading.Event()

    # LAUNCH IN BACKGROUND THREAD FOR NON-BLOCKING TERMINAL
    def threaded_launch() -> None:
        """Thread target to launch Gradio server, with demo built inside for thread-safe async init."""
        # CREATE AND SET NEW EVENT LOOP FOR THIS THREAD
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # BUILD DEMO INSIDE THREAD TO BIND ASYNC EVENTS CORRECTLY
        with gr.Blocks(
            title=f"{tag()} - UI Gradio Interface",
            delete_cache=(60, 60),
            theme = gr.themes.Base(
                primary_hue="zinc",
                secondary_hue="stone",
                neutral_hue="stone",
                spacing_size="md",
                radius_size="lg",
                text_size="md",
                font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
                font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
            ).set(
                body_background_fill="#000000",
                block_background_fill="#000000",
                panel_background_fill="#000000",
                block_border_width="0px",
                block_shadow="none",
                block_title_text_color="white",
                input_background_fill="#111111",
                input_border_color="#222222",
                input_shadow="none",
                button_primary_background_fill="#141414",
                button_primary_text_color="white",
                button_primary_background_fill_hover="#1a1a1a",
                button_secondary_background_fill="#141414",
                button_secondary_text_color="white",
                color_accent_soft="#111111",
                table_border_color="#222222",
            ),
            css = """
            /* === FORCE PAGE BACKGROUND TO PURE BLACK === */
            html, body {
                background-color: #000000 !important;
                background-image: none !important;
            }

            /* === OVERRIDE GLOBAL CONTAINER === */
            .gradio-container {
                background-color: #000000 !important;
                color: #ffffff !important;
            }

            /* === OVERRIDE CHATBOT CONTAINER + INNER LAYERS === */
            [data-testid="chatbot"] {
                background-color: #0b0b0b !important;
            }
            [data-testid="chatbot"] > div,
            [data-testid="chatbot"] .overflow-y-auto,
            [data-testid="chatbot"] .bg-slate-900,
            [data-testid="chatbot"] .bg-slate-950,
            [data-testid="chatbot"] .bg-gray-900,
            [data-testid="chatbot"] .bg-gray-950,
            [data-testid="chatbot"] *:where(.bg-slate-900, .bg-slate-950, .bg-gray-900, .bg-gray-950) {
                background-color: #0b0b0b !important;
            }

            /* === ENSURE TEXT IS WHITE === */
            [data-testid="chatbot"], [data-testid="chatbot"] * {
                color: #ffffff !important;
            }

            /* === BUBBLES === */
            .message.bot, .bubble.bot {
                background-color: #121212 !important;
                border: 1px solid #1f1f1f !important;
            }
            .message.user, .bubble.user {
                background-color: #141414 !important;
                border: 1px solid #222222 !important;
            }

            /* === CODE BLOCKS === */
            .prose pre, .prose code {
                background: #0f0f0f !important;
                color: #ffffff !important;
            }

            /* === TEXT INPUTS === */
            textarea, input, select {
                background: #121212 !important;
                color: #ffffff !important;
                border-color: #222222 !important;
            }

            /* === BUTTONS === */
            button {
                background: #141414 !important;
                color: #ffffff !important;
                border: 1px solid #222222 !important;
            }
            button:hover {
                background: #1a1a1a !important;
            }

            /* === TABS / PANELS / BLOCKS === */
            .tabs, .tabitem, .panel, .block, .form {
                background: #0b0b0b !important;
            }
            """,
        ) as demo:
            model_name_display = gr.Markdown("Loading model...", elem_id="model-name-display")
            plugins = load_plugins()

            session_state = gr.State(
                {
                    "use_query_engine_gr": True,
                    "status_gr": "",
                    "model_name_gr": "Unknown Model",
                    "model_path_gr": None,
                    "mode": "Local",
                    "hf_model_name": "deepseek-ai/DeepSeek-R1",
                    "openai_model_name": "gpt-3.5-turbo",
                    "openrouter_model_name": "x-ai/grok-3-mini",
                }
            )

            chatbot = gr.Chatbot(
                type="messages",
                value=[],
                height=600,
                allow_tags=["think", "answer"],
                render_markdown=True,
                show_label=False,
                layout="bubble",
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Enter your message or command (e.g., .info, .load, .create, .info, .delete, .unload)",
                    lines=1,
                    scale=10,
                    show_label=False,
                )
                send_btn = gr.Button("➤", scale=1)

            # MODE SELECTION
            mode_radio = gr.Radio(
                choices=["Local", "HuggingFace", "OpenAI", "OpenRouter"],
                value="Local",
                label="Model Mode",
            )
            warning_md = gr.Markdown("", visible=False)

            # LOCAL UI
            with gr.Row(visible=True) as local_row:
                models_dir_input = gr.Textbox(label="Models Directory", value=CONFIG["MODEL_PATH"])
                folder_dropdown = gr.Dropdown(choices=[], label="Select a Model")
                scan_dir_button = gr.Button("Scan Directory")

            # HF UI
            with gr.Row(visible=False) as hf_row:
                hf_model_dropdown = gr.Dropdown(
                    choices=[
                        "deepseek-ai/DeepSeek-R1",
                        "Qwen/Qwen3-235B-A22B-Instruct-2507",
                        "meta-llama/Llama-2-7b-chat-hf",
                        "mistralai/Mistral-7B-Instruct-v0.1",
                        "google/gemma-7b-it",
                        "Qwen/Qwen3-8B",
                    ],
                    label="HF Model",
                    value="deepseek-ai/DeepSeek-R1",
                )

            # OpenAI UI
            with gr.Row(visible=False) as openai_row:
                openai_model_dropdown = gr.Dropdown(
                    choices=fetch_openai_models(),
                    label="OpenAI Model",
                    value="gpt-4o",
                )

            # OpenRouter UI
            with gr.Row(visible=False) as openrouter_row:
                openrouter_model_text = gr.Textbox(
                    label="OpenRouter Model",
                    value="x-ai/grok-3-mini",
                    placeholder="Enter OpenRouter model name (e.g., x-ai/grok-3-mini)",
                )

            # Confirm model + Load DB
            confirm_model_button = gr.Button("Confirm Model")
            with gr.Row():
                load_button = gr.Button("Load Database")

            # ------------------------ PLUGINS UI (TOP-LEVEL) ------------------------
            with gr.Accordion("Plugins", open=True):
                if not plugins:
                    gr.Markdown("No plugins loaded.")
                else:
                    # define the handler once
                    def plugin_handler_simple(
                        name: str,
                        chat_history: List[Dict[str, str]],
                        state: Dict[str, Any],
                    ) -> Generator[Tuple[Any, Dict[str, Any]], None, None]:
                        """
                        Run a plugin by delegating to the unified backend via '.<plugin>' command,
                        streaming results while progressively updating the chatbot & state.
                        """
                        import gradio as gr

                        try:
                            # unwrap state if it's a gr.State
                            if hasattr(state, "value"):
                                state_dict = state.value or {}
                            elif isinstance(state, dict):
                                state_dict = state
                            else:
                                state_dict = {}

                            hist = chat_history if isinstance(chat_history, list) else []
                            cmd = f".{name}"

                            # make it visible that the user triggered the plugin
                            append_to_chatbot(hist, cmd, metadata={"role": "user"})
                            yield gr.update(value=list(hist)), state_dict

                            first = True
                            for item in process_input(cmd, state_dict, hist):
                                if isinstance(item, list):
                                    hist[:] = item
                                else:
                                    chunk = str(item)
                                    if not chunk:
                                        yield gr.update(value=list(hist)), state_dict
                                        continue
                                    if first:
                                        append_to_chatbot(hist, chunk, metadata={"role": "assistant"})
                                        first = False
                                    else:
                                        if hist and hist[-1].get("role") == "assistant":
                                            hist[-1]["content"] += chunk
                                        else:
                                            append_to_chatbot(hist, chunk, metadata={"role": "assistant"})

                                yield gr.update(value=list(hist)), state_dict

                            yield gr.update(value=list(hist)), state_dict

                        except Exception as e:
                            logger.error(f"[PLUGIN BTN] {name} error: {e}")
                            append_to_chatbot(chat_history, f"Plugin '{name}' error: {e}", metadata={"role": "assistant"})
                            yield gr.update(value=list(chat_history)), (state.value if hasattr(state, "value") else state or {})

                    with gr.Row():
                        for plugin_name in plugins.keys():
                            btn = gr.Button(value=plugin_name.capitalize())
                            btn.click(
                                fn=partial(plugin_handler_simple, plugin_name),
                                inputs=[chatbot, session_state],
                                outputs=[chatbot, session_state],
                                queue=True,
                            )

            def update_mode_visibility(selected_mode: str, state):
                if hasattr(state, "value"):
                    state_dict = state.value
                else:
                    state_dict = state if isinstance(state, dict) else {"mode": selected_mode}
                state_dict["mode"] = selected_mode
                warning_value = ""
                warning_visible = False
                hf_update = gr.update()
                if selected_mode == "HuggingFace":
                    if not os.getenv("HF_TOKEN"):
                        warning_value = (
                            "HF_TOKEN not detected. Please set it in your environment: "
                            "`export HF_TOKEN=your_token_here` and restart the app."
                        )
                        warning_visible = True
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi(token=os.getenv("HF_TOKEN"))
                        if not api.token:
                            logger.warning("HF_TOKEN not set; fetching public models only - may limit results.")
                        deployed = api.list_models(
                            inference="warm", pipeline_tag="text-generation", limit=100, sort="downloads", direction=-1
                        )
                        models = [m.id for m in deployed if m.id]
                        models = sorted(set(models))
                        hf_update = gr.update(choices=models, value=models[0] if models else "deepseek-ai/DeepSeek-R1")
                        logger.debug(f"Fetched {len(models)} warm inference-supported models from HfApi")
                    except Exception as e:
                        logger.warning(f"Failed to fetch HF models via HfApi: {e}")
                        models = [
                            "deepseek-ai/DeepSeek-R1",
                            "deepseek-ai/DeepSeek-V3",
                            "mistralai/Mistral-7B-Instruct-v0.2",
                            "meta-llama/Meta-Llama-3-8B-Instruct",
                            "Qwen/Qwen3-14B",
                        ]
                        hf_update = gr.update(choices=models, value="deepseek-ai/DeepSeek-R1")
                elif selected_mode == "OpenAI":
                    if not os.getenv("OPENAI_API_KEY"):
                        warning_value = (
                            "OPENAI_API_KEY not detected. Please set it in your environment: "
                            "`export OPENAI_API_KEY=sk-your_key_here` and restart the app."
                        )
                        warning_visible = True
                elif selected_mode == "OpenRouter":
                    if not os.getenv("OPENROUTER_API_KEY"):
                        warning_value = (
                            "OPENROUTER_API_KEY not detected. Please set it in your environment: "
                            "`export OPENROUTER_API_KEY=your_key_here` and restart the app."
                        )
                        warning_visible = True
                return (
                    gr.update(visible=selected_mode == "Local"),
                    gr.update(visible=selected_mode == "HuggingFace"),
                    gr.update(visible=selected_mode == "OpenAI"),
                    gr.update(visible=selected_mode == "OpenRouter"),
                    gr.update(value=warning_value, visible=warning_visible),
                    gr.State(state_dict),
                    hf_update,
                )

            mode_radio.change(
                fn=update_mode_visibility,
                inputs=[mode_radio, session_state],
                outputs=[local_row, hf_row, openai_row, openrouter_row, warning_md, session_state, hf_model_dropdown],
            )

            hf_model_dropdown.change(
                lambda model, state: (state.value.update({"hf_model_name": model}) or state)
                if hasattr(state, "value")
                else (state.update({"hf_model_name": model}) or state),
                inputs=[hf_model_dropdown, session_state],
                outputs=[session_state],
            )
            openai_model_dropdown.change(
                lambda model, state: (state.value.update({"openai_model_name": model}) or state)
                if hasattr(state, "value")
                else (state.update({"openai_model_name": model}) or state),
                inputs=[openai_model_dropdown, session_state],
                outputs=[session_state],
            )
            openrouter_model_text.change(
                lambda model, state: (state.value.update({"openrouter_model_name": model}) or state)
                if hasattr(state, "value")
                else (state.update({"openrouter_model_name": model}) or state),
                inputs=[openrouter_model_text, session_state],
                outputs=[session_state],
            )

            scan_dir_button.click(
                fn=update_model_dropdown,
                inputs=[models_dir_input],
                outputs=[folder_dropdown, chatbot],
            )

            # Confirm model
            def confirm_model_handler(
                selected_folder: str,
                state: gr.State,
                mode: str,
                hf_model: str,
                openai_model: str,
                openrouter_model: str,
                chat_history: List[Dict[str, str]],
            ) -> Tuple[List[Dict[str, str]], Dict, str]:
                """Handle model confirmation button in Gradio, updating state and loading model."""
                if hasattr(state, "value"):
                    state_dict = state.value
                else:
                    state_dict = state if isinstance(state, dict) else {}
                logger.debug(f"Confirming model for mode: {mode}, state: {state_dict}")
                state_dict["model_path_gr"] = selected_folder if mode == "Local" else None
                state_dict["hf_model_name"] = hf_model if mode == "HuggingFace" else state_dict.get("hf_model_name", "deepseek-ai/DeepSeek-R1")
                state_dict["openai_model_name"] = openai_model if mode == "OpenAI" else state_dict.get("openai_model_name", "gpt-3.5-turbo")
                state_dict["openrouter_model_name"] = openrouter_model if mode == "OpenRouter" else state_dict.get("openrouter_model_name", "x-ai/grok-3-mini")
                try:
                    status, updated_state = finalize_model_selection(mode, state_dict)
                    logger.info(f"Model confirmation successful for mode {mode}")
                    updated_chatbot = append_to_chatbot(chat_history, status, metadata={"role": "assistant"})
                    return updated_chatbot, updated_state, status
                except Exception as e:
                    error_msg = f"Model confirmation failed: {str(e)}"
                    logger.error(error_msg)
                    state_dict["status_gr"] = error_msg
                    state_dict["model_loaded_gr"] = False
                    updated_chatbot = append_to_chatbot(chat_history, error_msg, metadata={"role": "assistant"})
                    return updated_chatbot, state_dict, error_msg

            confirm_model_button.click(
                fn=confirm_model_handler,
                inputs=[folder_dropdown, session_state, mode_radio, hf_model_dropdown, openai_model_dropdown, openrouter_model_text, chatbot],
                outputs=[chatbot, session_state, model_name_display],
            )

            # Load DB
            load_button.click(
                fn=handle_load_button,
                inputs=[session_state],
                outputs=[chatbot, session_state],
            )
            send_btn.click(
                fn=chat_fn,
                inputs=[msg, chatbot, session_state],
                outputs=[msg, chatbot, session_state],
                queue=True,
            )
            msg.submit(
                fn=chat_fn,
                inputs=[msg, chatbot, session_state],
                outputs=[msg, chatbot, session_state],
                queue=True,
            )

        # INIT GRADIO SERVER LAUNCH
        demo.queue()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            debug=True,
            inbrowser=True,
            quiet=True,
        )
        # SIGNAL LAUNCH COMPLETE
        launch_event.set()

    try:
        if not args.cli:
            # START DAEMON THREAD FOR GRADIO
            launch_thread = threading.Thread(target=threaded_launch, daemon=True)
            launch_thread.start()
            time.sleep(1)
            logger.info("Gradio app launched in background thread. Main terminal freed for CLI.")
        # RUN CLI LOOP IN MAIN THREAD
        cli_loop()
    except Exception as e:
        logger.error(f"Failed to launch Gradio in thread or start CLI: {str(e)}")
        raise

def tag() -> str:
    """Generate timestamped prefix for logger, including current mode for context.

    Note: Relies on shared memory for state updates across threads (e.g., Gradio/CLI).
    """
    # TO LOAD APPLICATION STATE - ENSURES CURRENT CONFIGURATION IS USED; THREAD-SAFE READ.
    app_state: ProjectState = load_state()
    # TO DETECT PRESENCE FOR MODE - LOCAL IF MODEL SET (VIA LOAD_LLM FOR LOCAL), ELSE ONLINE.
    model = getattr(app_state, "MODEL", None)
    mode_str = "local" if model is not None else "online"
    # TO FORMAT TIMESTAMP - ENSURES CONSISTENT LOG PREFIX.
    current_time = datetime.now().strftime("[%H:%M:%S]")
    # TO BUILD PREFIX - COMBINES TIME AND MODE FOR TRACEABILITY.
    return f"{current_time} GIA@{mode_str}: "

def cli_prompt() -> str:
    """
    Generate a colored Linux-like terminal prompt string.

    Format: 'GIA@<mode>:<current_path>$ ' with color styling.
    Mode resolution:
      - 'idle'   -> no model loaded (LLM is None)
      - 'local'  -> MODE == 'Local' and LLM is set
      - 'online' -> otherwise (e.g., OpenAI/HF/OpenRouter providers)
    """
    from colorama import Fore
    import os, sys

    # --- safe access to state without crashing on older StateManager variants ---
    try:
        from gia.core.state import state_manager, load_state
    except Exception:
        state_manager, load_state = None, None

    def _safe_sm_get(key: str, default=None):
        try:
            if state_manager and hasattr(state_manager, "get_state"):
                return state_manager.get_state(key, default)
            if state_manager and hasattr(state_manager, "_state"):
                return getattr(state_manager, "_state", {}).get(key, default)
        except Exception:
            pass
        return default

    llm = None
    mode = None

    # prefer structured snapshot when available
    try:
        if load_state:
            snap = load_state()
            llm = getattr(snap, "LLM", None)
            mode = getattr(snap, "MODE", None)
    except Exception:
        # fall back to reading the live store
        llm = _safe_sm_get("LLM", None)
        mode = _safe_sm_get("MODE", None)

    mode_norm = (mode or "").strip().lower()
    if llm is None:
        mode_str = "idle"
    elif mode_norm == "local":
        mode_str = "local"
    else:
        mode_str = "online"

    # cwd → '~' abbreviation
    cwd = os.getcwd()
    home = os.environ.get("HOME", "")
    if not home:
        raise ValueError("HOME environment variable not set; required for path abbreviation.")
    path_str = "~" + cwd[len(home):] if cwd.startswith(home) else cwd
    if not path_str:
        raise ValueError("Current working directory could not be determined.")

    user = "GIA"
    host = mode_str
    prompt = f"{Fore.GREEN}{user}@{host}{Fore.RESET}:~$ "
    return prompt

def cli_loop() -> None:
    """
    CLI input loop with Rich Markdown rendering.

    - Mirrors Gradio chat flow: append user bubble for non-commands,
      then stream assistant updates from `process_input()` snapshots.
    - Renders deltas with `rich.markdown.Markdown` (monokai code theme).
    """
    chat_history: List[Dict[str, str]] = []
    state: Dict[str, Any] = {"cli_mode": True, "session_start": time.time()}

    console = Console(soft_wrap=True)
    console.print("Entering CLI mode. Type `.exit` to quit.")
    console.print(
        f"Session started: {datetime.fromtimestamp(state['session_start']).strftime('%H:%M:%S')}"
    )

    while True:
        try:
            user_input = input(cli_prompt()).strip()  # type: ignore[name-defined]
        except (EOFError, KeyboardInterrupt):
            dur = time.time() - state["session_start"]
            console.print(f"\nExiting CLI. Session duration: {dur:.1f}s")
            break

        if not user_input:
            continue
        if user_input == ".exit":
            console.print("Session ended.")
            break

        is_command = user_input.startswith(".")
        if not is_command:
            append_to_chatbot(  # type: ignore[name-defined]
                chat_history, user_input, metadata={"role": "user"}
            )
            console.print(Markdown(user_input, code_theme="monokai"))

        # STREAM ASSISTANT: echo raw to stdout; re-render full assistant message via Live
        assistant_buffer: str = ""
        last_len: int = 0

        # MIL: isolate a live panel per response to avoid flicker across turns
        with Live(
            Panel(
                Markdown("", code_theme="monokai"),
                title="[bold]assistant",
                border_style="cyan",
            ),
            console=console,
            refresh_per_second=24,
            transient=True,  # remove panel after completion; final text remains in history
        ) as live:
            for snapshot in process_input(  # type: ignore[name-defined]
                user_input, state, chat_history
            ):
                # STRING CHUNK → raw echo + buffer + live re-render
                if isinstance(snapshot, str):
                    chunk: str = snapshot
                    if chunk:
                        # Echo verbatim to stdout (no ANSI), as required by contract
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                        assistant_buffer += chunk
                        live.update(
                            Panel(
                                Markdown(assistant_buffer, code_theme="monokai"),
                                title="[bold]assistant",
                                border_style="cyan",
                            )
                        )
                    continue

                # LIST SNAPSHOT → sync full chat_history, then re-render full assistant
                if isinstance(snapshot, list):
                    chat_history[:] = snapshot
                    idx: Optional[int] = next(
                        (
                            i
                            for i in range(len(chat_history) - 1, -1, -1)
                            if chat_history[i].get("role") == "assistant"
                        ),
                        None,
                    )
                    if idx is None:
                        continue
                    content = chat_history[idx].get("content") or ""
                    if not isinstance(content, str):
                        content = str(content)
                    # Compute delta for stdout echo to preserve contract
                    if len(content) > last_len:
                        delta = content[last_len:]
                        last_len = len(content)
                        if delta:
                            sys.stdout.write(delta)
                            sys.stdout.flush()
                    assistant_buffer = content
                    live.update(
                        Panel(
                            Markdown(assistant_buffer, code_theme="monokai"),
                            title="[bold]assistant",
                            border_style="cyan",
                        )
                    )
                    continue

                # Unknown snapshot type → ignore (precise, no blanket except)
                # MIL: Do not raise; generators may yield control sentinels in some plugin paths.

        console.print("")

def load_database_wrapper(state: Dict[str, Any]) -> str:
    """Load DB + query engine using the current LLM; keep UI flags in sync."""
    app_state = load_state()
    if app_state.LLM is None:
        return "Error: Load a model first (Local/HF/OpenAI/OpenRouter)."
    try:
        msg = load_database(app_state.LLM)  # updates state_manager
        state["database_loaded_gr"] = True
        state["use_query_engine_gr"] = True
        state["status_gr"] = "Database loaded"
        return (msg or "Database loaded.").strip()
    except Exception as e:
        state["database_loaded_gr"] = False
        state["use_query_engine_gr"] = False
        state["status_gr"] = "Database load failed"
        logger.error("load_database_wrapper failed: %s", e)
        return f"Error loading database: {e}"

def signal_handler(sig, frame) -> None:
    """Handle termination signals."""
    # TO LOG SIGNAL
    print(
        f"{Style.DIM + Fore.LIGHTBLACK_EX}{tag()}Signal received: {sig}. Exiting gracefully.{Style.RESET_ALL}"
    )
    sys.exit(0)

def cleanup() -> None:
    """Clean up resources on exit."""
    # TO CLEAN UP
    print(
        f"{Style.DIM + Fore.LIGHTBLACK_EX}{tag()}Cleaning up resources...{Style.RESET_ALL}"
    )
    gc.collect()
    try:
        torch.cuda.empty_cache()
        print(
            f"{Style.DIM + Fore.LIGHTBLACK_EX}{tag()}CUDA cache cleared.{Style.RESET_ALL}"
        )
    except ImportError:
        print(
            f"{Style.DIM + Fore.LIGHTBLACK_EX}{tag()}No CUDA detected. Cleanup complete.{Style.RESET_ALL}"
        )

if __name__ == "__main__":
    import signal
    import argparse

    # PARSE COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser(description="GIA: General Intelligence Assistant")
    parser.add_argument("--cli", action="store_true", help="Run in CLI-only mode without Gradio")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        print(f"{dark_gray}{tag()}Starting application...{reset_style}")
        launch_app(args)
    except Exception as e:
        logger.exception(f"Critical error in main application: {e}")
        print(f"{tag()}A critical error occurred: {e}. Exiting.")
        traceback.print_exc()
    finally:
        cleanup()