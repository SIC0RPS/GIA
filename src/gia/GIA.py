# src/gia/GIA.py
import sys
import os
import importlib
import tomllib
import threading
import inspect
import json
import logging
import gc
import re
import shutil
import hashlib
import traceback
import argparse
import time
import asyncio
import shlex
import torch
import gradio as gr
import chromadb
from transformers import TextIteratorStreamer
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
from typing import Dict, Callable, List, Generator, Optional, Tuple, Any, Union, Tuple
from colorama import init

from gia.config import CONFIG, PROJECT_ROOT, system_prefix
from gia.core.logger import logger, log_banner
from gia.core.state import load_state, ProjectState, state_manager
from gia.core.utils import (
    save_database,
    load_database,
    get_system_info,
    append_to_chatbot,
    unload_model,
    load_llm,
    fetch_openai_models,
)

init(autoreset=True)

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

#DB CFG START
ACTIVE_DB_PATH: str = DB_PATH
CHROMA_CLIENT: Optional[chromadb.PersistentClient] = None  # type: ignore[attr-defined]

_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _ensure_client() -> chromadb.PersistentClient:  # type: ignore[override]
    """Ensure a Chroma PersistentClient is initialized at ACTIVE_DB_PATH."""
    global CHROMA_CLIENT
    if CHROMA_CLIENT is None:
        os.makedirs(ACTIVE_DB_PATH, exist_ok=True)
        CHROMA_CLIENT = _init_chroma_at_path(ACTIVE_DB_PATH)
    return CHROMA_CLIENT


def _is_valid_db_name(name: str) -> bool:
    """Validate DB name for safe subfolder creation (no traversal)."""
    return bool(_VALID_NAME_RE.fullmatch(name))

def _name_from_path(path: str) -> str:
    """Turn a path into a canonical UI name."""
    base = os.path.normpath(DB_PATH)
    path = os.path.normpath(path)
    if path == base:
        return "(base)"
    return os.path.basename(path)

def _init_chroma_at_path(path: str) -> chromadb.PersistentClient:  # type: ignore[override]
    """Initialize a Chroma PersistentClient at the given path."""
    client = chromadb.PersistentClient(path=path)  # type: ignore[attr-defined]
    # Light sanity ping; also ensures on-disk init
    try:
        client.heartbeat()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chroma heartbeat failed at %s: %s", path, exc)
    return client


def ui_refresh_db_list() -> Tuple[Any, str]:
    """
    Refresh the dropdown with real Chroma collections at ACTIVE_DB_PATH.

    Returns:
        (gr.update(choices=<collections>, value=<active or first>), status_message)
    """
    client = _ensure_client()
    names: List[str] = []
    try:
        cols = client.list_collections()
        for c in cols:
            nm = getattr(c, "name", None)
            if nm is None and isinstance(c, dict):
                nm = c.get("name")
            if nm:
                names.append(str(nm))
    except Exception as exc:
        logger.error("Failed to list Chroma collections: %s", exc)
    names = sorted(set(names), key=str.lower)

    active = None
    try:
        active = state_manager.get_state("COLLECTION_NAME") or None
    except Exception:
        active = None
    value = active if active in names else (names[0] if names else None)

    msg = f"Detected {len(names)} collection(s) in active DB."
    logger.info("Refreshed collections: %s (active=%s)", names, value)
    return gr.update(choices=names, value=value), msg


def ui_create_db(new_name: str) -> Tuple[Any, str, str]:
    """
    Create a new Chroma collection under ACTIVE_DB_PATH.
    Returns: (dropdown update, status message, cleared new_name)
    """
    name = (new_name or "").strip()
    if not _is_valid_db_name(name):
        return (
            gr.update(),
            "Invalid name. Use 1–64 chars: A-Z, a-z, 0-9, _ or -.",
            new_name,
        )

    client = _ensure_client()
    try:
        # Create (or get) the collection so it's immediately available
        client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        logger.info("Created collection '%s'", name)
    except Exception as exc:
        logger.error("Failed creating collection '%s': %s", name, exc)
        return gr.update(), f"Error creating collection '{name}': {exc}", new_name

    # Refresh list and select the new collection
    update, _ = ui_refresh_db_list()
    return update, f"Collection '{name}' created.", ""

def ui_load_db(selected_name: str) -> str:
    """
    Select the active Chroma collection name for subsequent DB operations.
    Does not build the query engine (that's done by load_database_wrapper()).
    """
    name = (selected_name or "").strip()
    if not name:
        return "Error: No collection selected."

    client = _ensure_client()
    try:
        # Validate it exists
        client.get_collection(name=name)
    except Exception as exc:
        logger.error("Load collection '%s' failed: %s", name, exc)
        return f"Error: Collection '{name}' not found."

    # Set the active collection override for utils._resolve_collection_name()
    try:
        state_manager.set_state("COLLECTION_NAME", name)
    except Exception as exc:
        logger.debug("Failed to set active collection in state: %s", exc)

    msg = f"Selected collection: {name}"
    logger.info(msg)
    return msg


def ui_delete_db(selected_name: str, confirm_text: str) -> Tuple[Any, str, str]:
    """
    Delete a Chroma collection by name.
    Requires confirm_text == 'DELETE'.
    Returns: (dropdown update, status message, cleared confirm_text)
    """
    if (confirm_text or "").strip().upper() != "DELETE":
        return gr.update(), "Type DELETE to confirm.", confirm_text

    name = (selected_name or "").strip()
    if not name:
        return gr.update(), "Error: No collection selected.", confirm_text

    client = _ensure_client()
    try:
        client.delete_collection(name=name)
        logger.info("Deleted collection '%s'", name)
        # Clear active override if it was pointing at the deleted collection
        try:
            if state_manager.get_state("COLLECTION_NAME") == name:
                state_manager.set_state("COLLECTION_NAME", None)
                state_manager.set_state("DATABASE_LOADED", False)
                state_manager.set_state("INDEX", None)
                state_manager.set_state("QUERY_ENGINE", None)
        except Exception:
            pass
    except Exception as exc:
        logger.error("Failed deleting collection '%s': %s", name, exc)
        return gr.update(), f"Error deleting collection '{name}': {exc}", confirm_text

    update, _ = ui_refresh_db_list()
    return update, f"Collection '{name}' deleted.", ""

def db_unload_handler(
    state: Dict[str, Any],
    chat_history: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], Dict[str, Any], str]:
    """Unload the current database and disable the query engine."""
    try:
        # Clear shared state
        state_manager.set_state("DATABASE_LOADED", False)
        state_manager.set_state("INDEX", None)
        state_manager.set_state("QUERY_ENGINE", None)
        state_manager.set_state("COLLECTION_NAME", None)

        # Reflect to Gradio state
        state = state if isinstance(state, dict) else {}
        state["database_loaded_gr"] = False
        state["use_query_engine_gr"] = False
        state["status_gr"] = "Database unloaded"

        append_to_chatbot(chat_history, "Database unloaded.", metadata={"role": "assistant"})
        logger.info("Database unloaded via UI")
        return chat_history, state, "Database unloaded."
    except Exception as e:
        logger.error(f"db_unload_handler failed: {e}")
        return (chat_history or []), (state or {}), f"Error: {e}"

# DB CFG END


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
        # ASSIGN METADATA AND IPC QUEUES
        self.plugin_name = plugin_name
        self.func = func
        self.args = args
        self.kwargs = kwargs

        # THREAD HANDLE + IPC
        self.thread: Optional[threading.Thread] = None
        self.output_queue: Queue = Queue()  # RESULTS/STREAMING
        self.input_queue: Queue = Queue()  # CONTROL COMMANDS (E.G., "stop")
        self.logger_queue: Queue = Queue()  # LOG RECORDS FROM WORKER
        self.stop_event = threading.Event()  # COOPERATIVE STOP SIGNAL

        # PER-SANDBOX QueueListener (TERMINAL/STDOUT ONLY)
        self.listener: Optional[QueueListener] = None

    def start(self) -> None:
        """Start the sandbox thread and logger listener.

        WHAT:
            - Spawn daemon thread to execute the plugin.
            - Start a QueueListener that drains records without emitting to CLI.
        WHY:
            - Keep CLI clean; all logs go to logger.py file handler (debug terminal tails them).
            - Still drain the queue to avoid buildup, without duplicating outputs.
        """
        # Start worker thread
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

        # Drain log records without printing to the main terminal
        # Root file handler (configured in logger.py) still captures everything.
        handlers = [logging.NullHandler()]
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
        # SIGNAL STOP
        self.stop_event.set()
        self.input_queue.put("stop")

        # JOIN WITH TIMEOUT FOR RELIABILITY
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

        # TEAR DOWN LISTENER
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
        # OBTAIN ROOT LOGGER VIA MODULE-LEVEL API (FIXES AttributeError)
        root_logger = logging.getLogger()

        # ATTACH A QueueHandler TO ROUTE RECORDS TO THIS SANDBOX'S QUEUE
        queue_handler: QueueHandler = QueueHandler(logger_queue)
        root_logger.addHandler(queue_handler)

        try:
            # DIAGNOSTIC — CONFIRM THREAD START
            try:
                logger.debug(
                    "Plugin thread started; executing '%s'.",
                    getattr(func, "__name__", "<anonymous>"),
                )
            except Exception:
                pass

            # EXECUTE PLUGIN (GENERATOR OR REGULAR)
            result = func(*args, **kwargs)

            if inspect.isgenerator(result):
                # STREAM CHUNKS WITH COOPERATIVE-STOP + COMMAND CHECK
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
                    # FORWARD OUTPUT
                    output_queue.put(chunk)
            else:
                # NON-GENERATOR RESULT PATH
                if stop_event.is_set():
                    try:
                        logger.info("Plugin stopped by event")
                    except Exception:
                        pass
                else:
                    output_queue.put(result)

            # SENTINEL — SIGNAL END OF STREAM
            output_queue.put(None)

        except Exception as exc:
            # SAFE ERROR TUNNEL — NO STACK TRACE LEAK TO UI BY DEFAULT
            try:
                logger.error("Plugin error: %s", exc)
            except Exception:
                pass
            output_queue.put(exc)
            output_queue.put(None)

        finally:
            # CRITICAL — ALWAYS REMOVE HANDLER TO PREVENT DUPLICATE LOGS/LEAKS
            try:
                root_logger.removeHandler(queue_handler)
                try:
                    queue_handler.close()
                except Exception:
                    pass
            except Exception:
                # NEVER RAISE ON CLEANUP
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
    global plugins
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

METADATA_TITLE = (
    f"Generated by {PROJECT_ROOT.name.upper()}-{hash(PROJECT_ROOT) % 10000}"
)
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

    Standard plugin signature (mandatory):
        def <plugin_name>(
            state: ProjectState,                           # read-only snapshot of all state vars
            chat_history: List[Dict[str, str]] | None = None,
            arg: str | None = None,                        # optional single CLI argument
        ) -> Generator[..., ...] | Iterable[...] | str
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
                        confirm = (
                            input("Confirm delete database? (y/n): ").strip().lower()
                        )
                        proceed = confirm == "y"
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
                state_dict["status_gr"] = (
                    "Model unloaded" if "success" in msg.lower() else "Unload attempted"
                )
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

    # Strict standardized call: provide a ProjectState snapshot, chat history, and optional arg
    kwargs_to_pass: Dict[str, Any] = {
        "state": app_state,
        "chat_history": chat_history,
        "arg": optional_arg,
    }

    # Validate developer errors early (clear message if signature is wrong)
    try:
        sig = inspect.signature(plugin_func)
        sig.bind_partial(**kwargs_to_pass)
    except TypeError as e:
        return (
            f"Error: Plugin '{plugin_name}' must implement "
            f"{plugin_name}(state, chat_history=None, arg=None): {e}"
        )

    sandbox = PluginSandbox(plugin_name, plugin_func, (), kwargs_to_pass)
    sandbox.start()
    with sandboxes_lock:
        active_sandboxes[plugin_name] = sandbox

    def _stream() -> Generator[Union[str, List[Dict[str, str]]], None, None]:
        """
        Stream sandbox output to caller with idle-timeout and cooperative stop.
        """
        try:
            last_activity = time.time()
            max_idle_seconds = 7200  # 2 hours inactivity ⇒ timeout

            while True:
                try:
                    item = sandbox.output_queue.get(timeout=0.05)
                except Empty:
                    if (time.time() - last_activity) > max_idle_seconds:
                        raise TimeoutError(f"Plugin '{plugin_name}' idle for >{max_idle_seconds}s")
                    continue

                last_activity = time.time()

                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item

                # Normalize to list[dict] and stamp metadata: plugin + title
                if isinstance(item, list) and all(isinstance(d, dict) for d in item):
                    msgs = item
                elif isinstance(item, dict):
                    msgs = [item]
                else:
                    try:
                        text = format_json_for_chat(item) if isinstance(item, (dict, list)) else str(item)
                    except Exception:
                        text = str(item)
                    msgs = [{"role": "assistant", "content": (text or "").strip(), "metadata": {}}]

                out: List[Dict[str, Any]] = []
                for m in msgs:
                    role = (m.get("role") or "assistant").strip().lower()
                    content = str(m.get("content", "")).strip()
                    if not content:
                        continue
                    meta = m.get("metadata") or {}
                    if not isinstance(meta, dict):
                        meta = {}
                    meta.setdefault("plugin", plugin_name)
                    meta.setdefault("title", METADATA_TITLE)
                    out.append({"role": role, "content": content, "metadata": meta})

                if out:
                    yield out
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
    # INPUT VALIDATION – PREVENT TYPE/SHAPE ERRORS EARLY.
    text = (user_text or "").strip()
    if not text:
        return
    if not isinstance(chat_history, list):
        raise TypeError("chat_history must be List[Dict[str, str]]")
    state = state if isinstance(state, dict) else {}

    # LOCAL EMITTER – APPEND/EXTEND ASSISTANT CHUNK AND YIELD.
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

    # DOT-COMMANDS – ASSISTANT-ONLY OUTPUT; NO USER BUBBLE HERE.
    if text.startswith("."):
        try:
            result = handle_command(text, chat_history, state=state, is_gradio=True)
        except Exception as exc:
            for out in _emit_chunk(f"Command error: {exc}"):
                yield out
            return

        if isinstance(result, list) and all(isinstance(d, dict) for d in result):
            for m in result:
                role = (m.get("role") or "assistant").strip().lower()
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                meta = m.get("metadata") or {}
                append_to_chatbot(chat_history, content, metadata={"role": role, **meta})
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
                    meta = m.get("metadata") or {}
                    append_to_chatbot(chat_history, content, metadata={"role": role, **meta})
                    yield list(chat_history)
            else:
                for out in _emit_chunk(str(item)):
                    yield out
        return

    # QUERY ENGINE – PREFER STREAM; SAFE FALLBACK TO LLM.
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

    if isinstance(llm, _HFL_TYPE):
        from transformers import TextIteratorStreamer

        # SUPPORT BOTH PRIVATE AND PUBLIC ATTR NAMES
        model = getattr(llm, "_model", None) or getattr(llm, "model", None)
        tok = getattr(llm, "_tokenizer", None) or getattr(llm, "tokenizer", None)
        if model is None or tok is None:
            for out in _emit_chunk("Error: HuggingFaceLLM missing model/tokenizer."):
                yield out
            return

        # Prefer tokenizer's chat template when available to induce correct EOS usage
        try:
            if hasattr(tok, "apply_chat_template"):
                messages = []
                if sys_pfx:
                    messages.append({"role": "system", "content": sys_pfx})
                messages.append({"role": "user", "content": text})
                tmpl = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if isinstance(tmpl, str) and tmpl.strip():
                    prompt = tmpl
        except Exception as e:
            logger.debug(f"(PI) chat template apply failed: {e}")

        device = getattr(
            model,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
        ids = enc.get("input_ids")
        if ids is None or ids.numel() == 0 or ids.shape[-1] == 0:
            # NON-EMPTY GUARD — PREVENTS CACHE-POSITION BUG PATH.
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

        # MOVE TO DEVICE W/O DTYPE CAST (EMBEDDING EXPECTS LONG)
        ids = ids.to(device)
        attn = attn.to(device)
        if ids.dtype is not torch.long:
            ids = ids.long()
        if attn.dtype is not torch.long:
            attn = attn.long()

        # Pad id (no global mutation)
        try:
            if (
                getattr(tok, "pad_token_id", None) is None
                and getattr(tok, "eos_token_id", None) is not None
            ):
                tok.pad_token_id = int(tok.eos_token_id)
        except Exception:
            pass

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

        # ENFORCE CONFIGURED TOKEN BUDGET (NO 256 FALLBACK) + SYNC WRAPPER CAPS
        target_tokens = int(MAX_NEW_TOKENS)
        try:
            if hasattr(llm, "max_new_tokens"):
                llm.max_new_tokens = target_tokens
            # SANITIZE WRAPPER GENERATE_KWARGS TO AVOID DUPLICATE max_new_tokens MERGE IN LlamaIndex
            if hasattr(llm, "generate_kwargs") and isinstance(
                llm.generate_kwargs, dict
            ):
                # REMOVE POTENTIAL CONFLICT KEYS THAT LlamaIndex MAY ALSO SET EXPLICITLY
                llm.generate_kwargs.pop("max_new_tokens", None)
                llm.generate_kwargs.pop("max_length", None)
        except Exception:
            pass

        gen_kwargs = {
            "input_ids": ids,
            "attention_mask": attn,
            "max_new_tokens": target_tokens,
            "do_sample": True,
            "top_p": float(TOP_P),
            "temperature": float(TEMPERATURE),
            "use_cache": False,
            "pad_token_id": int(
                getattr(tok, "pad_token_id", getattr(tok, "eos_token_id", 0))
            ),
            "streamer": streamer,
        }

        # Dynamic EOS: stop as soon as EOS is produced (if available)
        try:
            eos_tok = getattr(tok, "eos_token_id", None)
            if eos_tok is None:
                eos_tok = getattr(tok, "pad_token_id", None)
            if eos_tok is not None:
                gen_kwargs["eos_token_id"] = int(eos_tok)
        except Exception:
            pass

        # Anti-repetition controls from config.toml
        if isinstance(TOP_K, int) and TOP_K > 0:
            gen_kwargs["top_k"] = int(TOP_K)
        if (
            isinstance(REPETITION_PENALTY, (int, float))
            and float(REPETITION_PENALTY) != 1.0
        ):
            gen_kwargs["repetition_penalty"] = float(REPETITION_PENALTY)
        if isinstance(NO_REPEAT_NGRAM_SIZE, int) and NO_REPEAT_NGRAM_SIZE > 0:
            gen_kwargs["no_repeat_ngram_size"] = int(NO_REPEAT_NGRAM_SIZE)

        def _run():
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

        from gia.core.utils import stream_generate as _gen_safe

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
                delta = (
                    ev
                    if isinstance(ev, str)
                    else (getattr(ev, "delta", None) or getattr(ev, "text", ""))
                )
                if not delta:
                    continue
                streamed = True
                for out in _emit_chunk(delta):
                    yield out
            if streamed:
                return
        except Exception as exc:
            logger.warning(
                f"[PI] stream_complete failed; switching to complete(). err={exc}"
            )

    complete = getattr(llm, "complete", None)
    if callable(complete):
        try:
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


def update_model_dropdown(directory: str) -> Tuple[Any, List[Dict[str, str]]]:
    """Update model dropdown with available folders.

    Args:
        directory (str): Directory to scan for models.

    Returns:
        Tuple[Any, List[Dict[str, str]]]: gr.update(...) for dropdown and chat message.
    """
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        return gr.update(choices=[]), [
            {
                "role": "assistant",
                "content": f"Error: Directory '{directory}' does not exist.",
            }
        ]

    folders = list_all_folders(directory)
    names = [os.path.basename(os.path.normpath(p)) for p in folders]
    try:
        state_manager.set_state("LOCAL_MODELS_MAP", dict(zip(names, folders)))
    except Exception as e:
        logger.debug(f"(UI) Failed to store LOCAL_MODELS_MAP: {e}")

    return gr.update(choices=names, value=(names[0] if names else None)), [
        {"role": "assistant", "content": f"Found {len(names)} model folders."}
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
        Tuple[str, Dict]: Status message (Markdown) and updated state.
    """
    try:
        unload_model()
        logger.info("Previous model unloaded successfully")
    except Exception as e:
        logger.warning(f"Unload previous model failed: {str(e)}. Proceeding with load.")

    if not isinstance(state, dict):
        logger.error("Invalid state: expected dictionary")
        state = {}
        return "Error: Invalid state provided", state

    state_manager.set_state("MODE", mode)
    logger.info(f"Model mode set to {mode} globally")

    config: Dict[str, Any] = {}
    if mode == "Local":
        selected = state.get("model_path_gr", "") or CONFIG["MODEL_PATH"]
        resolved_path: Optional[str] = None
        try:
            mapping = state_manager.get_state("LOCAL_MODELS_MAP", {}) or {}
            if isinstance(mapping, dict) and selected in mapping:
                resolved_path = mapping[selected]
        except Exception as e:
            logger.debug(f"(UI) LOCAL_MODELS_MAP resolution failed: {e}")
        config["model_path"] = resolved_path or selected
    elif mode == "HuggingFace":
        config["model_name"] = state.get(
            "hf_model_name", "mistralai/Mistral-7B-Instruct-v0.2"
        )
    elif mode == "OpenAI":
        config["model_name"] = state.get("openai_model_name", "gpt-4o")
    elif mode == "OpenRouter":
        config["model_name"] = state.get("openrouter_model_name", "x-ai/grok-3-mini")
    else:
        error_msg = f"Invalid mode: {mode}"
        logger.error(error_msg)
        state["status_gr"] = error_msg
        return error_msg, state

    try:
        msg, llm = load_llm(mode, config)
        if not llm:
            raise RuntimeError(msg)

        state_manager.set_state("LLM", llm)
        raw_name = config.get("model_name", config.get("model_path", "Unknown Model"))
        display_name = (
            os.path.basename(os.path.normpath(raw_name)) if mode == "Local" else str(raw_name)
        )
        state_manager.set_state("MODEL_NAME", display_name)
        state_manager.set_state("MODEL_PATH", config.get("model_path", ""))

        # Project name/version for header
        proj_name, proj_ver = "GIA", "0.0.0"
        try:
            import tomllib  # Python 3.11+ stdlib
            py_path = PROJECT_ROOT.parent.parent / "pyproject.toml"
            if py_path.exists():
                with py_path.open("rb") as f:
                    data = tomllib.load(f)
                if isinstance(data, dict):
                    proj = data.get("project", {}) or {}
                    if isinstance(proj, dict):
                        proj_name = str(proj.get("name", proj_name))
                        proj_ver = str(proj.get("version", proj_ver))
        except Exception as e:
            logger.debug(f"pyproject read failed: {e}")

        # Hardware detection: show either GPU or CPU (never both)
        cpu_name = "Unknown CPU"
        try:
            cpuinfo_path = "/proc/cpuinfo"
            if os.path.exists(cpuinfo_path):
                with open(cpuinfo_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.lower().startswith("model name"):
                            parts = line.split(":", 1)
                            if len(parts) == 2:
                                cpu_name = parts[1].strip()
                                break
            if cpu_name == "Unknown CPU":
                import platform
                cpu_name = platform.processor() or cpu_name
        except Exception as e:
            logger.debug(f"CPU name detection failed: {e}")

        gpu_name: Optional[str] = None
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except Exception as e:
            logger.debug(f"GPU name detection failed: {e}")

        # Status header as Markdown H3
        if mode == "Local":
            loaded_on = f"Loaded on: GPU: {gpu_name}" if gpu_name else f"Loaded on: CPU: {cpu_name}"
            status_header = f"### {proj_name} v.{proj_ver} - Local model: {display_name} - {loaded_on}"
        else:
            # Online providers: concise header; hardware not applicable
            status_header = f"### {proj_name} v.{proj_ver} - {mode} model: {display_name}"

        # Database Config — reflect live active collection (not config)
        active_collection: Optional[str] = None
        try:
            active_collection = state_manager.get_state("COLLECTION_NAME") or None
        except Exception:
            active_collection = None
        if active_collection is None:
            # Fallback: show first available collection if none selected yet
            try:
                client = _ensure_client()
                cols = client.list_collections()
                if cols:
                    c0 = cols[0]
                    active_collection = getattr(c0, "name", None) or (
                        c0.get("name") if isinstance(c0, dict) else None
                    )
            except Exception:
                active_collection = None

        db_md = (
            f"\n\n#### Database Config\n"
            f"- Base path: `{DB_PATH}`\n"
            f"- Collection: `{active_collection or 'none'}`"
        )
        status_header += db_md

        state["status_gr"] = status_header
        state["model_loaded_gr"] = True
        logger.info(f"Loaded LLM type: {type(llm).__name__} for mode {mode}")
        return status_header, state

    except Exception as e:
        error_msg = f"Failed to load {mode} model: {str(e)}"
        logger.error(error_msg)
        state["status_gr"] = error_msg
        state["model_loaded_gr"] = False
        return error_msg, state



def db_load_handler(
    selected_name: str,
    state: Dict[str, Any],
    chat_history: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], Dict[str, Any], str]:
    """Set the active DB to the selected name and load the query engine.

    Returns:
        Tuple[chat_history, updated_state, status_markdown]
    """
    try:
        # 1) Switch active DB folder ('(base)' => DB_PATH)
        msg1 = ui_load_db(selected_name)

        # 2) Build/load the DB + QueryEngine with current LLM
        updated_chat, updated_state = handle_load_button(state)
        status_line = (updated_state.get("status_gr") or "").strip()
        combined = msg1 if msg1 else ""
        if status_line:
            combined = f"{combined}\n{status_line}" if combined else status_line

        return updated_chat, updated_state, (combined or "Database action completed.")
    except Exception as e:
        logger.error(f"db_load_handler failed: {e}")
        return chat_history or [], (state or {}), f"Error: {e}"


def chat_fn(
    message: str,
    chat_history: List[Dict[str, str]],
    state: Dict[str, Any],
) -> "Generator[Tuple[str, Any, Dict[str, Any]], None, None]":
    """
    GRADIO CHAT BRIDGE (MINIMAL, RELIABLE).

    - Appends user bubble (except dot-commands), clears textbox.
    - Forwards process_input() stream as Chatbot updates.
    - Yields: (textbox_value, gr.update(...), state).
    """
    chat_history = chat_history if isinstance(chat_history, list) else []
    state = state if isinstance(state, dict) else {}

    msg = (message or "").strip()
    if not msg:
        yield "", gr.update(value=list(chat_history)), state
        return

    is_command = msg.startswith(".")
    if not is_command:
        append_to_chatbot(chat_history, msg, metadata={"role": "user"})
        yield "", gr.update(value=list(chat_history)), state

    for updated_history in process_input(msg, state, chat_history):
        yield "", gr.update(value=updated_history), state


def handle_load_button(
    state: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Handle load button: ensure correct local path resolution, then load DB.

    Returns updated chatbot history and UI state, never raising to UI.
    """
    # ALWAYS RETURN A LIST (UI STABILITY)
    chat_history: List[Dict[str, str]] = []

    app_state = load_state()
    state_dict = (
        state
        if isinstance(state, dict)
        else {
            "use_query_engine_gr": True,
            "status_gr": "",
            "model_path_gr": None,
            "database_loaded_gr": False,
        }
    )

    # NO-OP IF DB IS ALREADY LOADED (DEDUPLICATION)
    if state_dict.get("database_loaded_gr", False):
        append_to_chatbot(
            chat_history, "Database already loaded.", metadata={"role": "assistant"}
        )
        return chat_history, state_dict

    mode = str(state_dict.get("mode", "Local"))
    model_path_gr = state_dict.get("model_path_gr")

    try:
        if app_state.LLM is None:
            config: Dict[str, Any] = {}
            display_name: Optional[str] = None

            if mode == "Local":
                resolved_path: Optional[str] = None
                mapping: Dict[str, str] = {}
                try:
                    mapping = state_manager.get_state("LOCAL_MODELS_MAP", {}) or {}
                except Exception:
                    mapping = {}

                if isinstance(model_path_gr, str) and model_path_gr in mapping:
                    resolved_path = mapping[model_path_gr]
                elif isinstance(app_state.MODEL_PATH, str) and app_state.MODEL_PATH:
                    resolved_path = app_state.MODEL_PATH
                elif isinstance(model_path_gr, str) and model_path_gr:
                    resolved_path = model_path_gr

                if not resolved_path or not os.path.isdir(resolved_path):
                    raise ValueError(
                        "No valid local model directory resolved. "
                        "Use 'Scan Directory' and 'Confirm Model' first."
                    )

                config = {"model_path": resolved_path}
                display_name = os.path.basename(os.path.normpath(resolved_path))

            elif mode == "HuggingFace":
                mn = (
                    state_dict.get("hf_model_name")
                    or "mistralai/Mistral-7B-Instruct-v0.2"
                )
                config = {"model_name": mn}
                display_name = str(mn)

            elif mode == "OpenAI":
                mn = state_dict.get("openai_model_name") or "gpt-4o"
                config = {"model_name": mn}
                display_name = str(mn)

            elif mode == "OpenRouter":
                mn = state_dict.get("openrouter_model_name") or "x-ai/grok-3-mini"
                config = {"model_name": mn}
                display_name = str(mn)

            else:
                raise ValueError(f"Invalid mode: {mode}")

            logger.info(
                "(LOAD) Loading model for mode=%s; target=%s",
                mode,
                config.get("model_path") or config.get("model_name"),
            )
            msg, llm = load_llm(mode, config)
            if llm is None:
                append_to_chatbot(
                    chat_history,
                    f"Error: Failed to load model: {msg}",
                    metadata={"role": "assistant"},
                )
                return chat_history, state_dict

            state_manager.set_state("LLM", llm)
            state_manager.set_state("MODEL_NAME", display_name)
            if mode == "Local":
                state_manager.set_state("MODEL_PATH", config["model_path"])
            else:
                state_manager.set_state("MODEL_PATH", None)

        result = load_database_wrapper(state_dict)
        message = str(result).strip()
        if message:
            append_to_chatbot(chat_history, message, metadata={"role": "assistant"})

        ok = not message.lower().startswith("error")
        state_dict["use_query_engine_gr"] = ok
        state_dict["status_gr"] = "Database loaded" if ok else "Database load failed"
        state_dict["database_loaded_gr"] = ok

    except Exception as e:
        logger.error("Error during load button handling: %s", e)
        append_to_chatbot(
            chat_history, f"Error during loading: {e}", metadata={"role": "assistant"}
        )

    return chat_history, state_dict


def launch_app(args: argparse.Namespace) -> None:
    """Launch Gradio application with fixed sidebars and full‑width chat."""
    launch_event = threading.Event()

    def threaded_launch() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        with gr.Blocks(
            title=f"{tag()} - UI Gradio Interface",
            delete_cache=(60, 60),
            theme=gr.themes.Base(
                primary_hue="zinc",
                secondary_hue="stone",
                neutral_hue="stone",
                spacing_size="md",
                radius_size="lg",
                text_size="md",
                # Fonts
                font=[gr.themes.GoogleFont("Roboto"), "ui-sans-serif", "system-ui", "sans-serif"],
                font_mono=[gr.themes.GoogleFont("Google Sans Code"), "ui-monospace", "monospace"],
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
            css="""
            /* Google Fonts (fast swap) */
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Google+Sans+Code:wght@400&display=swap');

            /* Base */
            html, body { background:#000 !important; height:100%; }
            .gradio-container { background:#000 !important; color:#fff !important; min-height:100%;
            font-family:'Roboto', ui-sans-serif, system-ui, sans-serif !important; }

            /* Fixed sidebars */
            :root { --gia-sidebar-w: clamp(240px, 15vw, 300px); --gia-gap: 10px; }
            #left-sidebar  { position:fixed !important; top:12px; bottom:12px; left: 8px;
                            width:var(--gia-sidebar-w); overflow-y:auto; z-index:5; }
            #right-sidebar { position:fixed !important; top:12px; bottom:12px; right:8px;
                            width:var(--gia-sidebar-w); overflow-y:auto; z-index:5; }

            /* Ensure left menus never line up side-by-side */
            #left-stack { display:flex !important; flex-direction:column !important; gap:16px !important; width:100% !important; }
            #left-stack > * { width:100% !important; }

            /* Slightly smaller menus than chat body */
            #left-sidebar, #right-sidebar { font-size:0.95rem !important; }

            /* Center column uses ALL remaining width (no clamps) */
            #center-col {
            margin-left:  calc(var(--gia-sidebar-w) + var(--gia-gap)) !important;
            margin-right: calc(var(--gia-sidebar-w) + var(--gia-gap)) !important;
            max-width:none !important;
            }
            #center-col .gr-block, #center-col .gr-form, #center-col .gr-panel, #center-col .gr-row, #center-col > * {
            max-width:none !important; width:100% !important;
            }

            /* Chatbot base + dark */
            [data-testid="chatbot"] { background:#0b0b0b !important; }
            [data-testid="chatbot"], [data-testid="chatbot"] * { color:#fff !important; }
            .message.bot, .bubble.bot   { background:#121212 !important; border:none !important; }
            .message.user, .bubble.user { background:#141414 !important; border:none !important; }

            /* Chat body text (≈18px) */
            #center-col [data-testid="chatbot"] .prose {
            font-size:1.125rem !important; line-height:1.75 !important;
            max-width:none !important; font-family:'Roboto', ui-sans-serif, system-ui, sans-serif !important;
            }

            /* Code font: Google Sans Code Regular 400 at 16px */
            #center-col [data-testid="chatbot"] .prose pre,
            #center-col [data-testid="chatbot"] .prose code,
            #center-col [data-testid="chatbot"] pre code {
            background:#0f0f0f !important; color:#fff !important;
            font-family:'Google Sans Code','Consolas',ui-monospace,monospace !important;
            font-weight:400 !important; font-size:16px !important; line-height:1.6 !important;
            }

            /* CRITICAL: remove all max-width clamps inside Chatbot (Gradio/Tailwind wrappers) */
            #center-col [data-testid="chatbot"] .message,
            #center-col [data-testid="chatbot"] .bubble { width:100% !important; max-width:100% !important; }

            /* Markdown container & its first child (gr.Markdown wrapper) */
            #center-col [data-testid="chatbot"] .gr-markdown, 
            #center-col [data-testid="chatbot"] .gr-markdown > * {
            max-width:none !important; width:100% !important;
            }

            /* Tailwind common utility clamps used by Gradio around chat */
            #center-col [data-testid="chatbot"] .max-w-prose,
            #center-col [data-testid="chatbot"] .md\\:max-w-prose,
            #center-col [data-testid="chatbot"] .max-w-2xl,
            #center-col [data-testid="chatbot"] .md\\:max-w-2xl,
            #center-col [data-testid="chatbot"] .max-w-3xl,
            #center-col [data-testid="chatbot"] .md\\:max-w-3xl,
            #center-col [data-testid="chatbot"] .max-w-screen-md,
            #center-col [data-testid="chatbot"] .max-w-screen-lg,
            #center-col [data-testid="chatbot"] .max-w-screen-xl,
            #center-col [data-testid="chatbot"] .md\\:max-w-screen-md,
            #center-col [data-testid="chatbot"] .md\\:max-w-screen-lg,
            #center-col [data-testid="chatbot"] .md\\:max-w-screen-xl {
            max-width:100% !important;
            }

            /* Nuke any other max-w-* utilities, and inline style max-width, inside chatbot */
            #center-col [data-testid="chatbot"] [class*="max-w-"] { max-width:100% !important; }
            #center-col [data-testid="chatbot"] *[style*="max-width"] { max-width:100% !important; }

            /* Inputs/buttons */
            textarea, input, select { background:#121212 !important; color:#fff !important; border:none !important; }
            button { background:#141414 !important; color:#fff !important; border:none !important; }
            button:hover { background:#1a1a1a !important; }

            /* Narrow screens: stack */
            @media (max-width: 1200px){
            #left-sidebar,#right-sidebar { position:static !important; width:auto; height:auto; }
            #center-col { margin-left:0 !important; margin-right:0 !important; }
            }
            """,
        ) as demo:
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

            with gr.Row():
                # ---------------------- LEFT SIDEBAR (fixed) ----------------------
                with gr.Column(scale=2, elem_id="left-sidebar"):
                    with gr.Column(elem_id="left-stack"):
                        with gr.Accordion("Mode & Models", open=True):
                            mode_radio = gr.Radio(
                                choices=["Local", "HuggingFace", "OpenAI", "OpenRouter"],
                                value="Local",
                                label="Model Mode",
                            )
                            warning_md = gr.Markdown("", visible=False)

                            # Local: Models Directory → Scan Directory → Select a Model → Confirm
                            with gr.Row(visible=True) as local_row:
                                with gr.Column(scale=6):
                                    models_dir_input = gr.Textbox(
                                        label="Models Directory", value=CONFIG["MODEL_PATH"]
                                    )
                                    scan_dir_button = gr.Button("Scan Directory")
                                    folder_dropdown = gr.Dropdown(choices=[], label="Select a Model")
                                    confirm_local_button = gr.Button("Confirm")

                            # HF UI
                            with gr.Row(visible=False) as hf_row:
                                with gr.Column(scale=5):
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
                                with gr.Column(scale=1):
                                    confirm_hf_button = gr.Button("Confirm")

                            # OpenAI UI
                            with gr.Row(visible=False) as openai_row:
                                with gr.Column(scale=5):
                                    openai_model_dropdown = gr.Dropdown(
                                        choices=fetch_openai_models(),
                                        label="OpenAI Model",
                                        value="gpt-4o",
                                    )
                                with gr.Column(scale=1):
                                    confirm_openai_button = gr.Button("Confirm")

                            # OpenRouter UI
                            with gr.Row(visible=False) as openrouter_row:
                                with gr.Column(scale=5):
                                    openrouter_model_text = gr.Textbox(
                                        label="OpenRouter Model",
                                        value="x-ai/grok-3-mini",
                                        placeholder="Enter OpenRouter model name (e.g., x-ai/grok-3-mini)",
                                    )
                                with gr.Column(scale=1):
                                    confirm_openrouter_button = gr.Button("Confirm")

                        # Database: Databases → Load → Unload → Refresh → New Database → Create Database
                        with gr.Accordion("Database", open=False):
                            db_dropdown = gr.Dropdown(choices=[], value=None, label="Databases")
                            db_load_btn = gr.Button("Load Database")
                            db_unload_btn = gr.Button("Unload Database")
                            db_refresh_btn = gr.Button("Refresh")
                            db_new_name = gr.Textbox(
                                label="New Database (subfolder of DB_PATH)", placeholder="my-project"
                            )
                            db_create_btn = gr.Button("Create Database")

                # ---------------------- CENTER (CHAT, dynamic) ----------------------
                with gr.Column(scale=8, elem_id="center-col"):
                    chatbot = gr.Chatbot(
                        type="messages",
                        value=[],
                        height=None,
                        allow_tags=["think", "answer"],
                        render_markdown=True,
                        show_label=False,
                        layout="bubble",
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Enter your message or command (e.g., .info, .load, .create, .delete, .unload)",
                            lines=1,
                            scale=10,
                            show_label=False,
                        )
                        send_btn = gr.Button("➤", scale=1)

                # ---------------------- RIGHT SIDEBAR (fixed) ----------------------
                with gr.Column(scale=2, elem_id="right-sidebar"):
                    with gr.Accordion("Plugins", open=True):
                        if not plugins:
                            gr.Markdown("No plugins loaded.")
                        else:
                            def plugin_handler_simple(
                                name: str,
                                chat_history: List[Dict[str, str]],
                                state: Dict[str, Any],
                            ) -> Generator[Tuple[Any, Dict[str, Any]], None, None]:
                                """Run a plugin by delegating to '.<plugin>' command, streaming results."""
                                import gradio as gr
                                try:
                                    state_dict = state.value if hasattr(state, "value") else (state if isinstance(state, dict) else {})
                                    hist = chat_history if isinstance(chat_history, list) else []
                                    cmd = f".{name}"
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

                    with gr.Accordion("Status", open=True):
                        model_name_display = gr.Markdown(
                            "Status will appear here.", elem_id="model-name-display"
                        )

            # ===== Wiring =====

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
                            inference="warm",
                            pipeline_tag="text-generation",
                            limit=100,
                            sort="downloads",
                            direction=-1,
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
                            "Mistral-7B-Instruct-v0.2",
                            "Meta-Llama-3-8B-Instruct",
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
                lambda model, state: ((state.value.update({"hf_model_name": model}) or state) if hasattr(state, "value") else (state.update({"hf_model_name": model}) or state)),
                inputs=[hf_model_dropdown, session_state],
                outputs=[session_state],
            )
            openai_model_dropdown.change(
                lambda model, state: ((state.value.update({"openai_model_name": model}) or state) if hasattr(state, "value") else (state.update({"openai_model_name": model}) or state)),
                inputs=[openai_model_dropdown, session_state],
                outputs=[session_state],
            )
            openrouter_model_text.change(
                lambda model, state: ((state.value.update({"openrouter_model_name": model}) or state) if hasattr(state, "value") else (state.update({"openrouter_model_name": model}) or state)),
                inputs=[openrouter_model_text, session_state],
                outputs=[session_state],
            )

            scan_dir_button.click(
                fn=update_model_dropdown,
                inputs=[models_dir_input],
                outputs=[folder_dropdown, chatbot],
            )

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
                state_dict = state.value if hasattr(state, "value") else (state if isinstance(state, dict) else {})
                logger.debug(f"Confirming model for mode: {mode}, state: {state_dict}")
                state_dict["model_path_gr"] = (selected_folder if mode == "Local" else None)
                state_dict["hf_model_name"] = (hf_model if mode == "HuggingFace" else state_dict.get("hf_model_name", "deepseek-ai/DeepSeek-R1"))
                state_dict["openai_model_name"] = (openai_model if mode == "OpenAI" else state_dict.get("openai_model_name", "gpt-3.5-turbo"))
                state_dict["openrouter_model_name"] = (openrouter_model if mode == "OpenRouter" else state_dict.get("openrouter_model_name", "x-ai/grok-3-mini"))
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

            confirm_local_button.click(
                fn=confirm_model_handler,
                inputs=[folder_dropdown, session_state, mode_radio, hf_model_dropdown, openai_model_dropdown, openrouter_model_text, chatbot],
                outputs=[chatbot, session_state, model_name_display],
            )
            confirm_hf_button.click(
                fn=confirm_model_handler,
                inputs=[folder_dropdown, session_state, mode_radio, hf_model_dropdown, openai_model_dropdown, openrouter_model_text, chatbot],
                outputs=[chatbot, session_state, model_name_display],
            )
            confirm_openai_button.click(
                fn=confirm_model_handler,
                inputs=[folder_dropdown, session_state, mode_radio, hf_model_dropdown, openai_model_dropdown, openrouter_model_text, chatbot],
                outputs=[chatbot, session_state, model_name_display],
            )
            confirm_openrouter_button.click(
                fn=confirm_model_handler,
                inputs=[folder_dropdown, session_state, mode_radio, hf_model_dropdown, openai_model_dropdown, openrouter_model_text, chatbot],
                outputs=[chatbot, session_state, model_name_display],
            )

            db_load_btn.click(
                fn=db_load_handler,
                inputs=[db_dropdown, session_state, chatbot],
                outputs=[chatbot, session_state, model_name_display],
            )
            db_unload_btn.click(
                fn=db_unload_handler,
                inputs=[session_state, chatbot],
                outputs=[chatbot, session_state, model_name_display],
            )
            db_refresh_btn.click(
                fn=ui_refresh_db_list,
                inputs=[],
                outputs=[db_dropdown, model_name_display],
            )
            db_create_btn.click(
                fn=ui_create_db,
                inputs=[db_new_name],
                outputs=[db_dropdown, model_name_display, db_new_name],
            )
            demo.load(fn=ui_refresh_db_list, inputs=[], outputs=[db_dropdown, model_name_display])

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

        demo.queue()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            debug=True,
            inbrowser=True,
            quiet=True,
        )
        launch_event.set()

    try:
        if not args.cli:
            launch_thread = threading.Thread(target=threaded_launch, daemon=True)
            launch_thread.start()
            time.sleep(1)
            logger.info("Gradio app launched in background thread. Main terminal freed for CLI.")
        cli_loop()
    except Exception as e:
        logger.error(f"Failed to launch Gradio in thread or start CLI: {str(e)}")
        raise


def tag() -> str:
    """Return a timestamped prefix including current mode, based on shared state."""
    app_state: ProjectState = load_state()
    llm = getattr(app_state, "LLM", None)
    mode = (getattr(app_state, "MODE", None) or "").strip().lower()
    if llm is None:
        mode_str = "idle"
    elif mode == "local":
        mode_str = "local"
    else:
        mode_str = "online"
    current_time = datetime.now().strftime("[%H:%M:%S]")
    return f"{current_time} GIA@{mode_str}: "


def cli_prompt() -> str:
    """
    Generate a colored terminal prompt 'GIA@<mode>:~$' without including the path.

    - Mode: idle (no LLM) | local (MODE == Local) | online (remote providers)
    - Robust to state access failures; never raises.
    """
    try:
        snap = load_state()
        llm = getattr(snap, "LLM", None)
        mode = (getattr(snap, "MODE", None) or "").strip().lower()
    except Exception:
        llm, mode = None, ""

    if llm is None:
        mode_str = "idle"
    elif mode == "local":
        mode_str = "local"
    else:
        mode_str = "online"

    user = "GIA"
    host = mode_str
    return f"{Fore.GREEN}{user}@{host}{Fore.RESET}:~$ "


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

        # isolate a live panel per response to avoid flicker across turns
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
                # Do not raise; generators may yield control sentinels in some plugin paths.

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
    parser.add_argument(
        "--cli", action="store_true", help="Run in CLI-only mode without Gradio"
    )
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
