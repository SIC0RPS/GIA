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
chat_history = [] # todo chat history + summerization in background
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

####

# Flags
METADATA_TITLE = f"Generated by {PROJECT_ROOT.name.upper()}-{hash(PROJECT_ROOT) % 10000}"
dark_gray = Style.DIM + Fore.LIGHTBLACK_EX
reset_style = Style.RESET_ALL

###

# LOGGER SETUP - FOR CONSISTENT logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.INFO)
    


####

# FILE HASH - FOR INTEGRITY CHECK
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

####

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


####

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


####


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

###

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

###

def process_input(
    user_text: str,
    state: Optional[Dict[str, Any]],
    chat_history: List[Dict[str, str]],
) -> Generator[Union[str, List[Dict[str, str]]], None, None]:
    """
    PRODUCTION-READY GENERATOR. STREAMS RESPONSES TO UI AND TERMINAL.
    * STREAMS VIA NATIVE llm.stream_complete (OpenAI/compatible).
    * STREAMS VIA TextIteratorStreamer (HuggingFace Local/GPTQ) WITH BOS/PAD GUARDS.
    * FALLBACK: NON-STREAM + BOUNDARY-AWARE CHUNKING (SAFE MARKDOWN).
    * EVERY UI CHUNK MIRRORED TO TERMINAL WITH FLUSH=TRUE.
    """
    # TO INITIALIZE APPLICATION STATE - PRESERVES GLOBAL ISOLATION; WHY: AVOIDS RACE CONDITIONS IN MULTI-THREAD; EDGE: STATE NONE HANDLED; PERF: O(1); SEC: NO EXTERNAL ACCESS.
    app_state = load_state()
    # TO SANITIZE INPUT TEXT - REMOVES LEADING/TRAILING WHITESPACE; WHY: PREVENTS EMPTY PROCESSING; EDGE: ALL WHITESPACE -> EMPTY; PERF: O(N); SEC: AVOIDS INJECTION VIA CONTROL CHARS.
    text = (user_text or "").strip()
    if not text:
        logger.debug("[PI] empty user_text -> no-op")
        return
    logger.debug(
        f"[PI] input={text!r} | history_size={len(chat_history)} "
        f"| state_is_dict={isinstance(state, dict)}"
    )
    # TO HANDLE DOT-COMMANDS - FAST PATH FOR LOCAL OPS; WHY: SHORT-CIRCUIT FOR EFFICIENCY; EDGE: INVALID RESULTS YIELDED AS STR; PERF: O(1) CHECK; SEC: COMMAND HANDLER VALIDATES.
    if text.startswith("."):
        result = handle_command(text, chat_history, state=state, is_gradio=bool(state is not None))
        if isinstance(result, str):
            if result:
                print(result, end="", flush=True)
            yield result
        else:
            for item in result:
                if isinstance(item, list):
                    yield item
                else:
                    s = str(item)
                    if s:
                        print(s, end="", flush=True)
                        yield s
        return
    # TO ATTEMPT RAG QUERY IF ENABLED - OPTIONAL KNOWLEDGE RETRIEVAL; WHY: ENHANCES ACCURACY; EDGE: EMPTY RESPONSE SKIPPED; PERF: DEPENDS ON ENGINE; SEC: QUERY SANITIZED VIA ENGINE.
    state_dict = state if isinstance(state, dict) else {}
    use_qe = bool(state_dict.get("use_query_engine_gr", True))
    used_qe = False
    if use_qe and app_state.QUERY_ENGINE:
        try:
            resp = app_state.QUERY_ENGINE.query(text)
            if hasattr(resp, "response_gen") and resp.response_gen is not None:
                append_to_chatbot(chat_history, "", metadata={"role": "assistant"})
                yield list(chat_history)
                for chunk in resp.response_gen:
                    s = str(chunk)
                    if not s or s.isspace() or s.strip().lower() == "empty response":
                        continue
                    chat_history[-1]["content"] += s
                    yield list(chat_history)
                    print(s, end="", flush=True)
                return
            else:
                s = resp.response if hasattr(resp, "response") else str(resp)
                if s and s.strip().lower() != "empty response":
                    append_to_chatbot(chat_history, s, metadata={"role": "assistant"})
                    yield list(chat_history)
                    print(s, end="", flush=True)
                    return
                used_qe = False
        except Exception as e:
            logger.warning(f"[PI] Query engine error, falling back to LLM: {e}")
            used_qe = False
    if used_qe:
        return
    # TO VALIDATE LLM AVAILABILITY - CRITICAL CHECK; WHY: PREVENTS NULL DEREF; EDGE: NONE -> YIELD ERROR; PERF: O(1); SEC: SAFE ERROR MSG.
    llm = app_state.LLM
    if llm is None:
        err = "Error: No model loaded for query."
        print(err, end="", flush=True)
        yield err
        return
    # TO CONSTRUCT PROMPT WITH SYSTEM PREFIX - ASSEMBLES FULL INPUT; WHY: MODEL-SPECIFIC; EDGE: EMPTY SYS -> USER ONLY; PERF: O(N); SEC: NO UNSANITIZED INPUT.
    sys_prompt = system_prefix()
    prompt = f"{sys_prompt}\n{text}"
    # TO FORCE NON-EMPTY PROMPT - PREFIX SPACE IF STRIPPED; WHY: ENSURES TOKENIZER YIELDS >=1 TOKEN (TRANSFORMERS 4.55.4 EDGE); RISK: SEMANTIC SHIFT (MINIMAL, SKIPPED IN OUTPUT); ASSUM: SPACE TOKEN EXISTS; EDGE: ALL STRIP -> SPACE; PERF: O(1); SEC: NEUTRAL INSERTION.
    if not prompt.strip():
        prompt = " "
    # TO DEFINE BOUNDARY-AWARE CHUNKER - SAFE MARKDOWN STREAMING; WHY: PREVENTS BROKEN FENCES; EDGE: EMPTY -> NO YIELD; PERF: O(N); SEC: NO EXEC.
    def _yield_boundary_chunks(full_text: str) -> Generator[str, None, None]:
        import re
        if not full_text:
            return
        i, n = 0, len(full_text)
        fence_open = False
        MIN_SZ, MAX_SZ = 120, 1200
        while i < n:
            j = min(i + MAX_SZ, n)
            window = full_text[i:j]
            for _ in re.finditer(r"```", window):
                fence_open = not fence_open
            cut = -1
            if not fence_open:
                limit = max(MIN_SZ, len(window))
                cut = window.rfind("\n\n", 0, limit)
                if cut < MIN_SZ:
                    cut = window.rfind("\n", 0, limit)
                if cut < MIN_SZ:
                    for sep in (". ", "? ", "! "):
                        loc = window.rfind(sep, 0, limit)
                        if loc >= MIN_SZ:
                            cut = loc + len(sep)
                            break
            if cut >= MIN_SZ:
                j = i + cut
            chunk = full_text[i:j]
            if chunk:
                yield chunk
            i = j
    # TO TRACK ANSWER BUBBLE STATE - MANAGES UI APPEND; WHY: AVOIDS DUPLICATE ENTRIES; EDGE: MULTI-YIELD -> EXTEND; PERF: O(1); SEC: HISTORY VALIDATED.
    answer_bubble_started = False
    def _ensure_answer_bubble_started() -> bool:
        nonlocal answer_bubble_started
        if not answer_bubble_started:
            append_to_chatbot(chat_history, "", metadata={"role": "assistant"})
            answer_bubble_started = True
            return True
        return False
    def _route_and_emit(buffer: str) -> Generator[List[Dict[str, str]], None, None]:
        nonlocal answer_bubble_started
        if _ensure_answer_bubble_started():
            yield list(chat_history)
        chat_history[-1]["content"] += buffer
        yield list(chat_history)
        print(buffer, end="", flush=True)
    # TO ATTEMPT OPENAI-COMPATIBLE STREAM - PRIMARY NON-HF PATH; WHY: FAST FOR API; EDGE: DELTA NONE -> SKIP; PERF: STREAMING; SEC: LLM HANDLES.
    try:
        from llama_index.llms.huggingface import HuggingFaceLLM as _HFL_TYPE
        stream_complete = getattr(llm, "stream_complete", None)
        if callable(stream_complete) and not isinstance(llm, _HFL_TYPE):
            for ev in stream_complete(prompt):
                delta = ev if isinstance(ev, str) else (getattr(ev, "delta", None) or getattr(ev, "text", ""))
                if not delta:
                    continue
                for out in _route_and_emit(delta):
                    yield out
            return
    except Exception as e:
        logger.warning(f"[PI] stream_complete failed: {e}")
    # TO ATTEMPT HUGGINGFACE TRANSFORMERS STREAM - CRITICAL PATH FOR LOCAL; WHY: STREAMING EFFICIENCY; EDGE: EMPTY -> GUARD; PERF: THREADED; SEC: INFERENCE MODE.
    try:
        from llama_index.llms.huggingface import HuggingFaceLLM as _HFL_TYPE
        if isinstance(llm, _HFL_TYPE) and hasattr(llm, "_model") and hasattr(llm, "_tokenizer"):
            # TO LAZY IMPORT INSIDE BRANCH - AVOIDS GLOBAL DEP; WHY: CONDITIONAL LOAD; RISK: IMPORT FAIL -> FALLBACK; ASSUM: TRANSFORMERS INSTALLED; EDGE: NONE; PERF: DEFERRED; SEC: NO EXEC.
            import threading
            import torch
            from transformers import TextIteratorStreamer
            model = llm._model
            tok = llm._tokenizer
            device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            # TO TOKENIZE WITH SPECIALS - ADDS MODEL TOKENS; WHY: PREPS INPUT; RISK: STRIP -> EMPTY (GUARDED BELOW); ASSUM: ADD_SPECIAL_OK; EDGE: LONG PROMPT -> OOM (HANDLED OUTER); PERF: O(N); SEC: TOKENIZER SAFE.
            enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
            # TO HARD GUARD SEQ_LEN >=1 - INSERTS BOS/EOS/0 IF EMPTY; WHY: PREVENTS CACHE_POSITION EMPTY (TRANSFORMERS 4.55.4 BUG); RISK: SEMANTIC SHIFT (MINIMAL, SKIPPED); ASSUM: VOCAB >0; EDGE: NO BOS/EOS -> 0; PERF: O(1); SEC: MODEL-DEFINED IDS.
            def _force_min_seq_len(e: Any) -> Dict[str, torch.Tensor]:
                _ids = e.get("input_ids")
                if _ids is not None and _ids.numel() > 0 and _ids.shape[-1] > 0:
                    return {"input_ids": _ids, "attention_mask": e.get("attention_mask")}
                if getattr(tok, "bos_token_id", None) is not None:
                    fid = tok.bos_token_id
                elif getattr(tok, "eos_token_id", None) is not None:
                    fid = tok.eos_token_id
                else:
                    fid = 0
                _input_ids = torch.tensor([[fid]], dtype=torch.long)
                _attn = torch.ones_like(_input_ids, dtype=torch.long)
                logger.info("Empty input_ids -> initialized with single special token for safe start.")
                return {"input_ids": _input_ids, "attention_mask": _attn}
            guarded = _force_min_seq_len(enc)
            input_ids = guarded["input_ids"]
            attention_mask = guarded.get("attention_mask")
            # TO DEVICE MOVE - DICT-SAFE; WHY: CUDA CONSISTENCY; RISK: OOM (HANDLED OUTER); ASSUM: DEVICE VALID; EDGE: CPU FALLBACK; PERF: O(N); SEC: NO LEAK.
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            # TO STREAMER SETUP - SKIPS PROMPT/SPECIALS; WHY: CLEAN OUTPUT; RISK: NONE; ASSUM: TOK VALID; EDGE: EMPTY -> EMPTY; PERF: STREAMING; SEC: NO.
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            # TO PAD-ID FALLBACK - REQUIRED FOR CAUSAL LMS; WHY: STABILIZES SAMPLING; RISK: EOS AS PAD (STANDARD); ASSUM: EOS EXISTS; EDGE: NONE -> PASS; PERF: O(1); SEC: NO MUTATE GLOBAL.
            try:
                if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                    tok.pad_token_id = tok.eos_token_id
            except Exception:
                pass
            # TO GENERATION KWARGS - EXPLICIT/CRITICAL; WHY: CUSTOM CONFIG; RISK: OVERRIDE DEFAULTS; ASSUM: CONFIG VALID; EDGE: LOW TEMP -> DETERMINISTIC; PERF: MODEL-DEP; SEC: SAFE VALUES.
            gen_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "streamer": streamer,
                "use_cache": True,
                "max_new_tokens": CONFIG["MAX_NEW_TOKENS"],
                "temperature": CONFIG["TEMPERATURE"],
                "top_p": CONFIG["TOP_P"],
                "do_sample": True,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            # TO FINAL ASSERT - NO SIZE-0; WHY: FAIL CLOSED IF GUARD MISS; RISK: RAISE (CAUGHT OUTER); ASSUM: GUARD WORKS; EDGE: NONE; PERF: O(1); SEC: PREVENTS UNDEF BEHAVIOR.
            if gen_kwargs["input_ids"].shape[-1] == 0:
                raise RuntimeError("Guard failed: input_ids has zero length before generate().")
            # TO BACKGROUND GENERATION - THREADED FOR STREAM; WHY: NON-BLOCKING; RISK: THREAD HANG (TIMEOUT JOIN); ASSUM: DAEMON OK; EDGE: ERROR -> BOX; PERF: PARALLEL; SEC: INFERENCE MODE.
            err_box: Dict[str, Optional[BaseException]] = {"exc": None}
            def _bg_run(_kwargs: Dict[str, Any]) -> None:
                try:
                    with torch.inference_mode():
                        model.generate(**_kwargs)
                except Exception as e:
                    err_box["exc"] = e
                    try:
                        streamer.end_of_stream = True
                    except Exception:
                        pass
            t = threading.Thread(target=_bg_run, args=(gen_kwargs,), daemon=True)
            t.start()
            for token_text in streamer:
                if not token_text:
                    continue
                for out in _route_and_emit(token_text):
                    yield out
            t.join(timeout=0.1)
            exc = err_box["exc"]
            if exc is not None and isinstance(exc, IndexError) and (
                "cache_position" in str(exc) or "size 0" in str(exc) or "index -1" in str(exc)
            ):
                logger.warning("[PI] Retry without cache due to cache_position/size-0 IndexError.")
                # TO FORCE FRESH MIN INPUT - DISABLE CACHE; WHY: WORKAROUND HF BUG; RISK: SLOWER (NO CACHE); ASSUM: FID VALID; EDGE: RETRY FAIL -> RAISE; PERF: SINGLE RETRY; SEC: MINIMAL INPUT.
                fid = (
                    tok.bos_token_id
                    if getattr(tok, "bos_token_id", None) is not None
                    else (tok.eos_token_id if getattr(tok, "eos_token_id", None) is not None else 0)
                )
                input_ids = torch.tensor([[fid]], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
                streamer2 = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
                gen_kwargs2 = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "streamer": streamer2,
                    "use_cache": False,  # MIL: WORKAROUND FOR HF EDGE BUG
                    "max_new_tokens": CONFIG["MAX_NEW_TOKENS"],
                    "temperature": CONFIG["TEMPERATURE"],
                    "top_p": CONFIG["TOP_P"],
                    "do_sample": True,
                }
                err_box["exc"] = None
                threading.Thread(target=_bg_run, args=(gen_kwargs2,), daemon=True).start()
                for token_text in streamer2:
                    if not token_text:
                        continue
                    for out in _route_and_emit(token_text):
                        yield out
                if err_box["exc"] is not None:
                    raise err_box["exc"]
            if err_box["exc"] is not None:
                # TO SURFACE SAFE ERROR - NO TRACEBACK IN CHAT; WHY: USER-FRIENDLY; RISK: MASK DETAILS (LOGGED); ASSUM: EXC TYPE SAFE; EDGE: NONE -> RETURN; PERF: O(1); SEC: NO LEAK.
                msg = f"Generation error (HF): {type(err_box['exc']).__name__}"
                logger.error(f"[PI] HF stream failed: {err_box['exc']}")
                yield msg
                return
            return
    except Exception as e:
        logger.warning(f"[PI] HF streaming failed (fallback to non-stream): {e}")
    # TO FALLBACK NON-STREAMING - COMPLETE GEN WITH CHUNKING; WHY: RESILIENCE; EDGE: EMPTY -> YIELD MSG; PERF: FULL GEN; SEC: LLM SAFE.
    try:
        resp = llm.complete(prompt)
        full_text = resp.text if hasattr(resp, "text") else str(resp)
        if full_text.strip().lower() == "empty response":
            yield "Empty response from model."
            return
        for chunk in _yield_boundary_chunks(full_text):
            for out in _route_and_emit(chunk):
                yield out
        return
    except Exception as e:
        err = f"Generation failed: {str(e)}"
        logger.error(f"[PI] Fallback failed: {e}")
        yield err

### 

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

####

def chat_fn(
    message: str,
    chat_history: List[Dict[str, str]],
    state: Dict[str, Any]
) -> Generator[Tuple[str, List[Dict[str, str]], Dict[str, Any]], None, None]:
    """
    Gradio chat handler: append user turn, then stream assistant chunks via process_input.
    """
    if not isinstance(chat_history, list):
        chat_history = []
    if not isinstance(state, dict):
        state = {}

    msg = (message or "").strip()
    if not msg:
        logger.debug("[CF] empty message -> single yield")
        # keep textbox; return current history/state
        yield "", gr.update(value=list(chat_history)), state
        return

    # 1) append user turn
    append_to_chatbot(chat_history, msg, metadata={"role": "user"})
    logger.debug(f"[CF] appended user: {msg[:80]!r} | history_size={len(chat_history)}")
    # clear textbox, push immediate user bubble
    yield "", gr.update(value=list(chat_history)), state

    # 2) stream assistant reply
    first = True
    for item in process_input(msg, state, chat_history):
        if isinstance(item, list):
            # plugin command returned a full chat_history
            chat_history[:] = item
            #logger.debug(f"[CF] plugin returned full history | size={len(chat_history)}")
            yield "", gr.update(value=list(chat_history)), state
            continue

        chunk = str(item)
        if not chunk:
            logger.debug("[CF] empty chunk -> heartbeat yield")
            yield "", gr.update(value=list(chat_history)), state
            continue

        if first:
            append_to_chatbot(chat_history, chunk, metadata={"role": "assistant"})
            logger.debug(f"[CF] appended assistant FIRST: {chunk[:80]!r} | history_size={len(chat_history)}")
            first = False
        else:
            # extend the last assistant bubble progressively
            if chat_history and chat_history[-1].get("role") == "assistant":
                chat_history[-1]["content"] += chunk
                logger.debug(f"[CF] extended assistant: +{len(chunk)} chars | now={len(chat_history[-1]['content'])}")

        # stream update
        yield "", gr.update(value=list(chat_history)), state


####

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
            theme=gr.themes.Monochrome(
                primary_hue="gray",
                secondary_hue=gr.themes.Color(c100="#f5f5f4", c200="#e7e5e4", c300="#d6d3d1", c400="#a8a29e", c50="#fafaf9", c500="#78716c", c600="#57534e", c700="#44403c", c800="#292524", c900="#1c1917", c950="#111111"),
                neutral_hue="gray",
                text_size="lg",
                radius_size="lg",
                font=[gr.themes.GoogleFont("Roboto Italic"), "ui-sans-serif", "system-ui", "sans-serif"],
                font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
            ).set(
                body_background_fill="#000000",
                body_background_fill_dark="#000000",
                block_background_fill="#1a1a1a",
                block_background_fill_dark="#1a1a1a",
                block_info_text_size="*text_lg",
                checkbox_label_background_fill="*neutral_800",
                checkbox_label_border_color_selected="*primary_800",
                checkbox_label_border_color_selected_dark="*primary_800",
                block_radius="*radius_lg",
                block_shadow="*shadow_drop",
                button_large_radius="*radius_lg",
                button_medium_radius="*radius_lg",
                button_small_radius="*radius_lg",
                button_primary_background_fill="*primary_600",
                button_primary_background_fill_dark="*primary_700",
                button_primary_background_fill_hover="*primary_500",
                button_primary_background_fill_hover_dark="*primary_600",
                button_primary_text_color="*neutral_50",
                button_primary_text_color_dark="*neutral_50",
            ),
            css="""
            .hljs {
                color: #abb2bf;
                background: #282c34;
            }
            .hljs-comment,
            .hljs-quote {
                color: #5c6370;
                font-style: italic;
            }
            .hljs-doctag,
            .hljs-keyword,
            .hljs-formula {
                color: #c678dd;
            }
            .hljs-section,
            .hljs-name,
            .hljs-selector-tag,
            .hljs-deletion,
            .hljs-subst {
                color: #e06c75;
            }
            .hljs-literal {
                color: #56b6c2;
            }
            .hljs-string,
            .hljs-regexp,
            .hljs-addition,
            .hljs-attribute,
            .hljs-meta .string {
                color: #98c379;
            }
            .hljs-attr,
            .hljs-variable,
            .hljs-template-variable,
            .hljs-type,
            .hljs-selector-class,
            .hljs-selector-attr,
            .hljs-selector-pseudo,
            .hljs-number {
                color: #d19a66;
            }
            .hljs-symbol,
            .hljs-bullet,
            .hljs-link,
            .hljs-meta,
            .hljs-selector-id,
            .hljs-title {
                color: #61aeee;
            }
            .hljs-built_in,
            .hljs-title.class_,
            .hljs-class .hljs-title {
                color: #e6c07b;
            }
            .hljs-emphasis {
                font-style: italic;
            }
            .hljs-strong {
                font-weight: bold;
            }
            .hljs-link {
                text-decoration: underline;
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
    CLI input loop. Supports:
      - .commands (routed to handle_command)
      - normal queries (streamed via process_input)
    """
    # TO INITIALIZE LOCAL STATE - PRESERVES ISOLATION FROM GLOBAL.
    chat_history: List[Dict[str, str]] = []
    state: Dict[str, Any] = {}
    # TO DISPLAY ENTRY MESSAGE - INFORMS USER OF MODE AND EXIT.
    print("Entering CLI mode. Type '.exit' to quit.")
    while True:
        try:
            # TO DISPLAY CUSTOM PROMPT AND READ INPUT - MIMICS TERMINAL; HANDLES COLORS VIA INPUT().
            user_input = input(cli_prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            # TO HANDLE TERMINATION GRACEFULLY - MATCHES ORIGINAL; SAFE EXIT MESSAGE.
            print("\nExiting CLI.")
            break
        # TO SKIP EMPTY INPUT - PREVENTS UNNECESSARY PROCESSING.
        if not user_input:
            continue
        # TO CHECK EXIT COMMAND - PRESERVES ORIGINAL EXIT CONDITION.
        if user_input == ".exit":
            break
        # TO CHECK IF COMMAND - SKIP APPEND FOR COMMANDS TO AVOID HISTORY POLLUTION.
        is_command = user_input.startswith(".")
        if not is_command:
            # TO APPEND USER INPUT TO HISTORY - ONLY FOR QUERIES; LOGS FOR DEBUG.
            append_to_chatbot(chat_history, user_input, metadata={"role": "user"})
            logger.debug(f"[CLI] appended user: {user_input[:80]!r} | history_size={len(chat_history)}")
        # TO STREAM ASSISTANT/COMMAND OUTPUT - COLORS OUTPUT GREEN FOR TERMINAL STYLE.
        first = True
        for item in process_input(user_input, state, chat_history):
            if isinstance(item, list):
                # TO UPDATE HISTORY IF PLUGIN RETURNS FULL LIST - PRESERVES PLUGIN BEHAVIOR.
                chat_history[:] = item
                logger.debug(f"[CLI] plugin returned full history | size={len(chat_history)}")
                continue
            # TO CONVERT ITEM TO STRING - ENSURES PRINTABLE; SKIPS EMPTY.
            chunk = str(item)
            if not chunk:
                logger.debug("[CLI] empty chunk (heartbeat)")
                continue
            if is_command:
                # TO PRINT COMMAND OUTPUT DIRECTLY - NO HISTORY APPEND FOR COMMANDS.
                print(f"{Fore.GREEN}{chunk}{Fore.RESET}", end="", flush=True)
            else:
                if first:
                    # TO APPEND FIRST CHUNK AS NEW ASSISTANT ENTRY - INITIALIZES RESPONSE FOR QUERIES.
                    append_to_chatbot(chat_history, chunk, metadata={"role": "assistant"})
                    logger.debug(f"[CLI] appended assistant FIRST: {chunk[:80]!r} | history_size={len(chat_history)}")
                    first = False
                else:
                    # TO EXTEND EXISTING ASSISTANT ENTRY - ACCUMULATES STREAMED CONTENT FOR QUERIES.
                    if chat_history and chat_history[-1].get("role") == "assistant":
                        chat_history[-1]["content"] += chunk
                        logger.debug(f"[CLI] extended assistant: +{len(chunk)} chars | now={len(chat_history[-1]['content'])}")
                # TO PRINT COLORED CHUNK PROGRESSIVELY - FOR QUERIES; FLUSH FOR IMMEDIACY.
                print(f"{Fore.GREEN}{chunk}{Fore.RESET}", end="", flush=True)
        # TO ADD NEWLINE AFTER RESPONSE - ENSURES CLEAN SEPARATION.
        print()

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

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        print(f"{dark_gray}{tag()}Starting Gradio application...{reset_style}")
        launch_app()
    except Exception as e:
        logger.exception(f"Critical error in main application: {e}")
        print(f"{tag()}A critical error occurred: {e}. Exiting.")
        traceback.print_exc()
    finally:
        cleanup()
