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
import queue
import traceback
import argparse
import time
import sys
import asyncio
import shlex
import threading
from types import GeneratorType
from functools import partial
from datetime import datetime
from pathlib import Path
from colorama import Style, Fore
from logging.handlers import QueueHandler, QueueListener
from queue import Queue, Empty
from typing import Dict, Callable, List, Generator, Optional, Tuple, Any, Union
from gptqmodel import GPTQModel
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import Settings
import torch
import gradio as gr
from functools import partial
import requests

from gia.core.logger import logger
from gia.core.utils import append_to_chatbot
from gia.core.logger import logger, log_banner
from gia.core.state import state_manager
from gia.core.state import load_state
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
from gia.core.logger import logger, log_banner
from gia.config import CONFIG, PROJECT_ROOT, system_prefix

# CONFIG VALUES FOR DIRECT USE
RULES_PATH = CONFIG["RULES_PATH"]
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
# Global chat_history and message queue
chat_history = []

log_banner(__file__)

class PluginSandbox:
    """Isolated sandbox for plugin execution using threading with cooperative stop.

    TO INITIALIZE WITH QUEUES AND EVENT FOR STOP.
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
        # TO SET UP PLUGIN DETAILS AND QUEUES
        self.plugin_name = plugin_name
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.thread: Optional[threading.Thread] = None
        self.output_queue: Queue = Queue()  # FOR RESULTS/STREAMING
        self.input_queue: Queue = Queue()  # FOR COMMANDS FROM MAIN (E.G., STOP)
        self.logger_queue: Queue = Queue()  # FOR LOGS
        self.stop_event = threading.Event()  # FOR COOPERATIVE STOP
        self.listener: Optional[QueueListener] = None

    def start(self) -> None:
        """Start the sandbox thread and logger listener.

        TO CREATE THREAD WITH STOP EVENT AND START DAEMON.
        """
        # TO CREATE AND START THREAD
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
        )
        self.thread.daemon = True
        self.thread.start()

        # TO CONFIGURE logger HANDLERS - TERMINAL ONLY
        handlers = [logger.StreamHandler(sys.stdout)]
        self.listener = QueueListener(self.logger_queue, *handlers)
        self.listener.start()

    def stop(self) -> None:
        """Stop the sandbox thread cooperatively and clean up resources.

        TO SIGNAL STOP VIA EVENT AND QUEUE, JOIN WITH TIMEOUT, AND STOP LISTENER.
        """
        # TO SET STOP SIGNAL FOR COOPERATIVE EXIT
        self.stop_event.set()
        self.input_queue.put("stop")

        # TO WAIT FOR THREAD COMPLETION WITH TIMEOUT TO AVOID HANGS
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)  # 5-SECOND TIMEOUT FOR GRACEFUL SHUTDOWN
            if self.thread.is_alive():
                logger.warning(f"Timeout waiting for plugin '{self.plugin_name}' thread to stop.")

        # TO STOP logger LISTENER AND CLEAN UP
        if self.listener:
            self.listener.stop()
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

        Args:
            func: Plugin function.
            args: Positional args.
            kwargs: Keyword args.
            output_queue: Queue for results/streaming.
            input_queue: Queue for commands from main.
            logger_queue: Queue for logs.
            stop_event: Event to check for stop signal.
        """
        # TO SET UP logger IN CHILD THREAD
        root_logger = logger.getLogger()
        queue_handler = QueueHandler(logger_queue)
        root_logger.addHandler(queue_handler)

        try:
            # DEBUG PRINT TO CONFIRM THREAD START
            logger.debug("Plugin thread started; executing function.")
            # TO EXECUTE PLUGIN FUNCTION
            result = func(*args, **kwargs)
            if inspect.isgenerator(result):
                # TO HANDLE GENERATOR WITH BIDIRECTIONAL CHECKS AND STOP EVENT
                for chunk in result:
                    if stop_event.is_set():
                        logger.info("Plugin stopped by event")
                        break
                    try:
                        cmd = input_queue.get_nowait()
                        if cmd == "stop":
                            logger.info("Plugin stopped by command")
                            break
                    except Empty:
                        pass
                    output_queue.put(chunk)
            else:
                # TO HANDLE NON-GENERATOR RESULT
                if stop_event.is_set():
                    logger.info("Plugin stopped by event")
                else:
                    output_queue.put(result)
            output_queue.put(None)  # SENTINEL FOR END
        except BaseException as exc:
            # TO SEND EXCEPTION ACROSS THREAD
            output_queue.put(exc)
            output_queue.put(None)
        except Exception as e:
            logger.error(f"Error in plugin resumption: {e}")
            output_queue.put(e)
            output_queue.put(None)

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
METADATA_TITLE = f"🛠️ Generated by {PROJECT_ROOT.name.upper()}-{hash(PROJECT_ROOT) % 10000}"
dark_gray = Style.DIM + Fore.LIGHTBLACK_EX
reset_style = Style.RESET_ALL

##################################################################################

# LOGGER SETUP - FOR CONSISTENT logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.INFO)
    


####

# TIMESTAMP TAG - FOR logger
def tag() -> str:
    """Generate timestamp for logger."""
    # TO FORMAT TIMESTAMP
    current_time = datetime.now().strftime("[%H:%M:%S]")
    return f"{current_time} : $ "

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


from queue import Empty

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

    # -------- built-ins --------
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

    # -------- plugins --------
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
    * STREAMS VIA TextIteratorStreamer (HuggingFace Local/GPTQ).
    * FALLBACK: NON-STREAM + BOUNDARY-AWARE CHUNKING (SAFE MARKDOWN).
    * EVERY UI CHUNK MIRRORED TO TERMINAL WITH FLUSH=TRUE.
    * RAG + PLUGINS UNTOUCHED.
    """
    # INITIALIZE APPLICATION STATE
    app_state = load_state()
    # SANITIZE INPUT TEXT; PREVENT EMPTY PROCESSING
    text = (user_text or "").strip()
    if not text:
        # LOG EMPTY INPUT FOR AUDIT; NO-OP TO AVOID RESOURCE WASTE
        logger.debug("[PI] empty user_text -> no-op")
        return
    # LOG INPUT DETAILS FOR TRACEABILITY AND DEBUGGING
    logger.debug(f"[PI] input={text!r} | history_size={len(chat_history)} | state_is_dict={isinstance(state, dict)}")
    # HANDLE DOT-COMMANDS FOR LOCAL OPERATIONS
    if text.startswith("."):
        # EXECUTE COMMAND HANDLER; VALIDATE RESULT TYPE FOR SAFE YIELD
        result = handle_command(text, chat_history, state=state, is_gradio=bool(state is not None))
        if isinstance(result, str):
            if result:
                # MIRROR TO TERMINAL WITH FLUSH FOR REAL-TIME OUTPUT
                print(result, end="", flush=True)
            yield result
        else:
            for item in result:
                if isinstance(item, list):
                    yield item
                else:
                    s = str(item)
                    if s:
                        # MIRROR TO TERMINAL WITH FLUSH; ENSURE NO EMPTY YIELDS
                        print(s, end="", flush=True)
                        yield s
        return
    # ATTEMPT RAG QUERY IF ENABLED
    state_dict = state if isinstance(state, dict) else {}
    use_qe = bool(state_dict.get("use_query_engine_gr", True))
    used_qe = False
    if use_qe and app_state.QUERY_ENGINE:
        try:
            # QUERY RAG ENGINE; HANDLE STREAMING OR STATIC RESPONSE
            resp = app_state.QUERY_ENGINE.query(text)
            if hasattr(resp, "response_gen") and resp.response_gen is not None:
                # INIT ANSWER BUBBLE FOR STREAMING RAG
                append_to_chatbot(chat_history, "", metadata={"role": "assistant"})
                yield list(chat_history)
                for chunk in resp.response_gen:
                    s = str(chunk)
                    if not s or s.isspace() or s.strip().lower() == "empty response":
                        continue
                    # APPEND CHUNK TO CHAT HISTORY; YIELD FOR UI UPDATE
                    chat_history[-1]["content"] += s
                    yield list(chat_history)
                    # MIRROR TO TERMINAL WITH FLUSH
                    print(s, end="", flush=True)
                return
            else:
                s = resp.response if hasattr(resp, "response") else str(resp)
                if s and s.strip().lower() != "empty response":
                    # APPEND FULL RESPONSE TO CHAT HISTORY
                    append_to_chatbot(chat_history, s, metadata={"role": "assistant"})
                    yield list(chat_history)
                    # MIRROR TO TERMINAL WITH FLUSH
                    print(s, end="", flush=True)
                    return
                used_qe = False
        except Exception as e:
            # LOG RAG ERROR; FALLBACK TO LLM FOR RESILIENCE
            logger.warning(f"[PI] Query engine error, falling back to LLM: {e}")
            used_qe = False
    if used_qe:
        return
    # VALIDATE LLM AVAILABILITY
    llm = app_state.LLM
    if llm is None:
        err = "Error: No model loaded for query."
        # OUTPUT ERROR TO TERMINAL AND YIELD
        print(err, end="", flush=True)
        yield err
        return
    # CONSTRUCT PROMPT WITH SYSTEM PREFIX
    sys_prompt = system_prefix()
    prompt = f"{sys_prompt}\n{text}"
    # DEFINE BOUNDARY-AWARE CHUNKER FOR SAFE MARKDOWN STREAMING
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
    # TRACK ANSWER BUBBLE STATE
    answer_bubble_started = False
    # ENSURE ANSWER BUBBLE INITIALIZATION
    def _ensure_answer_bubble_started() -> bool:
        nonlocal answer_bubble_started
        if not answer_bubble_started:
            # APPEND NEW ASSISTANT ENTRY TO CHAT HISTORY
            append_to_chatbot(chat_history, "", metadata={"role": "assistant"})
            answer_bubble_started = True
            return True
        return False
    # DEFINE STREAM ROUTER FOR DIRECT APPEND TO ANSWER
    def _route_and_emit(buffer: str) -> Generator[List[Dict[str, str]], None, None]:
        nonlocal answer_bubble_started
        if _ensure_answer_bubble_started():
            # YIELD UPDATED HISTORY FOR UI REDRAW ON BUBBLE CREATION
            yield list(chat_history)
        # APPEND BUFFER TO LAST CONTENT; ASSUME VALID TEXT
        chat_history[-1]["content"] += buffer
        # YIELD UPDATED HISTORY FOR UI REDRAW
        yield list(chat_history)
        # MIRROR TO TERMINAL WITH FLUSH
        print(buffer, end="", flush=True)
    # ATTEMPT OPENAI-COMPATIBLE STREAMING
    try:
        from llama_index.llms.huggingface import HuggingFaceLLM as _HFL_TYPE
        stream_complete = getattr(llm, "stream_complete", None)
        if callable(stream_complete) and not isinstance(llm, _HFL_TYPE):
            # PROCESS STREAM EVENTS; ROUTE DELTAS
            for ev in stream_complete(prompt):
                delta = ev if isinstance(ev, str) else (getattr(ev, "delta", None) or getattr(ev, "text", ""))
                if not delta:
                    continue
                for out in _route_and_emit(delta):
                    yield out
            return
    except Exception as e:
        # LOG STREAMING FAILURE; FALLBACK FOR RESILIENCE
        logger.warning(f"[PI] stream_complete failed: {e}")
    # ATTEMPT HF TRANSFORMERS STREAMING
    try:
        from llama_index.llms.huggingface import HuggingFaceLLM as _HFL_TYPE
        if isinstance(llm, _HFL_TYPE) and hasattr(llm, "_model") and hasattr(llm, "_tokenizer"):
            from transformers import TextIteratorStreamer
            import threading
            model = llm._model
            tok = llm._tokenizer
            # TOKENIZE PROMPT; VALIDATE NON-EMPTY
            enc = tok(prompt, return_tensors="pt")
            if "input_ids" not in enc or enc["input_ids"] is None or enc["input_ids"].numel() == 0:
                raise RuntimeError("Empty tokenized prompt (HF).")
            dev = getattr(model, "device", None)
            if dev is not None:
                # MOVE TENSORS TO DEVICE; PREVENT DEVICE MISMATCH
                enc = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in enc.items()}
            # INITIALIZE STREAMER; SKIP PROMPT AND SPECIALS
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            # PREPARE GENERATION KWARGS; MERGE USER SETTINGS SAFELY
            gen_kwargs = {
                "input_ids": enc["input_ids"],
                "attention_mask": enc.get("attention_mask", None),
                "max_new_tokens": MAX_NEW_TOKENS,
                "streamer": streamer,
            }
            if gen_kwargs["attention_mask"] is None:
                del gen_kwargs["attention_mask"]
            try:
                user_gk = getattr(llm, "generate_kwargs", None)
                if isinstance(user_gk, dict):
                    for k, v in user_gk.items():
                        if k not in gen_kwargs and v is not None:
                            gen_kwargs[k] = v
            except Exception:
                pass
            # CAPTURE EXCEPTIONS IN BACKGROUND THREAD
            err_box = {"exc": None}
            def _bg():
                try:
                    # EXECUTE MODEL GENERATION
                    model.generate(**gen_kwargs)
                except Exception as e:
                    err_box["exc"] = e
                    try:
                        streamer.end_of_stream = True
                    except Exception:
                        pass
            # START DAEMON THREAD FOR GENERATION
            threading.Thread(target=_bg, daemon=True).start()
            # PROCESS STREAMED TOKENS; ROUTE TO EMITTER
            for token_text in streamer:
                if not token_text:
                    continue
                for out in _route_and_emit(token_text):
                    yield out
            if err_box["exc"] is not None:
                raise err_box["exc"]
            return
    except Exception as e:
        # LOG HF STREAMING FAILURE; FALLBACK TO NON-STREAM
        logger.warning(f"[PI] HF streaming failed (fallback to non-stream): {e}")
    # NON-STREAMING FALLBACK GENERATION
    try:
        # GENERATE FULL RESPONSE; LIMIT TOKENS FOR SAFETY
        full = generate(text, sys_prompt, llm, max_new_tokens=MAX_NEW_TOKENS) or ""
        # LOG GENERATION LENGTH FOR MONITORING
        logger.debug(f"[PI] generate returned len={len(full)}")
    except Exception as e:
        msg = f"Error generating response: {e}"
        # OUTPUT ERROR TO TERMINAL AND YIELD; NO LEAK SENSITIVE INFO
        print(msg, end="", flush=True)
        yield msg
        return
    if not full:
        msg = "Empty response."
        # OUTPUT EMPTY RESPONSE NOTICE
        print(msg, end="", flush=True)
        yield msg
        return
    # INIT ANSWER BUBBLE FOR FULL RESPONSE
    if _ensure_answer_bubble_started():
        yield list(chat_history)
    # CHUNK AND YIELD FULL TEXT SAFELY
    for chunk in _yield_boundary_chunks(full):
        chat_history[-1]["content"] += chunk
        yield list(chat_history)
        print(chunk, end="", flush=True)

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
            title="SCRPS AI",
            delete_cache=(60, 60),
            theme="d8ahazard/rd_blue",
            css="#model-name-display{text-align:left;width:100%;font-size:1.1em;margin-top:0.5em;}",
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
                height=555,
                allow_tags=["think", "answer"],
                render_markdown=True,  
            )
            msg = gr.Textbox(placeholder="Enter your message or command (e.g., .gen)")
            submit_btn = gr.Button("Submit")

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
            submit_btn.click(
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


def cli_loop() -> None:
    """
    CLI input loop. Supports:
      - .commands (routed to handle_command)
      - normal queries (streamed via process_input)
    """
    chat_history: List[Dict[str, str]] = []
    state: Dict[str, Any] = {}

    print("Entering CLI mode. Type '.exit' to quit.")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting CLI.")
            break

        if not user_input:
            continue
        if user_input == ".exit":
            break

        # 1) append user turn
        append_to_chatbot(chat_history, user_input, metadata={"role": "user"})
        logger.debug(f"[CLI] appended user: {user_input[:80]!r} | history_size={len(chat_history)}")

        # 2) stream assistant/command output
        first = True
        for item in process_input(user_input, state, chat_history):
            if isinstance(item, list):
                chat_history[:] = item
                logger.debug(f"[CLI] plugin returned full history | size={len(chat_history)}")
                continue

            chunk = str(item)
            if not chunk:
                logger.debug("[CLI] empty chunk (heartbeat)")
                continue

            if first:
                append_to_chatbot(chat_history, chunk, metadata={"role": "assistant"})
                logger.debug(f"[CLI] appended assistant FIRST: {chunk[:80]!r} | history_size={len(chat_history)}")
                first = False
            else:
                if chat_history and chat_history[-1].get("role") == "assistant":
                    chat_history[-1]["content"] += chunk
                    logger.debug(f"[CLI] extended assistant: +{len(chunk)} chars | now={len(chat_history[-1]['content'])}")

            print(chunk, end="", flush=True)  # progressive terminal
        print()  # newline after full response


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
