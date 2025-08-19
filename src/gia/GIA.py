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

# TODO auto generate_pyproject_toml
# import atexit
# from utils.generate_pyproject import should_generate, generate_pyproject_toml

# if should_generate():
#    atexit.register(generate_pyproject_toml)
#    print("(G.I.A) Dependency capture registered. Run app for ~20 min to load all modules, then stop to generate pyproject.toml.")

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
from gia.config import CONFIG, PROJECT_ROOT

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
DEVICE_MAP = "cuda" if torch.cuda.is_available() else "cpu"
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


def system_prefix() -> str:
    """Load system prompt from config or rules file, handling all edge cases."""
    # TO DEFINE DEFAULT PROMPT AS FALLBACK
    default_prompt = "You are an expert. Provide accurate, reliable, and evidence-based assistance to solve user queries effectively."
    try:
        # TO RETRIEVE SYSTEM_PROMPT FROM CONFIG IF DEFINED
        system_prompt = CONFIG.get("SYSTEM_PROMPT", None)
        # TO EARLY RETURN DEFAULT IF NEITHER SYSTEM_PROMPT NOR RULES_PATH IS SET
        if not system_prompt and not RULES_PATH:
            return default_prompt
        # TO CHECK AND LOAD FROM RULES_PATH IF IT EXISTS
        rules_path_obj = Path(RULES_PATH) if RULES_PATH else None
        if rules_path_obj and rules_path_obj.exists():
            try:
                # TO LOAD JSON FROM FILE
                with rules_path_obj.open("r", encoding="utf-8") as f:
                    rules = json.load(f)
                # TO RETRIEVE SYSTEM_PROMPT WITH DEFAULT FALLBACK WITHOUT MODIFICATION
                prompt_data = rules.get("system_prompt", [default_prompt])
                # TO HANDLE IF PROMPT_DATA IS STRING INSTEAD OF LIST
                if isinstance(prompt_data, str):
                    return prompt_data
                # TO JOIN LIST INTO SINGLE STRING
                return "\n".join(prompt_data)
            except (json.JSONDecodeError, IOError) as e:
                # TO LOG SPECIFIC ERROR AND FALL BACK TO DEFAULT
                logger.error(f"Error reading or parsing RULES_PATH '{RULES_PATH}': {e}")
                return default_prompt
        # TO USE CONFIG SYSTEM_PROMPT IF RULES_PATH NOT USED
        if system_prompt:
            return system_prompt
    except Exception as e:
        # TO LOG UNEXPECTED ERROR AND FALL BACK
        logger.error(f"Unexpected error in system_prefix: {e}")
    # TO RETURN DEFAULT AS ULTIMATE FALLBACK
    return default_prompt


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

# GLOBAL QUEUES FOR BIDIRECTIONAL INTERACTION
input_queue: queue.Queue = queue.Queue()  # COMMANDS FROM CLI/GRADIO TO PLUGINS/HANDLERS
output_queue: queue.Queue = queue.Queue()  # STREAMING OUTPUTS/RESPONSES TO BOTH INTERFACES
logger_queue: queue.Queue = queue.Queue()  # LOGS TO TERMINAL ONLY
from queue import Empty

# Excerpt from src/gia/GIA.py (complete handle_command function)
def handle_command(
    cmd: str,
    chat_history: List[Dict[str, str]],
    is_gradio: bool = False,
    is_chat_fn: bool = False,
    state: Optional[Union[gr.State, Dict[str, Any]]] = None,
) -> Union[str, Generator[Union[List[Dict[str, str]], Tuple[str, List[Dict[str, str]], gr.State]], None, None]]:
    """Unified handler for all commands: built-in, plugins, and queries.

    Handles CLI and Gradio differences, reducing duplication.

    Args:
        cmd: Command or query string.
        chat_history: Shared chat history.
        is_gradio: If True, yield Gradio-compatible updates.
        is_chat_fn: If True and is_gradio, include empty string for textbox.
        state: Application state (gr.State or dict).

    Returns:
        String for CLI or generator for Gradio.
    """
    # TO LOAD STATE AT START FOR DOT ACCESS
    app_state = load_state()
    if not isinstance(chat_history, list):
        raise ValueError("chat_history must be a list of dicts.")
    cmd_lower = cmd.strip().lower()
    result = ""

    # TO SAFELY ACCESS STATE.VALUE IF GR.STATE
    state_dict: Dict[str, Any] = {}
    if state is not None:
        try:
            if isinstance(state, gr.State):
                state_value = state.value
                logger.debug(f"Accessed state.value of type: {type(state_value).__name__}")
                if not isinstance(state_value, dict):
                    raise ValueError("State.value must be a dictionary.")
                state_dict = state_value
            elif isinstance(state, dict):
                state_dict = state
            else:
                raise TypeError("State must be gr.State or dict.")
        except (AttributeError, ValueError, TypeError) as e:
            logger.error(f"Invalid state in handle_command: {e}")
            state_dict = {}  # FALLBACK TO EMPTY DICT

    # TO CHECK LLM INITIALIZATION FOR NON-LOAD COMMANDS
    if app_state.LLM is None and cmd_lower not in [".load", ".create", ".info", ".delete", ".unload"]:
        error_msg = "Error: No model loaded. Please confirm a model first.\n"
        result = error_msg

    use_query_engine = state_dict.get("use_query_engine_gr", False)
    try:
        # TO HANDLE BUILT-IN COMMANDS
        if cmd_lower in [".load", ".create", ".info", ".delete", ".unload"]:
            if cmd_lower == ".create":
                save_database()
                use_query_engine = True
                result = "Database created. Query engine re-enabled.\n"
                state_dict["status_gr"] = "Database created"
            elif cmd_lower == ".load":
                result = load_database_wrapper(state_dict)
                use_query_engine = True
                state_dict["status_gr"] = "Database loaded"
            elif cmd_lower == ".info":
                cpu_usage, memory_usage, gpu_usage = get_system_info()
                result = (
                    f"System Info:\n"
                    f"CPU Usage: {cpu_usage}%\n"
                    f"Memory Usage: {memory_usage}%\n"
                    f"GPU Usage: {gpu_usage}%\n"
                    f"Model: {app_state.MODEL_NAME or 'None'}\n"
                    f"Database Loaded: {app_state.DATABASE_LOADED}\n"
                )
                state_dict["status_gr"] = "System info retrieved"
            elif cmd_lower == ".delete":
                db_path = Path(DB_PATH)
                if db_path.exists():
                    if is_gradio:
                        logger.info("Deleting database...")
                        try:
                            import shutil
                            shutil.rmtree(db_path)
                            result = "Database deleted.\n"
                        except Exception as e:
                            result = f"Error deleting database: {str(e)}\n"
                    else:
                        confirm = input("Confirm delete database? (y/n): ").lower().strip()
                        if confirm == 'y':
                            try:
                                import shutil
                                shutil.rmtree(db_path)
                                result = "Database deleted.\n"
                            except Exception as e:
                                result = f"Error deleting database: {str(e)}\n"
                        else:
                            result = "Delete canceled.\n"
                    state_manager.set_state("DATABASE_LOADED", False)
                    state_manager.set_state("INDEX", None)
                    state_manager.set_state("QUERY_ENGINE", None)
                else:
                    result = "No database to delete.\n"
                state_dict["status_gr"] = "Database deleted"
            elif cmd_lower == ".unload":
                if app_state.LLM is None:
                    result = "No model loaded to unload.\n"
                else:
                    try:
                        if isinstance(app_state.LLM, HuggingFaceLLM):
                            del app_state.LLM.model
                            del app_state.LLM
                            torch.cuda.empty_cache()
                            gc.collect()
                            result = "Local model unloaded and VRAM cleared.\n"
                        else:  # LlamaOpenAI or similar
                            del app_state.LLM
                            gc.collect()
                            result = "Online model unloaded.\n"
                        state_manager.set_state("LLM", None)
                        state_manager.set_state("MODEL_NAME", None)
                        state_manager.set_state("MODEL_PATH", None)
                        state_dict["status_gr"] = "Model unloaded"
                    except Exception as e:
                        result = f"Unload failed: {str(e)}\n"
        # TO HANDLE PLUGIN COMMANDS (STARTS WITH .)
        elif cmd_lower.startswith("."):
            # PARSE COMMAND SAFELY WITH SHLEX FOR QUOTED SINGLE OPTIONAL ARG
            parts = shlex.split(cmd)
            if len(parts) == 0 or not parts[0].startswith('.') or len(parts[0]) <= 1:
                raise ValueError("Invalid plugin command format. Must start with '.' followed by plugin name.")
            if len(parts) > 2:
                raise ValueError("Command exceeds maximum 2 positions (plugin name and optional quoted arg).")
            plugin_name = parts[0][1:].lower()
            optional_arg = parts[1].strip() if len(parts) > 1 else None  # SANITIZE WITH STRIP
            if plugin_name not in plugins:
                raise KeyError(f"Plugin '{plugin_name}' not found in loaded plugins.")
            plugin = plugins[plugin_name]
            # PREPARE WHITELISTED DEFAULT KWARGS FOR INJECTION IF IN SIG
            available_kwargs = {
                'llm': app_state.LLM,
                'query_engine': app_state.QUERY_ENGINE,
                'embed_model': app_state.EMBED_MODEL,
                'chat_history': chat_history,
            }
            sig = inspect.signature(plugin)
            kwargs_to_pass = {k: v for k, v in available_kwargs.items() if k in sig.parameters}
            if optional_arg is not None:
                # CONDITIONAL ADD: 'ARG' IF IN SIG, ELSE 'QUERY' IF IN SIG, ELSE IGNORE WITH WARNING
                if 'arg' in sig.parameters:
                    kwargs_to_pass['arg'] = optional_arg
                elif 'query' in sig.parameters:
                    kwargs_to_pass['query'] = optional_arg
                else:
                    logger.warning(f"Ignoring optional_arg '{optional_arg}' for plugin '{plugin_name}' as neither 'arg' nor 'query' in signature.")
            try:
                # VALIDATE BINDING: PARTIAL THEN FULL TO CATCH MISSING REQUIRED
                bound = sig.bind_partial(**kwargs_to_pass)
                sig.bind(**kwargs_to_pass)  # RAISE IF MISSING REQUIRED
            except TypeError as e:
                missing = [p for p in sig.parameters if p not in kwargs_to_pass and sig.parameters[p].default is inspect.Parameter.empty]
                error_msg = f"Plugin '{plugin_name}' missing required args: {', '.join(missing)}. Load model/database first."
                raise ValueError(error_msg) from e
            # START SANDBOX WITH BOUND ARGS/KWARGS
            sandbox = PluginSandbox(plugin_name, plugin, bound.args, bound.kwargs)
            sandbox.start()
            with sandboxes_lock:
                active_sandboxes[plugin_name] = sandbox
            try:
                if is_gradio:
                    def generator():
                        history = chat_history  # SHARED REF FOR IN-PLACE APPENDS
                        start_time = time.time()
                        max_poll_time = 3600  # TIMEOUT FOR LONG-RUNNING PLUGINS
                        while True:
                            try:
                                item = sandbox.output_queue.get(timeout=0.05)  # OPTIMIZED LOW TIMEOUT
                                if item is None:
                                    break
                                if isinstance(item, BaseException):
                                    raise item
                                if isinstance(item, list) and all(isinstance(d, dict) for d in item):
                                    # HISTORY YIELDED: UPDATE AND YIELD FOR UI
                                    logger.debug(f"Plugin '{plugin_name}' yielded history; yielding for UI refresh.")
                                    history = item
                                    if is_chat_fn:
                                        yield "", history, gr.State(state_dict)
                                    else:
                                        yield history, gr.State(state_dict)
                                    continue
                                # SAFE CONVERSION: JSON IF DICT/LIST, ELSE STR
                                try:
                                    msg = json.dumps(item) if isinstance(item, (dict, list)) else str(item)
                                except Exception:
                                    msg = str(item)
                                if msg.strip():
                                    history = append_to_chatbot(history, msg, metadata={"role": "assistant"})
                                    print(msg, flush=True)  # SYNC TO TERMINAL
                                if is_chat_fn:
                                    yield "", history, gr.State(state_dict)
                                else:
                                    yield history, gr.State(state_dict)
                            except Empty:
                                # YIELD ON EMPTY FOR RESPONSIVE UI, NO CPU SPIN
                                if is_chat_fn:
                                    yield "", history, gr.State(state_dict)
                                else:
                                    yield history, gr.State(state_dict)
                                time.sleep(0.05)
                                if time.time() - start_time > max_poll_time:
                                    raise TimeoutError(f"Plugin {plugin_name} timed out after {max_poll_time}s")
                    return generator()
                else:
                    collected = ""
                    history = chat_history  # SHARED REF FOR IN-PLACE APPENDS
                    start_time = time.time()
                    max_poll_time = 3600  # TIMEOUT FOR LONG-RUNNING PLUGINS
                    while True:
                        try:
                            item = sandbox.output_queue.get(timeout=0.05)  # OPTIMIZED LOW TIMEOUT
                            if item is None:
                                break
                            if isinstance(item, BaseException):
                                raise item
                            if isinstance(item, list) and all(isinstance(d, dict) for d in item):
                                # HISTORY YIELDED: UPDATE HISTORY FOR SHARED SYNC
                                logger.debug(f"Plugin '{plugin_name}' yielded history; updating shared history.")
                                history = item
                                continue
                            # SAFE CONVERSION: JSON IF DICT/LIST, ELSE STR
                            try:
                                msg = json.dumps(item) if isinstance(item, (dict, list)) else str(item)
                            except Exception:
                                msg = str(item)
                            if msg.strip():
                                history = append_to_chatbot(history, msg, metadata={"role": "assistant"})
                            print(msg, flush=True)
                            collected += msg + "\n"
                        except Empty:
                            time.sleep(0.05)
                            if time.time() - start_time > max_poll_time:
                                raise TimeoutError(f"Plugin {plugin_name} timed out after {max_poll_time}s")
                    return collected.strip() or "Plugin completed with no output."
            finally:
                sandbox.stop()
                with sandboxes_lock:
                    active_sandboxes.pop(plugin_name, None)
        # TO HANDLE NORMAL QUERIES (NON-DOT)
        else:
            if not cmd.strip():
                result = "Empty query.\n"
            elif use_query_engine and app_state.QUERY_ENGINE:
                response = app_state.QUERY_ENGINE.query(cmd)
                streamed_text = ""
                if hasattr(response, "response_gen"):
                    for chunk in response.response_gen:
                        streamed_text += chunk
                else:
                    streamed_text = response.response if hasattr(response, 'response') else str(response)
                result = streamed_text + "\n" if streamed_text.strip() not in ["", "Empty Response"] else generate(cmd, system_prefix(), app_state.LLM, max_new_tokens=1024) + "\n"
            elif app_state.LLM is not None:
                result = generate(cmd, system_prefix(), app_state.LLM, max_new_tokens=1024) + "\n"
            else:
                result = "Error: No model loaded for query.\n"
            state_dict["status_gr"] = "Response generated"
            state_dict["use_query_engine_gr"] = use_query_engine
        # Uniform output handling
        if result.strip() and not any(h.get("content", "") == result.strip() for h in chat_history[-3:]):
            chat_history = append_to_chatbot(chat_history, result, metadata={"role": "assistant"})
        if is_gradio:
            print(result, flush=True)  # SYNC TO TERMINAL
            def generator():
                if is_chat_fn:
                    yield "", chat_history, gr.State(state_dict)
                else:
                    yield chat_history, gr.State(state_dict)
            return generator()
        else:
            return result

    except Exception as e:
        error_msg = f"Command failed: {str(e)}\n"
        logger.error(error_msg)
        if error_msg.strip() and not any(h.get("content", "") == error_msg.strip() for h in chat_history[-3:]):
            chat_history = append_to_chatbot(chat_history, error_msg, metadata={"role": "assistant"})
        if is_gradio:
            print(error_msg, flush=True)  # SYNC TO TERMINAL
            def generator():
                if is_chat_fn:
                    yield "", chat_history, gr.State(state_dict)
                else:
                    yield chat_history, gr.State(state_dict)
            return generator()
        else:
            return error_msg

####

# LIST FOLDERS FOR DIRECTORY SCAN
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


# UPDATE MODEL DROPDOWN - FOR UI UPDATE
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

def process_input(user_text: str, state: gr.State, chat_history: List[Dict[str, str]]) -> Tuple[str, bool, gr.State, List[Dict[str, str]]]:
    """Process user input with state management.

    Args:
        user_text: User input text.
        state: Gradio state.
        chat_history: Shared chat history reference.

    Returns:
        Response, query engine flag, updated state, updated history.
    """
    # TO LOAD STATE
    app_state = load_state()
    if not isinstance(state, gr.State):
        raise ValueError("Invalid state type in process_input.")
    state_dict = state.value if isinstance(state.value, dict) else {"use_query_engine_gr": True}

    # TO PROCESS COMMAND WITH SHARED HISTORY
    result = handle_command(
        user_text, chat_history=chat_history, is_gradio=True, is_chat_fn=True, state=state
    )
    # HANDLE_COMMAND NOW YIELDS, SO CONSUME GENERATOR FOR RESULT
    response = ""
    for item in result:
        if isinstance(item, tuple) and len(item) == 3:  # (textbox, history, state)
            _, updated_history, updated_state = item
            state = updated_state
        response += item[0] if isinstance(item, tuple) else str(item)

    use_query_engine_gr = state_dict.get("use_query_engine_gr", True)
    return response, use_query_engine_gr, state, chat_history  # RETURN UPDATED SHARED HISTORY

####

def handle_load_button(state: gr.State) -> Tuple[List[Dict[str, str]], gr.State]:
    """Handle load button action with early initialization and deduplication.

    Args:
        state (gr.State): Gradio state.

    Returns:
        Tuple[List[Dict[str, str]], gr.State]: Updated chat history, updated state.
    """
    # TO INITIALIZE CHAT HISTORY EARLY FOR ALL RETURNS
    chat_history: List[Dict[str, str]] = []

    # TO LOAD STATE
    app_state = load_state()
    state_dict = state.value if isinstance(state, gr.State) else {}
    if not isinstance(state_dict, dict):
        logger.warning("State_dict is not a dictionary; resetting to default.")
        state_dict = {
            "use_query_engine_gr": True,
            "status_gr": "",
            "model_path_gr": None,
            "database_loaded_gr": False,
        }
    if state_dict.get("database_loaded_gr", False):
        message = "Database already loaded."
        if not chat_history or chat_history[-1].get("content") != message:
            append_to_chatbot(chat_history, message, metadata={"role": "assistant"})
        return chat_history, gr.State(state_dict)
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
                return chat_history, gr.State(state_dict)
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

    return chat_history, gr.State(state_dict)


####

def chat_fn(
    message: str, 
    chat_history: List[Dict[str, str]], 
    state: Union[gr.State, Dict[str, Any]]
) -> Generator[Tuple[str, List[Dict[str, str]], Union[gr.State, Dict[str, Any]]], None, None]:
    """Handle chat input.
    Args:
        message: User input message.
        chat_history: Chat history.
        state: Gradio state or dict.
    Yields:
        Textbox value, updated chat history, updated state.
    """
    # TO LOAD STATE FOR DOT ACCESS
    app_state = load_state()
    # TO EXTRACT STATE DICT SAFELY
    state_dict = state.value if isinstance(state, gr.State) else state
    # TO HANDLE EMPTY MESSAGE
    if not message.strip():
        yield "", chat_history, state if isinstance(state, gr.State) else state_dict
        return
    # TO APPEND USER MESSAGE TO HISTORY
    append_to_chatbot(chat_history, message, metadata={"role": "user"})
    # TO YIELD UPDATED HISTORY
    yield "", chat_history, state if isinstance(state, gr.State) else state_dict
    # TO PROCESS WITH SHARED HISTORY AND TRY/EXCEPT FOR ROBUSTNESS
    try:
        response, use_query_engine_gr, state, chat_history = process_input(message, state, chat_history)
        # TO UPDATE STATE DICT
        state_dict = state.value if isinstance(state, gr.State) else state
        state_dict["use_query_engine_gr"] = use_query_engine_gr
        state_dict["status_gr"] = "Response generated"
        # TO YIELD FINAL UPDATED
        yield "", chat_history, gr.State(state_dict) if isinstance(state, gr.State) else state_dict
    except Exception as e:
        # TO LOG ERROR AND APPEND TO HISTORY
        logger.error(f"Chat processing error: {e}")
        append_to_chatbot(chat_history, f"Error: {str(e)}", metadata={"role": "assistant"})
        # TO YIELD ERROR UPDATED
        yield "", chat_history, state if isinstance(state, gr.State) else state_dict


# BACKGROUND THREAD FOR GRADIO OUTPUT POLLING
def gradio_output_poller(chatbot_component, state):
    """Poll output_queue and append to Gradio chatbot."""
    while True:
        try:
            output = output_queue.get(timeout=1)  # BLOCKING WITH TIMEOUT TO AVOID CPU SPIN
            append_to_chatbot(chatbot_component.value, output)  # UPDATE CHATBOT
        except queue.Empty:
            pass

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
            model_name_display = gr.Markdown(
                "Loading model...", elem_id="model-name-display"
            )
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
            chatbot = gr.Chatbot(type="messages", value=[], height=777)
            msg = gr.Textbox(placeholder="Enter your message or command (e.g., .gen)")
            submit_btn = gr.Button("Submit")
            # ADD MODE SELECTION RADIO
            mode_radio = gr.Radio(
                choices=["Local", "HuggingFace", "OpenAI", "OpenRouter"],
                value="Local",
                label="Model Mode"
            )
            # WARNING MARKDOWN FOR API KEYS
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
            # OPENAI UI
            with gr.Row(visible=False) as openai_row:
                openai_model_dropdown = gr.Dropdown(
                    choices=fetch_openai_models(),
                    label="OpenAI Model",
                    value="gpt-4o",
                )
            # OPENROUTER UI
            with gr.Row(visible=False) as openrouter_row:
                openrouter_model_text = gr.Textbox(
                    label="OpenRouter Model",
                    value="x-ai/grok-3-mini",
                    placeholder="Enter OpenRouter model name (e.g., x-ai/grok-3-mini)"
                )
            confirm_model_button = gr.Button("Confirm Model")
            with gr.Row():
                load_button = gr.Button("Load Database")
            # PLUGIN GROUPING IN ACCORDION
            with gr.Accordion("Plugins", open=True):
                with gr.Row():
                    if not plugins:
                        gr.Markdown("No plugins loaded.")
                    else:
                        for plugin_name in plugins.keys():
                            btn = gr.Button(value=plugin_name.capitalize())
                            def plugin_handler_simple(
                                name: str,
                                chat_history: List[Dict[str, str]],
                                state: gr.State
                            ) -> Generator[Tuple[List[Dict[str, str]], gr.State], None, None]:
                                logger.debug(f"Clicked plugin: {name}")
                                try:
                                    handler_gen = handle_command(
                                        f".{name}",
                                        chat_history=chat_history,
                                        is_gradio=True,
                                        is_chat_fn=False,
                                        state=state
                                    )
                                    for updated in handler_gen:
                                        if isinstance(updated, tuple) and len(updated) == 2:
                                            yield updated[0], updated[1]
                                        else:
                                            logger.warning("Invalid handler yield; skipping.")
                                    yield chat_history, state
                                except Exception as e:
                                    logger.error(f"Plugin {name} error: {e}")
                                    yield chat_history, state
                            btn.click(
                                fn=partial(plugin_handler_simple, plugin_name),
                                inputs=[chatbot, session_state],
                                outputs=[chatbot, session_state],
                                queue=True,
                            )
            # UI LOGIC FOR MODE SWITCHING
            def update_mode_visibility(selected_mode: str, state):
                if hasattr(state, 'value'):
                    state_dict = state.value
                else:
                    state_dict = state if isinstance(state, dict) else {"mode": selected_mode}
                state_dict["mode"] = selected_mode
                warning_value = ""
                warning_visible = False
                hf_update = gr.update()
                if selected_mode == "HuggingFace":
                    if not os.getenv("HF_TOKEN"):
                        warning_value = "HF_TOKEN not detected. Please set it in your environment: `export HF_TOKEN=your_token_here` in your terminal and restart the app."
                        warning_visible = True
                    try:
                        url = "https://huggingface.co/api/models?inference_provider=fireworks-ai"
                        response = requests.get(url, timeout=10)
                        response.raise_for_status()
                        models = [m['id'] for m in response.json()]
                        hf_update = gr.update(choices=models, value=models[0] if models else "deepseek-ai/DeepSeek-R1")
                        logger.debug(f"Fetched {len(models)} models from fireworks-ai")
                    except Exception as e:
                        logger.warning(f"Failed to fetch HF models from fireworks-ai: {e}. Using fallback list.")
                elif selected_mode == "OpenAI":
                    if not os.getenv("OPENAI_API_KEY"):
                        warning_value = "OPENAI_API_KEY not detected. Please set it in your environment: `export OPENAI_API_KEY=sk-your_key_here` in your terminal and restart the app."
                        warning_visible = True
                elif selected_mode == "OpenRouter":
                    if not os.getenv("OPENROUTER_API_KEY"):
                        warning_value = "OPENROUTER_API_KEY not detected. Please set it in your environment: `export OPENROUTER_API_KEY=your_key_here` in your terminal and restart the app."
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
                lambda model, state: (state.value.update({"hf_model_name": model}) or state) if hasattr(state, 'value') else (state.update({"hf_model_name": model}) or state),
                inputs=[hf_model_dropdown, session_state],
                outputs=[session_state],
            )
            openai_model_dropdown.change(
                lambda model, state: (state.value.update({"openai_model_name": model}) or state) if hasattr(state, 'value') else (state.update({"openai_model_name": model}) or state),
                inputs=[openai_model_dropdown, session_state],
                outputs=[session_state],
            )
            openrouter_model_text.change(
                lambda model, state: (state.value.update({"openrouter_model_name": model}) or state) if hasattr(state, 'value') else (state.update({"openrouter_model_name": model}) or state),
                inputs=[openrouter_model_text, session_state],
                outputs=[session_state],
            )
            scan_dir_button.click(
                fn=update_model_dropdown,
                inputs=[models_dir_input],
                outputs=[folder_dropdown, chatbot],
            )
            # CONFIRM MODEL HANDLER
            def confirm_model_handler(selected_folder: str, state: gr.State, mode: str, hf_model: str, openai_model: str, openrouter_model: str, chat_history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict, str]:
                """Handle model confirmation button in Gradio, updating state and loading model."""
                if hasattr(state, 'value'):
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
        # INIT GRADIO SERVER LAUNCH
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
    """Main CLI input loop with queue integration."""
    # TO HANDLE COMMANDS IN MAIN THREAD
    while True:
        print("~GIA$ > ", end='', flush=True)
        cmd = sys.stdin.readline().strip()
        if cmd:
            try:
                result = handle_command(cmd, chat_history=chat_history, is_gradio=False)
                if isinstance(result, GeneratorType):
                    output = ''.join(result)  # TO COLLECT CHUNKS WITHOUT EXTRA NEWLINES
                    print(output, flush=True)
                else:
                    print(result, flush=True)
                print('', flush=True)  # TO ENSURE NEWLINE FOR NEXT PROMPT VISIBILITY IN BUFFERED ENVS
                sys.stdout.flush()  # EXTRA FLUSH FOR WSL RELIABILITY
            except Exception as e:
                logger.error(f"CLI command error: {e}")
                print(f"Error: {e}", flush=True)
        # TO POLL OUTPUT QUEUE NON-BLOCKING (IF USED FOR OTHER OUTPUTS)
        try:
            while not output_queue.empty():
                output = output_queue.get_nowait()
                print(output, flush=True)
        except queue.Empty:
            pass


# Excerpt from src/gia/GIA.py (complete load_database_wrapper function)
def load_database_wrapper(state: Dict) -> str:
    """Wrapper for loading database with state validation.

    Args:
        state (Dict): Gradio state dictionary (with '_gr' keys).

    Returns:
        str: Result message.
    """
    # TO LOAD STATE
    app_state = load_state()

    # TO VALIDATE STATE
    if state is None or not isinstance(state, dict):
        raise ValueError("Invalid state: expected dictionary.")

    # Only allow database load for Local mode
    model_path_gr = state.get("model_path_gr")
    # For online modes, model_path_gr may be None
    if state_manager.get_state("DATABASE_LOADED"):
        return "Database already loaded.\n"
    try:
        if app_state.LLM is None:
            mode = state.get("mode", "Local")
            if mode not in ["Local", "HuggingFace", "OpenAI", "OpenRouter"]:
                raise ValueError(f"Invalid mode: {mode}")
            config = {"model_path": model_path_gr} if mode == "Local" else {}
            msg, llm = load_llm(mode, config)
            if llm is None:
                raise RuntimeError(f"Failed to load model: {msg}")
            state_manager.set_state("LLM", llm)
    except Exception as e:
        raise RuntimeError(f"Model load error during database wrapper: {str(e)}") from e
    return load_database(app_state.LLM)

##############################################################################
# SIGNAL HANDLER & CLEANUP
##############################################################################


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

