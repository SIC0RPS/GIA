# src/gia/GIA.py
import sys
import os
import importlib
import pickle
import threading
import time
import inspect
import json
import logging
import gc
import hashlib
import queue
import traceback
from datetime import datetime
from pathlib import Path
from colorama import Style, Fore
from logging.handlers import QueueHandler, QueueListener
from queue import Queue, Empty
from typing import Dict, Callable, List, Generator, Optional, Tuple, Any, Union
from gptqmodel import GPTQModel
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import Settings
import torch
import gradio as gr
from functools import partial
import shlex
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
from gia.config import CONFIG, PROJECT_ROOT
from gia.core.state import state_manager
from gia.core.state import load_state
from gia.core.utils import (
    generate,
    update_database,
    clear_vram,
    save_database,
    load_database,
    get_system_info,
    append_to_chatbot,
    LogitsProcessor,
    LogitsProcessorList,
    PresencePenaltyLogitsProcessor
)

log_banner(__file__)

sys.path.insert(0, str(PROJECT_ROOT))

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
        self.logging_queue: Queue = Queue()  # FOR LOGS
        self.stop_event = threading.Event()  # FOR COOPERATIVE STOP
        self.listener: Optional[QueueListener] = None

    def start(self) -> None:
        """Start the sandbox thread and logging listener.

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
                self.logging_queue,
                self.stop_event,
            ),
        )
        self.thread.daemon = True
        self.thread.start()

        # TO CONFIGURE LOGGING HANDLERS - TERMINAL ONLY
        handlers = [logging.StreamHandler(sys.stdout)]
        self.listener = QueueListener(self.logging_queue, *handlers)
        self.listener.start()

    @staticmethod
    def _run_func(
        func: Callable,
        args: Tuple,
        kwargs: Dict,
        output_queue: Queue,
        input_queue: Queue,
        logging_queue: Queue,
        stop_event: threading.Event,
    ) -> None:
        """Target function for the thread: execute plugin with logging and stop check.

        Args:
            func: Plugin function.
            args: Positional args.
            kwargs: Keyword args.
            output_queue: Queue for results/streaming.
            input_queue: Queue for commands from main.
            logging_queue: Queue for logs.
            stop_event: Event to check for stop signal.
        """
        # TO SET UP LOGGING IN CHILD THREAD
        root_logger = logging.getLogger()
        queue_handler = QueueHandler(logging_queue)
        root_logger.addHandler(queue_handler)

        try:
            # DEBUG PRINT TO CONFIRM THREAD START
            logging.debug("Plugin thread started; executing function.")
            # TO EXECUTE PLUGIN FUNCTION
            result = func(*args, **kwargs)
            if inspect.isgenerator(result):
                # TO HANDLE GENERATOR WITH BIDIRECTIONAL CHECKS AND STOP EVENT
                for chunk in result:
                    if stop_event.is_set():
                        logging.info("Plugin stopped by event")
                        break
                    try:
                        cmd = input_queue.get_nowait()
                        if cmd == "stop":
                            logging.info("Plugin stopped by command")
                            break
                    except Empty:
                        pass
                    output_queue.put(chunk)
            else:
                # TO HANDLE NON-GENERATOR RESULT
                if stop_event.is_set():
                    logging.info("Plugin stopped by event")
                else:
                    output_queue.put(result)
            output_queue.put(None)  # SENTINEL FOR END
        except BaseException as exc:
            # TO SEND EXCEPTION ACROSS THREAD
            output_queue.put(exc)
            output_queue.put(None)
        except Exception as e:
            logging.error(f"Error in plugin resumption: {e}")
            output_queue.put(e)
            output_queue.put(None)

plugins: Dict[str, Callable] = {}
module_mtimes: Dict[str, float] = {}
plugins_dir_path: Path = PROJECT_ROOT / "gia" / "plugins"
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

####


# SYSTEM PROMPT - FOR CONSISTENT PROMPT LOADING
def system_prefix() -> str:
    """Load system prompt from rules file.

    Returns:
        str: System prompt string.
    """
    # TO LOAD AND PARSE RULES
    with open(RULES_PATH, "r") as f:
        rules = json.load(f)
    return "\n".join(rules["system_prompt"])


####

# Flags
METADATA_TITLE = "🛠️ Generated by GIA System (GIA.py)"
dark_gray = Style.DIM + Fore.LIGHTBLACK_EX
reset_style = Style.RESET_ALL

##################################################################################

# LOGGER SETUP - FOR CONSISTENT LOGGING
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

####


# TIMESTAMP TAG - FOR LOGGING
def tag() -> str:
    """Generate timestamp for logging."""
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

# CLI LOOP - UPDATED FOR QUEUES AND NON-BLOCKING POLL
def cli_loop():
    """Main CLI input loop with queue integration."""
    while True:
        cmd = input("CLI > ")
        if cmd.strip():
            input_queue.put(cmd)  # PUSH TO INPUT QUEUE FOR PROCESSING
            output_queue.put(f"CLI Command: {cmd}")  # ECHO TO OUTPUT FOR BIDIRECTIONAL
            # PROCESS COMMAND (UNIFIED HANDLER)
            result = handle_command(cmd, is_gradio=False)
            print(result)
        # NON-BLOCKING POLL OUTPUT QUEUE
        try:
            while not output_queue.empty():
                output = output_queue.get_nowait()
                print(output)
        except queue.Empty:
            pass


# GLOBAL QUEUES FOR BIDIRECTIONAL INTERACTION
input_queue: queue.Queue = queue.Queue()  # COMMANDS FROM CLI/GRADIO TO PLUGINS/HANDLERS
output_queue: queue.Queue = queue.Queue()  # STREAMING OUTPUTS/RESPONSES TO BOTH INTERFACES
logging_queue: queue.Queue = queue.Queue()  # LOGS TO TERMINAL ONLY
from queue import Empty

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
            state_dict = {}  # Fallback to empty dict

    # TO CHECK LLM INITIALIZATION FOR NON-LOAD COMMANDS
    if app_state.LLM is None and cmd_lower not in [".load", ".save", ".info", ".delete"]:
        # Fallback: try to generate directly if LLM is available
        if app_state.LLM is not None and cmd.strip():
            result = generate(cmd, system_prefix(), app_state.LLM, max_new_tokens=1024) + "\n"
            if is_gradio:
                if is_chat_fn:
                    yield "", chat_history, gr.State(state_dict)
                else:
                    yield chat_history, gr.State(state_dict)
                return
            return result
        error_msg = "Error: No model loaded. Please confirm a model first.\n"
        if is_gradio:
            if is_chat_fn:
                yield "", chat_history, gr.State(state_dict)
            else:
                yield chat_history, gr.State(state_dict)
            return
        return error_msg

    use_query_engine = state_dict.get("use_query_engine_gr", False)
    try:
        # TO HANDLE BUILT-IN COMMANDS
        if cmd_lower in [".load", ".save", ".info", ".delete"]:
            if cmd_lower == ".save":
                save_database()
                use_query_engine = True
                result = "Database saved. Query engine re-enabled.\n"
                state_dict["status_gr"] = "Database saved"
            elif cmd_lower == ".load":
                result = load_database_wrapper(state_dict)
                use_query_engine = True
                state_dict["status_gr"] = "Database loaded"
            elif cmd_lower == ".info":
                cpu_usage, memory_usage, gpu_usage = get_system_info()
                num_entries = app_state.CHROMA_COLLECTION.count() if app_state.CHROMA_COLLECTION else 0
                gpu_str = ' | '.join([f"GPU {i}: {load}%" for i, load in enumerate(gpu_usage)]) if gpu_usage else "No GPUs"
                result = f"[Database Entries: {num_entries} | CPU: {cpu_usage}% | RAM: {memory_usage}% | {gpu_str}]\n"
                state_dict["status_gr"] = "Info displayed"
            elif cmd_lower == ".delete":
                if app_state.CHROMA_COLLECTION:
                    app_state.CHROMA_COLLECTION.delete()
                    result = "Database successfully deleted.\n"
                    state_dict["status_gr"] = "Database deleted"
                else:
                    result = "No database to delete.\n"
                    state_dict["status_gr"] = "Error"
        # TO HANDLE PLUGIN COMMANDS (START WITH '.' BUT NOT BUILT-IN)
        elif cmd_lower.startswith('.'):
            # TO PARSE COMMAND SAFELY
            parts = shlex.split(cmd)
            if len(parts) == 0 or not parts[0].startswith('.') or len(parts[0]) <= 1:
                raise ValueError("Invalid plugin command format. Must start with '.' followed by plugin name.")
            if len(parts) > 2:
                raise ValueError("Command exceeds maximum 2 positions (plugin name and optional arg).")
            plugin_name = parts[0][1:]
            optional_arg = parts[1] if len(parts) > 1 else None
            if plugin_name not in plugins:
                raise KeyError(f"Plugin '{plugin_name}' not found in loaded plugins.")
            plugin = plugins[plugin_name]
            # TO PREPARE WHITELISTED KWARGS
            available_kwargs = {
                'llm': app_state.LLM,
                'query_engine': app_state.QUERY_ENGINE,
                'embed_model': app_state.EMBED_MODEL,
                'chat_history': chat_history,
            }
            sig = inspect.signature(plugin)
            kwargs_to_pass = {k: v for k, v in available_kwargs.items() if k in sig.parameters}
            if optional_arg is not None:
                kwargs_to_pass['arg'] = optional_arg
            # TO BIND AND CALL PLUGIN IN SANDBOX
            bound = sig.bind_partial(**kwargs_to_pass)
            sandbox = PluginSandbox(plugin_name, plugin, bound.args, bound.kwargs)
            sandbox.start()
            with sandboxes_lock:
                active_sandboxes[plugin_name] = sandbox
            try:
                collected = ""
                history = chat_history  # Shared ref; plugin appends in-place
                start_time = time.time()
                max_poll_time = 3600  # 1 hour timeout for long plugins
                while True:
                    try:
                        item = sandbox.output_queue.get(timeout=0.1)  # Non-blocking poll to avoid hang
                        if item is None:
                            break
                        if isinstance(item, BaseException):
                            raise item
                        if isinstance(item, list) and all(isinstance(d, dict) for d in item):
                            # History yielded; yield to refresh UI with appended message
                            logger.debug(f"Plugin '{plugin_name}' yielded history; yielding for UI refresh.")
                            history = item  # Update shared
                            if is_gradio:
                                if is_chat_fn:
                                    yield "", history, gr.State(state_dict)
                                else:
                                    yield history, gr.State(state_dict)
                            continue
                        msg = str(item) if not isinstance(item, str) else item  # Safe for other types
                        if msg.strip():
                            history = append_to_chatbot(history, msg, metadata={"title": METADATA_TITLE})
                        if is_gradio:
                            if is_chat_fn:
                                yield "", history, gr.State(state_dict)
                            else:
                                yield history, gr.State(state_dict)
                        else:
                            print(msg, flush=True)
                            collected += msg + "\n"
                    except Empty:
                        # Yield to keep UI responsive during long run
                        if is_gradio:
                            if is_chat_fn:
                                yield "", history, gr.State(state_dict)
                            else:
                                yield history, gr.State(state_dict)
                        time.sleep(0.1)  # Poll interval to avoid CPU spin
                        if time.time() - start_time > max_poll_time:
                            raise TimeoutError(f"Plugin {plugin_name} timed out after {max_poll_time}s")
                if not is_gradio:
                    return collected.strip()
            finally:
                sandbox.stop()
                with sandboxes_lock:
                    active_sandboxes.pop(plugin_name, None)
        # TO HANDLE QUERIES (NON-DOT COMMANDS)
        else:
            if not cmd.strip():
                result = "You entered an empty query. Please try again.\n"
                state_dict["status_gr"] = "Error"
            else:
                if use_query_engine and app_state.QUERY_ENGINE is not None:
                    response = app_state.QUERY_ENGINE.query(cmd)
                    streamed_text = ""
                    if hasattr(response, "response_gen"):
                        for chunk in response.response_gen:
                            streamed_text += chunk
                    else:
                        streamed_text = response.response if hasattr(response, 'response') else str(response)
                    result = streamed_text + "\n"
                    if streamed_text.strip() in ["", "Empty Response"]:
                        result = generate(cmd, system_prefix(), app_state.LLM, max_new_tokens=1024) + "\n"
                        state_dict["status_gr"] = "Fallback response generated"
                    else:
                        state_dict["status_gr"] = "Response generated"
                else:
                    result = generate(cmd, system_prefix(), app_state.LLM, max_new_tokens=1024) + "\n"
                    state_dict["status_gr"] = "Response generated"
        # TO UPDATE STATE FOR QUERY ENGINE
        state_dict["use_query_engine_gr"] = use_query_engine
        # TO RETURN BASED ON INTERFACE WITH DEDUP CHECK
        if result.strip() and not any(h.get("content") == result for h in chat_history[-3:]):
            append_to_chatbot(chat_history, result, metadata={"role": "assistant"})
        if is_gradio:
            if is_chat_fn:
                yield "", chat_history, gr.State(state_dict)
            else:
                yield chat_history, gr.State(state_dict)
            return
        return result
    except (ValueError, KeyError, Exception) as e:
        error_msg = f"Error processing command: {str(e)}"
        state_dict["status_gr"] = "Error"
        if is_gradio:
            if is_chat_fn:
                yield "", chat_history, gr.State(state_dict)
            else:
                yield chat_history, gr.State(state_dict)
            return
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


####

def fetch_openai_models() -> list:
    """Fetch available OpenAI models from the API for dropdown choices, only valid chat/completion models."""
    valid_models = [
        "o1", "o1-2024-12-17", "o1-pro", "o1-pro-2025-03-19", "o1-preview", "o1-preview-2024-09-12", "o1-mini", "o1-mini-2024-09-12",
        "o3-mini", "o3-mini-2025-01-31", "o3", "o3-2025-04-16", "o3-pro", "o3-pro-2025-06-10", "o4-mini", "o4-mini-2025-04-16",
        "gpt-4", "gpt-4-32k", "gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo-preview", "gpt-4-vision-preview", "gpt-4-1106-vision-preview",
        "gpt-4-turbo-2024-04-09", "gpt-4-turbo", "gpt-4o", "gpt-4o-audio-preview", "gpt-4o-audio-preview-2024-12-17", "gpt-4o-audio-preview-2024-10-01",
        "gpt-4o-mini-audio-preview", "gpt-4o-mini-audio-preview-2024-12-17", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20",
        "gpt-4.5-preview", "gpt-4.5-preview-2025-02-27", "chatgpt-4o-latest", "gpt-4o-mini", "gpt-4o-mini-2024-07-18", "gpt-4-0613",
        "gpt-4-32k-0613", "gpt-4-0314", "gpt-4-32k-0314", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1-2025-04-14",
        "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0301", "text-davinci-003", "text-davinci-002", "gpt-3.5-turbo-instruct",
        "text-ada-001", "text-babbage-001", "text-curie-001", "ada", "babbage", "curie", "davinci", "gpt-35-turbo-16k", "gpt-35-turbo",
        "gpt-35-turbo-0125", "gpt-35-turbo-1106", "gpt-35-turbo-0613", "gpt-35-turbo-16k-0613"
    ]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return valid_models
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Only include valid models present in the API response
        available = set(m["id"] for m in data.get("data", []))
        filtered = [m for m in valid_models if m in available]
        return filtered or valid_models
    except Exception as e:
        logger.warning(f"Could not fetch OpenAI models for dropdown: {e}")
        return valid_models

def load_llm(mode: str, config: Dict[str, Any]) -> Tuple[str, Any]:
    """Load LLM based on mode (local or online)."""
    if mode == "Local":
        clear_vram()
    try:
        if mode == "Local":
            model_path = config.get("model_path")
            if not model_path:
                raise ValueError("Model path required for Local mode.")
            model = GPTQModel.from_quantized(model_path, device_map=DEVICE_MAP)
            tokenizer = model.tokenizer
            shared_generate_kwargs = {
                "temperature": TEMPERATURE,
                "do_sample": True,
                "top_k": TOP_K,
                "top_p": TOP_P,
                "min_p": 0.0,
                "repetition_penalty": REPETITION_PENALTY,
                "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
                "use_cache": True,
                "eos_token_id": tokenizer.eos_token_id,
            }
            logits_processor = LogitsProcessorList([PresencePenaltyLogitsProcessor()])
            try:
                model.config.attn_implementation = "sdpa"
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                model = torch.compile(
                    model, mode="max-autotune", fullgraph=True
                )
                logger.info("Local model optimizations applied.")
            except Exception as e:
                logger.warning(f"Local optimizations skipped: {e}")
            llm = HuggingFaceLLM(
                model=model,
                tokenizer=tokenizer,
                context_window=CONTEXT_WINDOW,
                max_new_tokens=MAX_NEW_TOKENS,
                generate_kwargs={
                    **shared_generate_kwargs,
                    "logits_processor": logits_processor,
                },
                system_prompt=system_prefix(),
                device_map=DEVICE_MAP,
                model_name=model_path,
                stopping_ids=[151643, 151645, 23483, 4689, 57208],
            )
            base_name = os.path.basename(model_path)
            msg = f"Loaded local model: {base_name}"
        elif mode == "HuggingFace Online":
            token = os.getenv("HF_TOKEN")
            if not token:
                raise ValueError("HF_TOKEN not set in environment. Please set it via 'export HF_TOKEN=your_token' in your terminal.")
            model_name = config.get("model_name")
            if not model_name:
                raise ValueError("Model name required for HuggingFace Online.")
            llm = HuggingFaceInferenceAPI(
                model_name=model_name,
                token=token,
                context_window=CONTEXT_WINDOW,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_k=TOP_K,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
            )
            msg = f"Loaded HF Online model: {model_name}"
        elif mode == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in environment. Please set it via 'export OPENAI_API_KEY=sk-your_key' in your terminal.")
            model_name = config.get("model_name")
            if not model_name:
                raise ValueError("Model name required for OpenAI.")
            # Only pass supported parameters for each model family
            # o1, o3, o4 do not support temperature/top_p, etc
            if model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4"):
                llm = LlamaOpenAI(
                    model=model_name,
                    api_key=api_key
                )
            else:
                llm = LlamaOpenAI(
                    model=model_name,
                    api_key=api_key,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_NEW_TOKENS,
                    additional_kwargs={"top_p": TOP_P},
                )
            msg = f"Loaded OpenAI model: {model_name}"
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return msg, llm
    except Exception as e:
        error_msg = f"Failed to load LLM in {mode} mode: {str(e)}"
        logger.error(error_msg)
        return error_msg, None

def finalize_model_selection(
    selected_folder: str, session_state: gr.State, mode: str = "Local"
) -> Tuple[List[Dict[str, str]], gr.State, str]:
    torch.set_float32_matmul_precision('high')
    app_state = load_state()
    state_dict = session_state.value if isinstance(session_state, gr.State) else {}
    if not isinstance(state_dict, dict):
        state_dict = {"use_query_engine_gr": True, "status_gr": "", "mode": "Local"}
    state_dict["mode"] = mode
    state_manager.set_state("MODE", mode)
    logger.info(f"Model mode set to {mode} globally.")
    # Only require folder for Local mode
    if mode == "Local":
        if not selected_folder:
            return (
                [{"role": "assistant", "content": "No folder selected!"}],
                gr.State(state_dict),
                "Model: Selection Failed",
            )
        state_dict["model_path_gr"] = selected_folder
        if app_state.MODEL_PATH == selected_folder:
            base = os.path.basename(selected_folder)
            return (
                [
                    {
                        "role": "assistant",
                        "content": f"Model unchanged: {base}",
                    }
                ],
                gr.State(state_dict),
                f"Model: {base}",
            )
        config = {"model_path": selected_folder}
    elif mode == "HuggingFace Online":
        model_name = state_dict.get("hf_model_name", "deepseek-ai/DeepSeek-R1")
        config = {"model_name": model_name}
        state_dict["model_path_gr"] = None
    elif mode == "OpenAI":
        # Use fetch_openai_models for latest models
        available_models = fetch_openai_models()
        default_model = available_models[0] if available_models else "gpt-4o"
        model_name = state_dict.get("openai_model_name", default_model)
        config = {"model_name": model_name}
        state_dict["model_path_gr"] = None
    else:
        return (
            [{"role": "assistant", "content": f"Unknown mode: {mode}"}],
            gr.State(state_dict),
            "Model: Selection Failed",
        )
    msg, llm = load_llm(mode, config)
    if llm is None:
        return (
            [{"role": "assistant", "content": f"Failed to load: {msg}"}],
            gr.State(state_dict),
            "Model: Load Failed",
        )
    Settings.llm = llm
    state_manager.set_state("LLM", llm)
    state_dict["model_name_gr"] = msg.split(": ")[-1] if ": " in msg else "Unknown"
    logger.info(f"Loaded LLM type: {type(llm).__name__} for mode {mode}.")
    return (
        [{"role": "assistant", "content": msg}],
        gr.State(state_dict),
        f"Model: {state_dict['model_name_gr']}",
    )

####

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
    message: str, chat_history: List[Dict[str, str]], state: gr.State
) -> Generator[Tuple[str, List[Dict[str, str]], gr.State], None, None]:
    """Handle chat input.

    Args:
        message: User input message.
        chat_history: Chat history.
        state: Gradio state.

    Yields:
        Textbox value, updated chat history, updated state.
    """
    # TO LOAD STATE
    app_state = load_state()
    state_dict = state.value if isinstance(state.value, dict) else {}
    # TO APPEND USER MESSAGE
    append_to_chatbot(chat_history, message, metadata={"role": "user"})
    yield "", chat_history, state

    # TO PROCESS WITH SHARED HISTORY
    response, use_query_engine_gr, state, chat_history = process_input(message, state, chat_history)
    state_dict = state.value
    state_dict["use_query_engine_gr"] = use_query_engine_gr
    state_dict["status_gr"] = "Response generated"
    yield "", chat_history, gr.State(state_dict)

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
from functools import partial

def launch_app() -> None:
    """Launch Gradio application."""
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
                "hf_model_name": "meta-llama/Llama-2-7b-chat-hf",
                "openai_model_name": "gpt-3.5-turbo",
            }
        )
        chatbot = gr.Chatbot(type="messages", value=[], height=777)
        msg = gr.Textbox(placeholder="Enter your message or command (e.g., .gen)")
        submit_btn = gr.Button("Submit")

        # Add mode selection radio
        mode_radio = gr.Radio(
            choices=["Local", "HuggingFace Online", "OpenAI"],
            value="Local",
            label="Model Mode"
        )
        # Warning markdown for API keys
        warning_md = gr.Markdown("", visible=False)

        # Local UI (visible only in Local mode)
        with gr.Row(visible=True) as local_row:
            models_dir_input = gr.Textbox(label="Models Directory", value=MODEL_PATH)
            folder_dropdown = gr.Dropdown(choices=[], label="Select a Model")
            scan_dir_button = gr.Button("Scan Directory")

        # HF Online UI (hidden initially)
        with gr.Row(visible=False) as hf_row:
            hf_model_dropdown = gr.Dropdown(
                choices=[
                    "deepseek-ai/DeepSeek-R1",
                    "Qwen/Qwen3-30B-A3B-Instruct-2507",
                    "meta-llama/Llama-2-7b-chat-hf",
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    "google/gemma-7b-it",
                    "Qwen/Qwen3-8B",
                ],
                label="HF Model",
                value="deepseek-ai/DeepSeek-R1",
            )

        # OpenAI UI (hidden initially)
        with gr.Row(visible=False) as openai_row:
            openai_model_dropdown = gr.Dropdown(
                choices=fetch_openai_models(),
                label="OpenAI Model",
                value="gpt-4o",
            )

        confirm_model_button = gr.Button("Confirm Model")
        with gr.Row():
            load_button = gr.Button("Load Database")
        # PLUGIN GROUPING IN ACCORDION WITH ROW
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
        # UI logic for mode switching
        def update_mode_visibility(selected_mode: str, state):
            # Robustly handle both gr.State and dict
            if hasattr(state, 'value'):
                state_dict = state.value
            else:
                state_dict = state
            if not isinstance(state_dict, dict):
                state_dict = {"mode": selected_mode}
            state_dict["mode"] = selected_mode
            warning_value = ""
            warning_visible = False
            if selected_mode == "HuggingFace Online":
                if not os.getenv("HF_TOKEN"):
                    warning_value = "HF_TOKEN not detected. Please set it in your environment: `export HF_TOKEN=your_token_here` in your terminal and restart the app."
                    warning_visible = True
            elif selected_mode == "OpenAI":
                if not os.getenv("OPENAI_API_KEY"):
                    warning_value = "OPENAI_API_KEY not detected. Please set it in your environment: `export OPENAI_API_KEY=sk-your_key_here` in your terminal and restart the app."
                    warning_visible = True
            return (
                gr.update(visible=selected_mode == "Local"),
                gr.update(visible=selected_mode == "HuggingFace Online"),
                gr.update(visible=selected_mode == "OpenAI"),
                gr.update(value=warning_value, visible=warning_visible),
                gr.State(state_dict),
            )
        mode_radio.change(
            fn=update_mode_visibility,
            inputs=[mode_radio, session_state],
            outputs=[local_row, hf_row, openai_row, warning_md, session_state],
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
        scan_dir_button.click(
            fn=update_model_dropdown,
            inputs=[models_dir_input],
            outputs=[folder_dropdown, chatbot],
        )
        def confirm_model_handler(selected_folder, state, mode, hf_model, openai_model):
            if hasattr(state, 'value'):
                state_dict = state.value
            else:
                state_dict = state
            if mode == "HuggingFace Online":
                state_dict["hf_model_name"] = hf_model
                return finalize_model_selection(selected_folder, gr.State(state_dict), mode)
            elif mode == "OpenAI":
                state_dict["openai_model_name"] = openai_model
                return finalize_model_selection(selected_folder, gr.State(state_dict), mode)
            else:
                return finalize_model_selection(selected_folder, gr.State(state_dict), mode)
        confirm_model_button.click(
            fn=confirm_model_handler,
            inputs=[folder_dropdown, session_state, mode_radio, hf_model_dropdown, openai_model_dropdown],
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
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, debug=True)

####


# LOAD DATABASE WRAPPER - FOR DATABASE LOADING
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
    print(f"{Style.DIM + Fore.LIGHTBLACK_EX}{tag()}Cleanup completed.{Style.RESET_ALL}")


##############################################################################

if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        print(f"{dark_gray}{tag()}Starting Gradio application...{reset_style}")
        launch_app()
    except Exception as e:
        logging.exception(f"Critical error in main application: {e}")
        print(f"{tag()}A critical error occurred: {e}. Exiting.")
        traceback.print_exc()
    finally:
        cleanup()

