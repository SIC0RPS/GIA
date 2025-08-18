# src/gia/core/logger.py
import sys
import os
import logging
import traceback
import textwrap
import platform
import subprocess
import time
import tempfile
import atexit  # ADDED FOR SAFE TEMP FILE CLEANUP IN VIEWER

from colorama import Fore, Style
from gia.config import PROJECT_ROOT, CONFIG

# TO CONFIGURE ROOT LOGGER EARLY FOR GLOBAL CAPTURE (EVIDENCE: PYTHON LOGGING HOW-TO; PREVENTS THIRD-PARTY STDOUT)
logging.basicConfig(level=logging.WARNING)  # SET ROOT TO WARNING FOR UNHANDLED/THIRD-PARTY LOGS (WARNING+ ONLY)
root_logger = logging.getLogger()
root_logger.propagate = False  # TO STOP ROOT PROPAGATION; CONTROL VIA EXPLICIT HANDLERS/LEVELS

# TO SUPPRESS SPECIFIC THIRD-PARTY LOGGERS TO WARNING (WARNING/ERROR ONLY; GROUPED FOR MAINTAINABILITY; EVIDENCE: PILLOW SOURCE ON GITHUB, SETLEVEL PREVENTS DEBUG IMPORTS)
third_party_loggers = [
    "PIL", "htmldate", "asyncio", "markdown_it", "markdown_it.rules_block", "markdown_it.rules_inline",
    "httpx", "httpcore", "urllib3.connectionpool", "gradio"
]
for name in third_party_loggers:
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(logging.WARNING)
    logger_obj.propagate = False

class ColoredFormatter(logging.Formatter):
    MAX_LENGTH = 1000
    LEVEL_COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED,
    }

    def format(self, record):
        func_display = record.funcName if record.funcName != "" else record.name
        func_color = self.LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        asctime = self.formatTime(record, "%H:%M:%S")  # FIXED: SHORT TIME ONLY (HH:MM:SS, NO DATE)
        msg = record.getMessage()
        if len(msg) > self.MAX_LENGTH:
            msg = msg[: self.MAX_LENGTH - 3] + f"...{Fore.WHITE}]{Style.RESET_ALL}"
        formatted = (
            f"\n{Fore.WHITE}[{Fore.CYAN}{asctime}{Fore.WHITE}|{Fore.CYAN}{record.levelname}{Fore.WHITE}|"
            f"{func_color}{func_display}{Fore.WHITE}|{Fore.LIGHTBLACK_EX}{msg}{Fore.WHITE}]{Style.RESET_ALL}\n"
        )  # FIXED: NEW FORMAT [HH:MM:SS|LEVEL|funcName|message]
        return formatted

logger = logging.getLogger("app")
logger.propagate = True  # TO ALLOW APP LOGS TO PROPAGATE TO ROOT HANDLER (SECOND TERMINAL); CHANGED FROM FALSE
logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.INFO)  # APP LOGS AT DEBUG IF ENABLED

BANNER = "#" * 69
_seen = {}

def log_banner(path: str) -> None:
    real = os.path.realpath(path)
    rel_path = os.path.relpath(real, PROJECT_ROOT)
    caller_path = os.path.relpath(
        os.path.realpath(sys._getframe(1).f_code.co_filename), PROJECT_ROOT
    )
    if real in _seen:
        first_module, first_path = _seen[real]
        logger.debug(
            f"\n{BANNER}\nDUPLICATE LOAD  -> {rel_path}"
            f"\n\tfirst seen as module '{first_module}' from {first_path}"
            f"\n\tsecond seen as module '{sys._getframe(1).f_globals.get('__name__','?')}' from {caller_path}"
            f"\n{BANNER}"
        )
        return
    _seen[real] = (sys._getframe(1).f_globals.get("__name__", "?"), caller_path)
    logger.debug(f"\n{BANNER}\nMODULE LOADED  -> {rel_path}\n{BANNER}")

from llama_index.core.callbacks.base import (
    BaseCallbackHandler,
    CBEventType,
    CallbackManager,
)
from llama_index.core import Settings
from typing import Dict

class LlamaIndexLoggerHandler(BaseCallbackHandler):
    def __init__(self, _logger: logging.Logger):
        super().__init__(
            event_starts_to_ignore=["chunking"], event_ends_to_ignore=["chunking"]
        )
        self.logger = _logger

    def _sanitize_payload(self, payload: Dict[str, any]) -> Dict[str, any]:
        if not payload:
            return None
        sanitized = {}
        for key, value in payload.items():
            # Skip large data like embeddings or chunks
            if key in ["chunks", "embeddings"]:
                sanitized[key] = f"<{len(value)} items>"
                continue
            # Make paths relative to PROJECT_ROOT
            if isinstance(value, str) and os.path.isabs(value):
                try:
                    sanitized[key] = os.path.relpath(value, PROJECT_ROOT)
                except ValueError:
                    sanitized[key] = value  # Path outside PROJECT_ROOT, keep as is
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_payload(value)  # Recursive sanitization
            else:
                sanitized[key] = value
        return sanitized

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Dict[str, any],
        event_id: str,
        **kwargs,
    ) -> None:
        sanitized_payload = self._sanitize_payload(payload)
        self.logger.info(
            f"[llama-index:{event_type}:{event_id}]\nPayload: {sanitized_payload}"
        )

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Dict[str, any],
        event_id: str,
        **kwargs,
    ) -> None:
        if payload:  # Only log END if payload has meaningful data
            sanitized_payload = self._sanitize_payload(payload)
            self.logger.info(
                f"[llama-index:{event_type}:{event_id}]\nPayload: {sanitized_payload}"
            )

    def start_trace(self, trace_id: str, **kwargs) -> None:
        self.logger.info(f"[llama-index:TRACE-START:{trace_id}]")

    def end_trace(self, trace_id: str, **kwargs) -> None:
        self.logger.info(f"[llama-index:TRACE-END:{trace_id}]")

try:
    Settings.callback_manager = CallbackManager([LlamaIndexLoggerHandler(logger)])
    logger.debug(f"[LOG-SETUP] callback stack -> {Settings.callback_manager.handlers}")
except Exception as e:
    logger.exception("Failed to attach LlamaIndexLoggerHandler: %s", e)

def _excepthook(exctype, exc, tb):
    logger.critical(
        "UNCAUGHT EXCEPTION:\n%s", "".join(traceback.format_exception(exctype, exc, tb))
    )

sys.excepthook = _excepthook

# Additions for secondary terminal logging

log_file = 'app_logs.txt'

viewer_code = textwrap.dedent('''
import sys
import time
import os
import atexit

# TO DELETE SELF ON EXIT TO AVOID RACE CONDITION
atexit.register(lambda: os.unlink(sys.argv[0]) if os.path.exists(sys.argv[0]) else None)

if len(sys.argv) < 2:
    print("Error: No log file provided.")
    sys.exit(1)

log_file = sys.argv[1]
try:
    if not os.path.exists(log_file):
        print("Log file not found. Waiting...")
        time.sleep(0.1)  # SHORTER WAIT FOR LOG FILE TO APPEAR
    with open(log_file, 'r') as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            print(line, end="")
except KeyboardInterrupt:
    print("\\nLog viewer stopped.")
except Exception as e:
    print(f"Error in log viewer: {e}")
''')

_launched = False

def is_wsl():
    return 'microsoft' in platform.uname().release.lower()

def open_log_terminal():
    global _launched
    if _launched:
        return
    _launched = True
    system = platform.system()
    python_path = 'python'  # Or 'python3' if needed
    # TO CREATE TEMP SCRIPT TO AVOID -C QUOTING ISSUES
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as temp_file:
        temp_file.write(viewer_code)
        temp_script = temp_file.name
    cmd = [python_path, temp_script, log_file]
    if system == 'Windows':
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    elif system == 'Darwin':  # macOS
        apple_script = f'''
        tell application "Terminal"
            do script "{python_path} {temp_script} {log_file}"
        end tell
        '''
        subprocess.Popen(['osascript', '-e', apple_script])
    elif system == 'Linux':
        if is_wsl():
            wsl_cmd = ['wt.exe', 'wsl', python_path, temp_script, log_file]
            subprocess.Popen(wsl_cmd)
        else:
            terminal_cmd = ['xterm', '-e'] + cmd
            subprocess.Popen(terminal_cmd)
    else:
        raise OSError(f"Unsupported OS: {system}")
    # NO UNLINK HERE; HANDLED BY VIEWER ITSELF VIA ATEXIT

# Setup secondary logging: Remove stream handler, add file handler to root, open terminal
for handler in list(root_logger.handlers):  
    if isinstance(handler, logging.StreamHandler):
        root_logger.removeHandler(handler)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(ColoredFormatter())
root_logger.addHandler(file_handler)  # ALL LOGS (APP/THIRD-PARTY/UNEXPECTED) TO SECOND TERMINAL VIA ROOT
open_log_terminal()

# TO ADD FILE HANDLER TO THIRD-PARTY LOGGERS AFTER ROOT SETUP (FOR CONSISTENT ROUTING TO SECOND TERMINAL)
for name in third_party_loggers:
    logging.getLogger(name).addHandler(file_handler)

# SPECIAL SETUP FOR GPTQMODEL TO SHOW ALL LOGS (DEBUG IF CONFIG["DEBUG"], ELSE INFO)
gptq_logger = logging.getLogger("gptqmodel")
gptq_logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.INFO)
gptq_logger.propagate = False
gptq_logger.addHandler(file_handler)