# src/gia/core/logger.py (complete module with restored functionality)
import sys
import os
import logging
import traceback
import textwrap
import platform
import subprocess
import time
import tempfile
import atexit
from colorama import Fore, Style
from gia.config import PROJECT_ROOT, CONFIG

# TO CONFIGURE ROOT LOGGER EARLY FOR GLOBAL CAPTURE
logging.basicConfig(level=logging.WARNING)
root_logger = logging.getLogger()
root_logger.propagate = False

def _anonymize_path(msg: str) -> str:
    """Replace user home directory with './' in logs."""
    home = os.path.expanduser("~")
    if home and home in msg:
        return msg.replace(home, "./")
    return msg


# TO SUPPRESS SPECIFIC THIRD-PARTY LOGGERS
third_party_loggers = [
    "PIL", "htmldate", "asyncio", "markdown_it", "markdown_it.rules_block", "markdown_it.rules_inline",
    "httpx", "httpcore", "urllib3.connectionpool", "gradio"
]
for name in third_party_loggers:
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(logging.WARNING)
    logger_obj.propagate = False

# TO FILTER DEPRECATION AND DEPENDENCY WARNINGS
class DeprecationWarningFilter(logging.Filter):
    """Filter out specific deprecation and dependency warnings."""
    def filter(self, record):
        if record.levelno == logging.WARNING:
            msg = record.getMessage()
            if any(
                s in msg for s in [
                    "websockets.legacy is deprecated",
                    "websockets.server.WebSocketServerProtocol is deprecated",
                    "urllib3 (2.0.7) or chardet (3.0.4) doesn't match a supported version",
                    "webdriver_manager"
                ]
            ):
                return False
        return True

class ColoredFormatter(logging.Formatter):
    """Format log messages with dynamic length and color based on level."""
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
        asctime = self.formatTime(record, "%H:%M:%S")
        msg = record.getMessage()
        # TO SET DYNAMIC MAX_LENGTH BASED ON LOG LEVEL
        max_length = {
            logging.DEBUG: 5000 if CONFIG["DEBUG"] else 500,
            logging.INFO: 500,
            logging.WARNING: 500,
            logging.ERROR: float('inf'),  # NO TRUNCATION
            logging.CRITICAL: float('inf'),  # NO TRUNCATION
        }.get(record.levelno, 500)
        if max_length != float('inf') and len(msg) > max_length:
            msg = msg[: max_length - 3] + f"...{Fore.WHITE}]{Style.RESET_ALL}"
        formatted = (
            f"{Fore.WHITE}[{Fore.WHITE}{asctime}{Fore.WHITE}|{Fore.CYAN}{record.levelname}{Fore.WHITE}|"
            f"{func_color}{func_display}{Fore.WHITE}|{Fore.LIGHTBLACK_EX}{msg}{Fore.WHITE}]{Style.RESET_ALL}"
        )
        return _anonymize_path(formatted)

logger = logging.getLogger("app")
logger.propagate = True
logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.INFO)

BANNER = "#" * 69
_seen = {}

def log_banner(path: str) -> None:
    """Log module load with anonymized path relative to PROJECT_ROOT."""
    real = os.path.realpath(path)
    try:
        rel_path = os.path.relpath(real, PROJECT_ROOT)
    except ValueError:
        rel_path = real  # FALLBACK IF OUTSIDE PROJECT_ROOT
    caller_path = os.path.relpath(
        os.path.realpath(sys._getframe(1).f_code.co_filename), PROJECT_ROOT
    )
    if real in _seen:
        first_module, first_path = _seen[real]
        logger.debug(
            f"\n{BANNER}\nDUPLICATE LOAD -> {rel_path}"
            f"\n\tfirst seen as module '{first_module}' from {first_path}"
            f"\n\tsecond seen as module '{sys._getframe(1).f_globals.get('__name__','?')}' from {caller_path}"
            f"\n{BANNER}"
        )
        return
    _seen[real] = (sys._getframe(1).f_globals.get("__name__", "?"), caller_path)
    logger.debug(f"\n{BANNER}\nMODULE LOADED -> {rel_path}\n{BANNER}")

from llama_index.core.callbacks.base import (
    BaseCallbackHandler,
    CBEventType,
    CallbackManager,
)
from llama_index.core import Settings
from typing import Dict

class LlamaIndexLoggerHandler(BaseCallbackHandler):
    """Handle LlamaIndex callbacks with anonymized payload paths."""
    def __init__(self, _logger: logging.Logger):
        super().__init__(
            event_starts_to_ignore=["chunking"], event_ends_to_ignore=["chunking"]
        )
        self.logger = _logger

    def _sanitize_payload(self, payload: Dict[str, any]) -> Dict[str, any]:
        """Anonymize absolute paths in payload relative to PROJECT_ROOT."""
        if not payload:
            return None
        sanitized = {}
        for key, value in payload.items():
            if key in ["chunks", "embeddings"]:
                sanitized[key] = f"<{len(value)} items>"
                continue
            if isinstance(value, str) and os.path.isabs(value):
                try:
                    sanitized[key] = os.path.relpath(value, PROJECT_ROOT)
                except ValueError:
                    sanitized[key] = "./outside_project"  # ANONYMIZE OUTSIDE PATHS
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_payload(value)
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
            _anonymize_path(f"[llama-index:{event_type}:{event_id}]\nPayload: {sanitized_payload}")
        )

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Dict[str, any],
        event_id: str,
        **kwargs,
    ) -> None:
        if payload:
            sanitized_payload = self._sanitize_payload(payload)
            self.logger.info(
                _anonymize_path(f"[llama-index:{event_type}:{event_id}]\nPayload: {sanitized_payload}")
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
    """Log uncaught exceptions to the file handler with anonymized paths."""
    trace = "".join(traceback.format_exception(exctype, exc, tb))
    logger.critical("UNCAUGHT EXCEPTION:\n%s", _anonymize_path(trace)
    )


sys.excepthook = _excepthook

# SETUP FILE LOGGING WITH SINGLE FILE AND CLEANUP
log_file = os.path.join(PROJECT_ROOT, "db", "logs", "app.log")
log_dir = os.path.dirname(log_file)
try:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)  # CREATE LOG DIRECTORY SAFELY
except Exception as e:
    logger.error(f"Failed to create log directory {log_dir}: {e}")
    raise


if os.path.exists(log_file) and os.path.getsize(log_file) > 5 * 1024 * 1024:
    try:
        os.remove(log_file)  # REMOVE IF EXCEEDS 5MB
    except Exception as e:
        logger.error(f"Failed to remove old log file: {e}")

# VIEWER CODE FOR SECONDARY TERMINAL
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
            time.sleep(1)
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
        print(f"Error in log viewer: {str(e)}")
''')

_launched = False

def is_wsl():
    """Check if running in WSL environment."""
    return 'microsoft' in platform.uname().release.lower()

def open_log_terminal():
    """Open a secondary terminal to display logs from app.log."""
    global _launched
    if _launched:
        return
    _launched = True
    system = platform.system()
    python_path = 'python'
    # CREATE TEMP SCRIPT FOR VIEWER
    try:
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as temp_file:
            temp_file.write(viewer_code)
            temp_script = temp_file.name
        cmd = [python_path, temp_script, log_file]
        # WRITE INITIAL LOG TO CONFIRM STARTUP
        with open(log_file, 'a') as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] Log viewer started.\n")
        process = None
        if system == 'Windows':
            process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif system == 'Darwin':
            apple_script = f'''
            tell application "Terminal"
                do script "{python_path} {temp_script} {log_file}"
            end tell
            '''
            process = subprocess.Popen(['osascript', '-e', apple_script])
        elif system == 'Linux':
            if is_wsl():
                wsl_cmd = ['wt.exe', 'wsl', python_path, temp_script, log_file]
                process = subprocess.Popen(wsl_cmd)
            else:
                terminal_cmd = ['xterm', '-e'] + cmd
                process = subprocess.Popen(terminal_cmd)
        else:
            raise OSError(f"Unsupported OS: {system}")
        # REGISTER CLEANUP FOR SUBPROCESS
        atexit.register(lambda: process.terminate() if process and process.poll() is None else None)
    except Exception as e:
        logger.error(f"Failed to open log terminal: {e}")

for handler in list(root_logger.handlers):
    if isinstance(handler, logging.StreamHandler):
        root_logger.removeHandler(handler)
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.addFilter(DeprecationWarningFilter())
file_handler.setFormatter(ColoredFormatter())
root_logger.addHandler(file_handler)

for name in third_party_loggers:
    logging.getLogger(name).addHandler(file_handler)

gptq_logger = logging.getLogger("gptqmodel")
gptq_logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.INFO)
gptq_logger.propagate = False
gptq_logger.addHandler(file_handler)

if CONFIG["DEBUG"]:
    open_log_terminal()