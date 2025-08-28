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
import atexit
import inspect  # TO SUPPORT FUNCTION SIGNATURE EXTRACTION FOR DEBUG LOGS
from colorama import Fore, Style
from gia.config import PROJECT_ROOT, CONFIG

# TO CONFIGURE ROOT LOGGER EARLY FOR GLOBAL CAPTURE
logging.basicConfig(level=logging.WARNING)
root_logger = logging.getLogger()
root_logger.propagate = False

def _anonymize_path(msg: str) -> str:
    """Replace user home directory with './' in logs without extra '/'."""
    home = os.path.expanduser("~")
    if home and home in msg:
        # TO REPLACE HOME + SEP WITH './' TO AVOID DOUBLE '/'
        msg = msg.replace(home + os.sep, './')
        msg = msg.replace(home, './')  # FALLBACK FOR NON-PATH CASES
    return msg

# TO SUPPRESS SPECIFIC THIRD-PARTY LOGGERS (EXCEPT HTTPX FOR DETAILED LOGS)
third_party_loggers = [
    "PIL", "htmldate", "asyncio", "markdown_it", "markdown_it.rules_block", "markdown_it.rules_inline",
    "httpcore", "urllib3.connectionpool", "gradio"
]
for name in third_party_loggers:
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(logging.WARNING)
    logger_obj.propagate = False

# SPECIAL SETUP FOR HTTPX TO SHOW DETAILED HTTP REQUESTS IN DEBUG MODE (WITH 'ONLINE' PREFIX IF APPLICABLE)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.WARNING)
httpx_logger.propagate = False
class HttpxFilter(logging.Filter):
    """Add 'ONLINE' prefix to httpx logs for HTTP requests."""
    def filter(self, record):
        if "Sending request" in record.getMessage() or "Received response" in record.getMessage():
            record.msg = f"ONLINE {record.msg}"
        return True
httpx_logger.addFilter(HttpxFilter())

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

# TO ADD FUNCTION SIGNATURE TO DEBUG LOGS IN DEBUG MODE
class SignatureFilter(logging.Filter):
    """Attach function signature to LogRecord for DEBUG level in DEBUG mode."""
    def filter(self, record):
        if CONFIG["DEBUG"]:
            try:
                frame = sys._getframe(3)  # GET CALLER FRAME (SKIPPING LOGGING STACK)
                func = frame.f_code
                sig = inspect.signature(func)
                # TO FORMAT SIGNATURE AS (param1, param2, etc) WITH VARARGS HANDLING
                params = []
                for param in sig.parameters.values():
                    if param.kind == inspect.Parameter.VAR_POSITIONAL:
                        params.append(f'*{param.name}')
                    elif param.kind == inspect.Parameter.VAR_KEYWORD:
                        params.append(f'**{param.name}')
                    else:
                        params.append(str(param).split('=')[0])  # OMIT DEFAULTS FOR BREVITY
                record.signature = f'({", ".join(params)})'
            except Exception:
                record.signature = ''
        else:
            record.signature = ''
        return True

class ColoredFormatter(logging.Formatter):
    """Format log messages with dynamic length and color based on level.
    Falls back to plain formatting when NO_COLOR is set or stdout is not a TTY.
    """
    LEVEL_COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED,
    }

    def format(self, record):
        # AUTO-DISABLE COLOR IN NON-TTY OR WHEN NO_COLOR IS SET
        use_color = (
            hasattr(sys, "stdout")
            and sys.stdout is not None
            and getattr(sys.stdout, "isatty", lambda: False)()
            and not os.environ.get("NO_COLOR")
            and not CONFIG.get("NO_COLOR", False)
        )

        func_display = record.funcName if record.funcName != "" else record.name
        if hasattr(record, 'signature') and record.signature:
            func_display += record.signature
        asctime = self.formatTime(record, "%H:%M:%S")
        msg = _anonymize_path(record.getMessage().replace("\n", "\\n"))
        if CONFIG["DEBUG"] and record.exc_info and record.levelno >= logging.WARNING:
            exc_text = self.formatException(record.exc_info)
            exc_text = _anonymize_path(exc_text)
            msg += f"\n{exc_text}"
        max_length = {
            logging.DEBUG: 5000 if CONFIG["DEBUG"] else 500,
            logging.INFO: 500,
            logging.WARNING: float('inf') if CONFIG["DEBUG"] else 500,
            logging.ERROR: float('inf'),
            logging.CRITICAL: float('inf'),
        }.get(record.levelno, 500)
        if max_length != float('inf') and len(msg) > max_length:
            msg = msg[: max_length - 3] + "..."

        if not use_color:
            # PLAIN FALLBACK IN CRITICAL/NON-TTY PATHS
            return f"[{asctime}|{record.levelname}|{func_display}|{msg}]"

        func_color = self.LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        formatted = (
            f"{Fore.WHITE}[{Fore.WHITE}{asctime}{Fore.WHITE}|{Fore.CYAN}{record.levelname}{Fore.WHITE}|"
            f"{func_color}{func_display}{Fore.WHITE}|{Fore.LIGHTBLACK_EX}{msg}{Fore.WHITE}]{Style.RESET_ALL}"
        )
        return formatted

class PlainFormatter(logging.Formatter):
    """Format log messages without colors, with dynamic length truncation."""
    def format(self, record):
        func_display = record.funcName if record.funcName != "" else record.name
        # TO APPEND SIGNATURE IF AVAILABLE
        if hasattr(record, 'signature') and record.signature:
            func_display += record.signature
        asctime = self.formatTime(record, "%H:%M:%S")
        msg = _anonymize_path(record.getMessage().replace("\n", "\\n"))
        # TO INCLUDE FULL TRACEBACK FOR WARNING/ERROR/CRITICAL IN DEBUG MODE
        if CONFIG["DEBUG"] and record.exc_info and record.levelno >= logging.WARNING:
            exc_text = self.formatException(record.exc_info)
            exc_text = _anonymize_path(exc_text)
            msg += f"\n{exc_text}"
        # TO SET DYNAMIC MAX_LENGTH BASED ON LOG LEVEL (NO TRUNCATION FOR WARNING IN DEBUG)
        max_length = {
            logging.DEBUG: 5000 if CONFIG["DEBUG"] else 500,
            logging.INFO: 500,
            logging.WARNING: float('inf') if CONFIG["DEBUG"] else 500,
            logging.ERROR: float('inf'),  # NO TRUNCATION
            logging.CRITICAL: float('inf'),  # NO TRUNCATION
        }.get(record.levelno, 500)
        if max_length != float('inf') and len(msg) > max_length:
            msg = msg[: max_length - 3] + "..."
        formatted = (
            f"[{asctime}|{record.levelname}|{func_display}|{msg}]"
        )
        return formatted

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
    logger.critical("UNCAUGHT EXCEPTION:\n%s", _anonymize_path(trace))

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
if os.path.exists(log_file):
    try:
        os.remove(log_file)
    except Exception as e:
        logger.error(f"Failed to remove old log file: {e}")

# VIEWER CODE FOR SECONDARY TERMINAL

viewer_code = textwrap.dedent('''
    import sys
    import time
    import os
    import atexit
    try:
        from colorama import Fore, Style, init
    except Exception:
        # If colorama missing, run plain
        Fore = Style = None
        def init(*a, **k): pass

    init(autoreset=False)

    # COLOR CONTROL: disable if NO_COLOR or non-TTY
    use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR", "")

    def _plain(asctime, level, func, msg):
        return f"[{asctime}|{level}|{func}|{msg}]"

    def _colored(asctime, level, func, msg):
        # If color disabled or colorama unavailable, return plain
        if not use_color or Fore is None or Style is None:
            return _plain(asctime, level, func, msg)
        LEVEL_COLORS = {
            'DEBUG': Fore.BLUE,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED,
        }
        level_color = LEVEL_COLORS.get(level, Fore.CYAN)
        func_color = LEVEL_COLORS.get(level, Fore.WHITE)
        return (
            f"{Fore.WHITE}[{Fore.WHITE}{asctime}{Fore.WHITE}|{level_color}{level}{Fore.WHITE}|"
            f"{func_color}{func}{Fore.WHITE}|{Fore.LIGHTBLACK_EX}{msg}{Fore.WHITE}]{Style.RESET_ALL}"
        )

    # TO DELETE SELF ON EXIT TO AVOID RACE CONDITION
    atexit.register(lambda: os.unlink(sys.argv[0]) if os.path.exists(sys.argv[0]) else None)

    if len(sys.argv) < 2:
        try:
            print("Error: No log file provided.")
        except OSError:
            pass
        sys.exit(1)

    log_file = sys.argv[1]

    try:
        if not os.path.exists(log_file):
            try:
                print("Log file not found. Waiting...")
            except OSError:
                pass
            time.sleep(1)
        with open(log_file, 'r', encoding='utf-8') as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                line = line.rstrip()
                out = line
                if line.startswith('[') and line.endswith(']'):
                    parts = line[1:-1].split('|', 3)
                    if len(parts) == 4:
                        asctime, level, func, msg = parts
                        msg = msg.replace('\\\\n', '\\n')
                        out = _colored(asctime, level, func, msg)
                try:
                    print(out)
                except OSError:
                    # CRITICAL PATH: permanently disable color and retry plain once
                    use_color = False
                    try:
                        print(out if out == line else _plain(asctime, level, func, msg))
                    except OSError:
                        # stdout is gone; exit quietly
                        break
    except KeyboardInterrupt:
        try:
            print("\\nLog viewer stopped.")
        except OSError:
            pass
    except Exception as e:
        try:
            print(f"Error in log viewer: {str(e)}")
        except OSError:
            pass
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
file_handler.addFilter(SignatureFilter())  # ADD SIGNATURE FILTER FOR ALL LOGS IN DEBUG MODE
file_handler.setFormatter(PlainFormatter())
root_logger.addHandler(file_handler)
for name in third_party_loggers:
    logging.getLogger(name).addHandler(file_handler)
gptq_logger = logging.getLogger("gptqmodel")
gptq_logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.INFO)
gptq_logger.propagate = False
gptq_logger.addHandler(file_handler)
if CONFIG["DEBUG"]:
    open_log_terminal()