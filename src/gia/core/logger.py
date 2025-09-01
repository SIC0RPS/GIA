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
import inspect
from typing import Dict
from colorama import Fore, Style
from gia.config import PROJECT_ROOT, CONFIG
from llama_index.core.callbacks.base import (
    BaseCallbackHandler,
    CBEventType,
    CallbackManager,
)
from llama_index.core import Settings

logging.basicConfig(level=logging.WARNING)
root_logger = logging.getLogger()
root_logger.propagate = False
# Route Python warnings into logging; honor DEBUG for global verbosity
logging.captureWarnings(True)
root_logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.INFO)


def _anonymize_path(msg: str) -> str:
    """Replace user home directory with './' in logs."""
    home = os.path.expanduser("~")
    if home and home in msg:
        msg = msg.replace(home + os.sep, "./")
        msg = msg.replace(home, "./")
    return msg


# Proven policy: route libraries to root; mute only noisy ones
NOISY_LOGGERS = [
    "PIL",
    "htmldate",
    "asyncio",
    "markdown_it",
    "markdown_it.rules_block",
    "markdown_it.rules_inline",
    "gradio",
]
VERBOSE_LOGGERS = [
    "httpx",
    "httpcore",
    "urllib3",
    "urllib3.connectionpool",
    "llama_index",
    "gptqmodel",
    "py.warnings",
]


def _configure_library_loggers(debug: bool) -> None:
    """
    Route library logs to root; mute only known noisy libs.
    In DEBUG: verbose libs at DEBUG, noisy libs at WARNING.
    In non-DEBUG: all libs at WARNING.
    """
    default_level = logging.DEBUG if debug else logging.WARNING

    for name in VERBOSE_LOGGERS:
        lg = logging.getLogger(name)
        lg.setLevel(default_level)
        lg.propagate = True
        if lg.handlers:
            lg.handlers.clear()

    for name in NOISY_LOGGERS:
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        lg.propagate = True
        if lg.handlers:
            lg.handlers.clear()


def enforce_file_only_handlers() -> None:
    """Remove all non-file StreamHandlers across all loggers and force propagation."""
    # Clean root explicitly
    for h in list(root_logger.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)

    # Iterate all known loggers and strip console handlers
    mgr = logging.root.manager.loggerDict
    for name, obj in list(mgr.items()):
        if not isinstance(obj, logging.Logger):
            continue
        for h in list(obj.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                obj.removeHandler(h)
        # Ensure records bubble to root file handler
        obj.propagate = True


httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.WARNING)
httpx_logger.propagate = True


class HttpxFilter(logging.Filter):
    """Add 'ONLINE' prefix to httpx logs for HTTP requests."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Sending request" in msg or "Received response" in msg:
            record.msg = f"ONLINE {record.msg}"
        return True


httpx_logger.addFilter(HttpxFilter())


class DeprecationWarningFilter(logging.Filter):
    """Filter out specific deprecation and dependency warnings."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno == logging.WARNING:
            msg = record.getMessage()
            if any(
                s in msg
                for s in [
                    "websockets.legacy is deprecated",
                    "websockets.server.WebSocketServerProtocol is deprecated",
                    "urllib3 (2.0.7) or chardet (3.0.4) doesn't match a supported version",
                    "webdriver_manager",
                ]
            ):
                return False
        return True


class SignatureFilter(logging.Filter):
    """Attach function signature to LogRecord for DEBUG level in DEBUG mode."""

    def filter(self, record: logging.LogRecord) -> bool:
        if CONFIG["DEBUG"]:
            try:
                frame = sys._getframe(3)
                func = frame.f_code  # type: ignore[assignment]
                # inspect.signature on code objects will raise; safely ignore
                sig = inspect.signature(func)  # noqa: F841
                params = []
                for param in sig.parameters.values():  # type: ignore[attr-defined]
                    if param.kind == inspect.Parameter.VAR_POSITIONAL:
                        params.append(f"*{param.name}")
                    elif param.kind == inspect.Parameter.VAR_KEYWORD:
                        params.append(f"**{param.name}")
                    else:
                        params.append(str(param).split("=")[0])
                record.signature = f"({', '.join(params)})"
            except Exception:
                record.signature = ""
        else:
            record.signature = ""
        return True


class ColoredFormatter(logging.Formatter):
    """
    Color formatter for console streams. Not used by default (CLI kept clean),
    but preserved for optional direct CLI logging if needed.
    """

    LEVEL_COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        use_color = (
            hasattr(sys, "stdout")
            and sys.stdout is not None
            and getattr(sys.stdout, "isatty", lambda: False)()
            and not os.environ.get("NO_COLOR")
            and not CONFIG.get("NO_COLOR", False)
        )

        func_display = record.funcName if record.funcName != "" else record.name
        if hasattr(record, "signature") and record.signature:
            func_display += record.signature
        asctime = self.formatTime(record, "%H:%M:%S")
        msg = _anonymize_path(record.getMessage().replace("\n", "\\n"))
        if CONFIG["DEBUG"] and record.exc_info and record.levelno >= logging.WARNING:
            exc_text = self.formatException(record.exc_info)
            exc_text = _anonymize_path(exc_text)
            msg += f"\n{exc_text}"
        max_length = {
            logging.DEBUG: 5000 if CONFIG["DEBUG"] else 1000,
            logging.INFO: 1000,
            logging.WARNING: float("inf") if CONFIG["DEBUG"] else 1000,
            logging.ERROR: float("inf"),
            logging.CRITICAL: float("inf"),
        }.get(record.levelno, 500)
        if max_length != float("inf") and len(msg) > max_length:
            msg = msg[: max_length - 3] + "..."

        if not use_color:
            return f"[{asctime}|{record.levelname}|{func_display}|{msg}]"

        func_color = self.LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        formatted = (
            f"{Fore.WHITE}[{Fore.WHITE}{asctime}{Fore.WHITE}|{Fore.CYAN}{record.levelname}{Fore.WHITE}|"
            f"{func_color}{func_display}{Fore.WHITE}|{Fore.LIGHTBLACK_EX}{msg}{Fore.WHITE}]{Style.RESET_ALL}"
        )
        return formatted


class PlainFormatter(logging.Formatter):
    """Format log messages without colors, with dynamic length truncation."""

    def format(self, record: logging.LogRecord) -> str:
        func_display = record.funcName if record.funcName != "" else record.name
        if hasattr(record, "signature") and record.signature:
            func_display += record.signature
        asctime = self.formatTime(record, "%H:%M:%S")
        msg = _anonymize_path(record.getMessage().replace("\n", "\\n"))
        if CONFIG["DEBUG"] and record.exc_info and record.levelno >= logging.WARNING:
            exc_text = self.formatException(record.exc_info)
            exc_text = _anonymize_path(exc_text)
            msg += f"\n{exc_text}"
        max_length = {
            logging.DEBUG: 5000 if CONFIG["DEBUG"] else 1000,
            logging.INFO: 1000,
            logging.WARNING: float("inf") if CONFIG["DEBUG"] else 1000,
            logging.ERROR: float("inf"),
            logging.CRITICAL: float("inf"),
        }.get(record.levelno, 1000)
        if max_length != float("inf") and len(msg) > max_length:
            msg = msg[: max_length - 3] + "..."
        return f"[{asctime}|{record.levelname}|{func_display}|{msg}]"


logger = logging.getLogger("app")
logger.propagate = True
logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.INFO)

BANNER = "#" * 69
_seen: Dict[str, tuple] = {}


def log_banner(path: str) -> None:
    """Log module load with anonymized path relative to PROJECT_ROOT."""
    real = os.path.realpath(path)
    try:
        rel_path = os.path.relpath(real, PROJECT_ROOT)
    except ValueError:
        rel_path = real
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


class LlamaIndexLoggerHandler(BaseCallbackHandler):
    """Handle LlamaIndex callbacks with anonymized payload paths."""

    def __init__(self, _logger: logging.Logger):
        super().__init__(event_starts_to_ignore=["chunking"], event_ends_to_ignore=["chunking"])
        self.logger = _logger

    def _sanitize_payload(self, payload: Dict[str, any]) -> Dict[str, any]:
        """Anonymize absolute paths in payload relative to PROJECT_ROOT."""
        if not payload:
            return None
        sanitized: Dict[str, any] = {}
        for key, value in payload.items():
            if key in ["chunks", "embeddings"]:
                sanitized[key] = f"<{len(value)} items>"
                continue
            if isinstance(value, str) and os.path.isabs(value):
                try:
                    sanitized[key] = os.path.relpath(value, PROJECT_ROOT)
                except ValueError:
                    sanitized[key] = "./outside_project"
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


def _excepthook(exctype, exc, tb) -> None:
    """Log uncaught exceptions to the file handler with anonymized paths."""
    trace = "".join(traceback.format_exception(exctype, exc, tb))
    logger.critical("UNCAUGHT EXCEPTION:\n%s", _anonymize_path(trace))


sys.excepthook = _excepthook

# SETUP FILE LOGGING WITH SINGLE FILE AND CLEANUP
log_file = os.path.join(PROJECT_ROOT, "db", "logs", "app.log")
log_dir = os.path.dirname(log_file)
try:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
except Exception as e:
    logger.error(f"Failed to create log directory {log_dir}: {e}")
    raise
if os.path.exists(log_file):
    try:
        os.remove(log_file)
    except Exception as e:
        logger.error(f"Failed to remove old log file: {e}")

# RICH-POWERED VIEWER CODE FOR SECONDARY TERMINAL
viewer_code = textwrap.dedent(
    """
    import sys
    import time
    import os
    import atexit
    try:
        from rich.console import Console
        from rich.text import Text
    except Exception:
        print("Rich is required for the debug log viewer. `pip install rich`")
        sys.exit(1)

    console = Console(force_terminal=True)

    def _render(asctime: str, level: str, func: str, msg: str):
        # Rehydrate escaped newlines
        msg = msg.replace('\\\\n', '\\n')

        level_styles = {
            "DEBUG": "blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red",
        }
        lv_style = level_styles.get(level, "cyan")

        t = Text()
        t.append("[", style="white")
        t.append(asctime, style="white")
        t.append("|", style="white")
        t.append(level, style=lv_style)
        t.append("|", style="white")
        t.append(func, style=lv_style)
        t.append("|", style="white")

        if msg.startswith("ONLINE "):
            t.append("ONLINE ", style="bold cyan")
            t.append(msg[7:], style="bright_black")
        else:
            t.append(msg, style="bright_black")
        t.append("]", style="white")
        return t

    atexit.register(lambda: os.unlink(sys.argv[0]) if os.path.exists(sys.argv[0]) else None)

    if len(sys.argv) < 2:
        console.print("Error: No log file provided.", style="red")
        sys.exit(1)

    log_file = sys.argv[1]

    try:
        if not os.path.exists(log_file):
            console.print("Log file not found. Waiting...", style="yellow")
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
                        out = _render(asctime, level, func, msg)

                if isinstance(out, Text):
                    console.print(out, soft_wrap=True, overflow="fold")
                else:
                    console.print(out)
    except KeyboardInterrupt:
        console.print("\\nLog viewer stopped.", style="yellow")
    except Exception as e:
        console.print(f"Error in log viewer: {str(e)}", style="red")
    """
)

_launched = False


def is_wsl() -> bool:
    """Check if running in WSL environment."""
    return "microsoft" in platform.uname().release.lower()


def open_log_terminal() -> None:
    """Open a secondary terminal to display logs from app.log."""
    global _launched
    if _launched:
        return
    _launched = True
    system = platform.system()
    python_path = "python"
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as temp_file:
            temp_file.write(viewer_code)
            temp_script = temp_file.name
        cmd = [python_path, temp_script, log_file]
        # Write a simple marker so file exists before tailing
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] Log viewer started.\n")

        process = None
        if system == "Windows":
            process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif system == "Darwin":
            apple_script = f"""
            tell application "Terminal"
                do script "{python_path} {temp_script} {log_file}"
            end tell
            """
            process = subprocess.Popen(["osascript", "-e", apple_script])
        elif system == "Linux":
            if is_wsl():
                wsl_cmd = ["wt.exe", "wsl", python_path, temp_script, log_file]
                process = subprocess.Popen(wsl_cmd)
            else:
                terminal_cmd = ["xterm", "-e"] + cmd
                process = subprocess.Popen(terminal_cmd)
        else:
            raise OSError(f"Unsupported OS: {system}")

        atexit.register(lambda: process.terminate() if process and process.poll() is None else None)
    except Exception as e:
        logger.error(f"Failed to open log terminal: {e}")


for handler in list(root_logger.handlers):
    if isinstance(handler, logging.StreamHandler):
        root_logger.removeHandler(handler)

file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.addFilter(DeprecationWarningFilter())
file_handler.addFilter(SignatureFilter())
file_handler.setFormatter(PlainFormatter())
root_logger.addHandler(file_handler)

# Configure libraries once, then enforce file-only handlers globally
_configure_library_loggers(CONFIG["DEBUG"])
enforce_file_only_handlers()

# Ensure key libraries bubble up; avoid their own handlers
gptq_logger = logging.getLogger("gptqmodel")
gptq_logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.INFO)
gptq_logger.propagate = True
if gptq_logger.handlers:
    gptq_logger.handlers.clear()

llama_logger = logging.getLogger("llama_index")
llama_logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.INFO)
llama_logger.propagate = True
if llama_logger.handlers:
    llama_logger.handlers.clear()

if CONFIG["DEBUG"]:
    open_log_terminal()