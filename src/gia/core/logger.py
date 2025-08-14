import sys, os, logging, traceback
from colorama import Fore, Style
from gia.config import PROJECT_ROOT, CONFIG


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
        func_display = record.funcName if record.funcName != "<module>" else record.name
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
logger.propagate = False
h = logging.StreamHandler(stream=sys.stdout)
h.setFormatter(ColoredFormatter())
logger.handlers.clear()
logger.addHandler(h)

# FIXED: SET LEVEL BASED ON CONFIG["DEBUG"] (DEBUG IF TRUE, ELSE INFO)
logger.setLevel(logging.DEBUG if CONFIG["DEBUG"] else logging.INFO)  # FIXED: TOGGLE VIA CONFIG; HIDES DEBUG LOGS WHEN FALSE

# FIXED: GLOBAL SUPPRESSION FOR MARKDOWN_IT + SUB-LOGGERS (SET TO INFO REGARDLESS; HIDES INTERNAL TRACES LIKE 'entering fence...' DYNAMICALLY)
logging.getLogger('markdown_it').setLevel(logging.INFO)  # FIXED: ROOT FOR LIB
logging.getLogger('markdown_it.rules_block').setLevel(logging.INFO)  # FIXED: SUB FOR BLOCK RULES (E.G., 'entering hr/list/reference')
logging.getLogger('markdown_it.rules_inline').setLevel(logging.INFO)  # FIXED: COVER INLINE IF NEEDED (PREVENTS ANY REMAINING FLOOD)

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