from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler


# Resolve the logs directory relative to this file (backend/logs), not the
# process's current working directory — a bare Path("logs") depends on
# wherever uvicorn happens to be launched from, which is fragile (it
# happened to line up with the Docker WORKDIR by coincidence, but breaks
# for local `uvicorn app.main:app` runs from a different directory).
BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """
    Wires up root logging with a rotating file handler plus console output.

    This used to be module-level code that ran as a side effect of
    `import app.logging_config` — but nothing in the app ever imported this
    module, so it never actually ran. Python's root logger defaults to
    WARNING with no handlers, which meant every `logger.info(...)` call
    throughout the app (in routes.py, etc.) was silently dropped.

    Call this explicitly and early — from main.py, before anything else
    logs — so it's obvious where logging gets configured and that it
    actually is.
    """
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = RotatingFileHandler(
        filename=LOG_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,  # Rotate after 10 MB
        backupCount=5,              # Keep the last 5 log files
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        handlers=[file_handler, console_handler],
        force=True,  # Override any handlers a library may have already attached
    )

    _configured = True