"""
src/run_logger.py — Logging bridge between the pipeline and Streamlit.

Every src/ module calls:
    logger = logging.getLogger("pipeline")

When running in Streamlit, app.py attaches a StreamlitLogHandler to that
logger so every log line appears live in the st.status panel.
When running from CLI scripts the same logger prints to stdout.
"""
import logging


class StreamlitLogHandler(logging.Handler):
    """Appends each log record as a line inside a Streamlit container."""

    def __init__(self, container):
        super().__init__()
        self._container = container
        self._lines: list[str] = []
        self.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._lines.append(msg)
            # Render the full log as a code block so it scrolls naturally
            self._container.code("\n".join(self._lines), language=None)
        except Exception:
            pass


def attach_streamlit_handler(container) -> None:
    """Attach a StreamlitLogHandler to the 'pipeline' logger."""
    logger = logging.getLogger("pipeline")
    # Remove any existing Streamlit handlers to avoid duplicates on re-run
    logger.handlers = [
        h for h in logger.handlers if not isinstance(h, StreamlitLogHandler)
    ]
    handler = StreamlitLogHandler(container)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def detach_streamlit_handler() -> None:
    """Detach any StreamlitLogHandler from the 'pipeline' logger."""
    logger = logging.getLogger("pipeline")
    logger.handlers = [
        h for h in logger.handlers if not isinstance(h, StreamlitLogHandler)
    ]


def setup_console_logging() -> None:
    """Configure the 'pipeline' logger to write to stdout (for CLI scripts)."""
    logger = logging.getLogger("pipeline")
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                               datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
