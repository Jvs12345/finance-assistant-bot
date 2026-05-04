"""
Logging configuration for the Document Search & Retrieval System.
Provides structured JSON logging with request ID tracking.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict
from contextvars import ContextVar

# Context variable for request ID tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: The log record to format

        Returns:
            JSON formatted log string
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up application logging configuration.

    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Set JSON formatter
    json_formatter = JSONFormatter()
    console_handler.setFormatter(json_formatter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)

    # Set log level for third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: The logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_request_id(request_id: str) -> None:
    """
    Set the request ID for the current context.

    Args:
        request_id: The request ID to set
    """
    request_id_var.set(request_id)


def get_request_id() -> str:
    """
    Get the request ID for the current context.

    Returns:
        The current request ID, or empty string if not set
    """
    return request_id_var.get()


def log_with_extra(
    logger: logging.Logger,
    level: int,
    message: str,
    **extra_fields: Any
) -> None:
    """
    Log a message with extra fields.

    Args:
        logger: The logger instance
        level: The log level (logging.DEBUG, logging.INFO, etc.)
        message: The log message
        **extra_fields: Additional fields to include in the log
    """
    record = logger.makeRecord(
        logger.name,
        level,
        "(unknown file)",
        0,
        message,
        (),
        None
    )
    record.extra_fields = extra_fields
    logger.handle(record)
