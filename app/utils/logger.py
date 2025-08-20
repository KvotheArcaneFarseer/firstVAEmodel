# app/utils/logger.py
# Description: This module provides a centralized, professional logging setup,
# enabling structured and context-aware logging throughout the application.

import logging
import sys
import threading
from typing import Dict, Any, Optional

# --- Context Filter for Request Tracing ---
# This filter injects contextual information (like a request_id) into log records.
class ContextFilter(logging.Filter):
    """
    A logging filter that adds contextual information to log records.
    It uses thread-local storage to ensure context is specific to each request.
    """
    def __init__(self):
        super().__init__()
        # threading.local() creates a storage object that is unique to each thread.
        # This is how we keep the context of different requests separate.
        self._local = threading.local()

    def set_context(self, **kwargs: Any):
        """Sets context data for the current thread."""
        self._local.context = kwargs

    def clear_context(self):
        """Clears context data for the current thread."""
        self._local.context = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """
        This method is automatically called for every log message.
        It adds the stored context to the log record before it's formatted.
        """
        if hasattr(self._local, 'context'):
            for key, value in self._local.context.items():
                setattr(record, key, value)
        return True

# --- Central Logger Manager ---
# This class manages the entire logging setup as a singleton.
class LoggerManager:
    """
    Manages the application's logging configuration to ensure consistency.
    Implemented as a singleton to guarantee a single configuration state.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.context_filter = ContextFilter()
        self.setup_logging()

    def setup_logging(self):
        """Configures the root logger for the application."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Create a handler to send logs to the console.
        stream_handler = logging.StreamHandler(sys.stdout)
        
        # Define a format that can include our custom context fields.
        # Note: If a context field (like 'request_id') is not present, it will be ignored.
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
        )
        stream_handler.setFormatter(formatter)
        
        # Add our context filter to the handler.
        stream_handler.addFilter(self.context_filter)

        if not root_logger.handlers:
            root_logger.addHandler(stream_handler)

# --- Global Instance and Helper Functions ---
# These are the simple functions the rest of our application will use.

_logger_manager = LoggerManager()

def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance for a specific module."""
    return logging.getLogger(name)

def set_log_context(**kwargs: Any):
    """A global helper to set the log context for the current request."""
    _logger_manager.context_filter.set_context(**kwargs)

def clear_log_context():
    """A global helper to clear the log context after a request is done."""
    _logger_manager.context_filter.clear_context()
