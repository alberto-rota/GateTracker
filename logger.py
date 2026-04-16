import logging
import sys
import os
import re
from datetime import datetime
from enum import Enum, auto
from rich.console import Console
from rich.theme import Theme
from rich.logging import RichHandler
from utilities.formatting import align, strip_rich_markup
from typing import Optional, Any, Union,Dict


class LogContext(Enum):
    """
    Enumeration of logging contexts for consistent categorization of log messages.
    
    This enum defines different contexts that can be used to categorize and style
    log messages. Each context corresponds to a different aspect of the system
    and can have its own visual styling in the console output.
    """
    IMPORT = auto()
    DATASET = auto()
    GCLOUD = auto()
    SAVE = auto()
    OPTIMIZATION = auto()
    ENGINE = auto()
    TRAINING = auto()
    VALIDATION = auto()
    TEST = auto()
    WANDB = auto()
    WARNING = auto()
    ERROR = auto()
    INFO = auto()
    DEBUG = auto()


# Define a rich theme for consistent styling
CUSTOM_THEME = Theme(
    {
        "import": "cyan",
        "gcloud": "blue",
        "save": "magenta",
        "dataset": "green",
        "optimization": "magenta",
        "engine": "yellow",
        "warning": "yellow",
        "training": "orange1",
        "validation": "green",
        "test": "cyan",
        "wandb": "bright_yellow",
        "error": "bold red",
        "info": "white",
        "debug": "dim cyan",
    }
)


class CustomFormatter(logging.Formatter):
    """
    Custom formatter that strips Rich markup from log messages.
    
    This formatter ensures that log messages written to files don't contain
    Rich markup characters, while preserving the formatted output for console
    display. It processes the message before standard formatting is applied.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record, stripping Rich markup for file output.
        
        Args:
            record: The log record to format
            
        Returns:
            Formatted log message with Rich markup stripped
        """
        # Strip Rich markup from the message before standard formatting
        # The record.msg is the raw message, before any formatting
        record.msg = strip_rich_markup(record.msg)

        # Let the parent formatter create record.message and do standard formatting
        result = super().format(record)

        # At this point, we could log the results, but it's not needed in production
        # print(f"Original Message: {record.msg}")
        # print(f"Formatted message: {record.message}")
        # print(f"Formatted record: {record}")

        return result


class CustomLogger:
    """
    Custom logger class that handles both console and file logging with rich formatting.
    
    This logger provides a unified interface for logging with rich console output
    and clean file output. It supports context-based styling, multiple log levels,
    and automatic timestamp formatting. The logger can be configured to write to
    both console and file simultaneously with different formatting for each.
    """

    def __init__(self, name: str, log_file: Optional[str] = None) -> None:
        """
        Initialize the logger with the given name and optional log file.

        Args:
            name: Name of the logger (typically module name)
            log_file: Path to the log file. If None, no file logging is performed.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.name = name
        self.default_context = LogContext.INFO  # Default context

        # Rich console for terminal output
        self.console = Console(theme=CUSTOM_THEME)

        # Clear any existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Set up rich console handler with a custom format that doesn't show the log level
        class CustomRichHandler(RichHandler):
            def emit(self, record: logging.LogRecord) -> None:
                """Emit a log record with custom formatting."""
                # Skip the usual formatting and just use our custom message
                record.message = record.getMessage()
                self.console.print(record.message)

        # Create our custom handler
        rich_handler = CustomRichHandler(
            console=self.console,
            rich_tracebacks=True,
            show_time=False,  # We'll handle time ourselves
            show_path=False,
            show_level=False,  # Don't show the log level
            markup=True,
        )
        rich_handler.setLevel(logging.INFO)
        self.logger.addHandler(rich_handler)

        # Set up file handler if log_file is provided
        if log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)

            # Use our custom formatter that strips Rich markup
            file_formatter = CustomFormatter(
                "%(asctime)s - %(name)s - %(context)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def set_context(self, context: Union[LogContext, str]) -> 'CustomLogger':
        """
        Set the default context for all subsequent log messages.

        Args:
            context: LogContext enum value or string to use as default

        Returns:
            Self for method chaining
        """
        self.default_context = context
        return self  # Allow method chaining

    def _log(self, level: int, context: Optional[Union[LogContext, str]] = None, *message_args: Any, style: Optional[str] = None, end: str = "\n", **kwargs: Any) -> None:
        """
        Internal method to handle logging with context and rich formatting.

        Args:
            level: Logging level (e.g., logging.INFO)
            context: LogContext enum value or string (uses default if None)
            *message_args: Multiple message arguments (like print function)
            style: Rich style string to override default context style
            end: String appended after the last message argument (default: "\n")
            **kwargs: Additional arguments for the logger
        """
        # Use the provided context or fall back to the default
        context = context if context is not None else self.default_context

        # Convert all message arguments to strings and join with spaces (like print)
        message = " ".join(str(arg) for arg in message_args)

        time_str = datetime.now().strftime("%H:%M:%S")

        # Handle both enum and string contexts
        if isinstance(context, LogContext):
            context_name = context.name
            default_style = context_name.lower()
        else:
            # String context
            context_name = str(context)
            # Convert to lowercase for style lookup and handle special characters
            default_style = context_name.lower().replace(" ", "_").replace("-", "_")

        # Use provided style or default to context name
        context_style = style if style else default_style

        # Format for rich console output - context first, then time
        rich_message = f"[{context_style}]{align(context_name,8,'left')}[/{context_style}] [{time_str}] {message}"

        # Append the end string (like print's end parameter)
        if end != "\n":
            rich_message += end

        # Pass context as an extra parameter for file logging
        extra = kwargs.get("extra", {})
        extra["context"] = context_name
        kwargs["extra"] = extra

        # Log with the specified level
        self.logger.log(level, rich_message, **kwargs)

    def info(self, *message_args: Any, context: Optional[Union[LogContext, str]] = None, style: Optional[str] = None, end: str = "\n", **kwargs: Any) -> None:
        """
        Log an info message with the specified context and optional style.
        Works like print() function, accepting multiple arguments.

        Args:
            *message_args: Multiple message parts to be joined with spaces
            context: LogContext enum value or string context (uses default if None)
            style: Rich style string to override default context style
            end: String appended after the last message argument (default: "\n")
            **kwargs: Additional arguments for the logger
        """
        self._log(logging.INFO, context, *message_args, style=style, end=end, **kwargs)

    def warning(self, *message_args: Any, context: Optional[Union[LogContext, str]] = None, style: Optional[str] = None, end: str = "\n", **kwargs: Any) -> None:
        """
        Log a warning message with the specified context and optional style.
        Works like print() function, accepting multiple arguments.

        Args:
            *message_args: Multiple message parts to be joined with spaces
            context: LogContext enum value or string context (uses default if None)
            style: Rich style string to override default context style
            end: String appended after the last message argument (default: "\n")
            **kwargs: Additional arguments for the logger
        """
        self._log(
            logging.WARNING, context, *message_args, style=style, end=end, **kwargs
        )

    def error(self, *message_args: Any, context: Optional[Union[LogContext, str]] = None, style: Optional[str] = None, end: str = "\n", **kwargs: Any) -> None:
        """
        Log an error message with the specified context and optional style.
        Works like print() function, accepting multiple arguments.

        Args:
            *message_args: Multiple message parts to be joined with spaces
            context: LogContext enum value or string context (uses default if None)
            style: Rich style string to override default context style
            end: String appended after the last message argument (default: "\n")
            **kwargs: Additional arguments for the logger
        """
        self._log(logging.ERROR, context, *message_args, style=style, end=end, **kwargs)

    def debug(self, *message_args: Any, context: Optional[Union[LogContext, str]] = None, style: Optional[str] = None, end: str = "\n", **kwargs: Any) -> None:
        """
        Log a debug message with the specified context and optional style.
        Works like print() function, accepting multiple arguments.

        Args:
            *message_args: Multiple message parts to be joined with spaces
            context: LogContext enum value or string context (uses default if None)
            style: Rich style string to override default context style
            end: String appended after the last message argument (default: "\n")
            **kwargs: Additional arguments for the logger
        """
        self._log(logging.DEBUG, context, *message_args, style=style, end=end, **kwargs)

    def print(self, *args: Any, style: Optional[str] = None, **kwargs: Any) -> None:
        """
        Direct access to rich console's print functionality for advanced formatting.

        Args:
            *args: Arguments to print
            style: Style to apply to the printed content
            **kwargs: Additional kwargs for rich Console.print()
        """
        time_str = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{time_str}] "

        # Prepend the time to the first argument if it's a string
        if args and isinstance(args[0], str):
            args = (prefix + args[0],) + args[1:]
        else:
            self.console.print(prefix, end="")

        self.console.print(*args, style=style, **kwargs)


# Module-level loggers cache
_loggers: Dict[str, CustomLogger] = {}


def get_logger(module_name: str, log_to_file: bool = False, log_dir: Optional[str] = None) -> CustomLogger:
    """
    Get or create a logger for the specified module.

    This function implements a singleton pattern for loggers, ensuring that
    each module gets a unique logger instance that can be configured for
    both console and file output. The logger is cached to avoid creating
    multiple instances for the same module.

    Args:
        module_name: Name of the module
        log_to_file: Whether to log to a file in addition to console
        log_dir: Directory for log files if log_to_file is True

    Returns:
        The logger instance for the specified module
    """
    if module_name in _loggers:
        return _loggers[module_name]

    log_file = None
    if log_to_file:
        # logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        log_file = os.path.join(log_dir, f"{module_name.split('.')[-1]}.log")

    logger = CustomLogger(module_name, log_file)
    _loggers[module_name] = logger
    return logger
