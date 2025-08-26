import sys
import logging
from pathlib import Path
from logging import Logger
from typing import Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


def setup_logger(
    name: Optional[str] = None,
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    error_level: int = logging.ERROR
) -> Logger:
    """
    Setup a logger with console, file, and error file handlers.
    """
    try:
        logger_name = name or __name__
        logger = logging.getLogger(logger_name)
        if logger.handlers:
            logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        main_log_file = log_path / "app.log"
        file_handler = RotatingFileHandler(
            filename=str(main_log_file),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        error_log_file = log_path / "errors.log"
        error_handler = TimedRotatingFileHandler(
            filename=str(error_log_file),
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8"
        )
        error_handler.setLevel(error_level)
        error_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)
        logger.propagate = False
        logger.info("Logger setup completed successfully.")
        return logger
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        fallback_logger = logging.getLogger(__name__)
        fallback_logger.error("Failed to setup logger: %s.", e)
        raise ValueError(f"Error setting up logger: {e}.") from e
    

logger = setup_logger(
    name="app",
)