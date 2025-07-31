import logging
import os
from datetime import datetime


def setup_logger(name="video_generation", level=logging.INFO):
    """
    Setup logger for the project.

    Args:
        name (str): Logger name
        level: Logging level

    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create logs directory if it doesn't exist
    # logs_dir = "logs"
    # if not os.path.exists(logs_dir):
    #     os.makedirs(logs_dir)

    # # Create file handler
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # file_handler = logging.FileHandler(
    # #     f"{logs_dir}/video_generation_{timestamp}.log", encoding="utf-8"
    #     f"{logs_dir}/video_generation.log", encoding="utf-8"
    # )
    # file_handler.setLevel(level)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    return logger


# Create global logger
logger = setup_logger()
