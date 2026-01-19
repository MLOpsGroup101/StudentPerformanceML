import sys
from datetime import datetime
from pathlib import Path
from loguru import logger


def setup_logger() -> None:
    """Configure loguru logger with file and terminal handlers."""
    
    # Remove all default Loguru handlers.
    logger.remove()

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Get the current date and time as a string and build the full path to the log file.
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"my_log_{date_str}.log"
    
    logger.add(
        log_file,
        level="DEBUG", # Log everything from DEBUG level and higher
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",# When the log file reaches 10 MB, start a new one
        retention="30 days",# Automatically delete log files older than 30 days
    )

    logger.add(
        sys.stderr, # Send logs to the terminal's error output
        level="WARNING", # Only show WARNING, ERROR, and CRITICAL messages in the terminal
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    
    
    