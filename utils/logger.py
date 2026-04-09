import logging
import sys

def setup_logger(name="SmartRoute", level=logging.DEBUG):
    """
    Sets up a custom logger to print intermediate steps for debugging
    and explainable output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate logs if instantiated multiple times
    if not logger.handlers:
        # Create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        
        # Format: [INFO] - Message
        formatter = logging.Formatter('\n[%(levelname)s] - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

logger = setup_logger()
