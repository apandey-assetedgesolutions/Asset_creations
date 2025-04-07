import os 
import sys
import logging as logger 

def setup_logger(name, log_file, level=logger.INFO):
    logger = logger.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    file_handler = logger.FileHandler(log_file)
    file_handler.setLevel(level)

    formatter = logger.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger