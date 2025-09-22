import logging
import os

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, "project.log")

def setup_logger():
    """
    Configures the logger to output to both the console and a log file.
    """
    logger = logging.getLogger("breast_cancer_classification")
    logger.setLevel(logging.DEBUG)  

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
  
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  
    file_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger() 
