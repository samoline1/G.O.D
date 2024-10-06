import logging
import sys

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

#    file_handler = logging.FileHandler('logging.log')
#    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
#    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
#    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

