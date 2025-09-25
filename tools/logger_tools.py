import logging

def get_instance_logger(instance_name, log_file):
    logger = logging.getLogger(instance_name)
    logger.setLevel(logging.DEBUG)  # Set your desired logging level

    # Remove all handlers associated with the logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Ensure no StreamHandler is present (i.e., no logging to the console)
    logger.propagate = False

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)

    return logger
