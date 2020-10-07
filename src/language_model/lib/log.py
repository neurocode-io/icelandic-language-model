import logging
from language_model.config import Settings


def get_logger(name):
    extra_fields = None
    logger = logging.getLogger(name)
    logger.setLevel(Settings().logging_level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger = logging.LoggerAdapter(logger, extra=extra_fields)

    return logger
