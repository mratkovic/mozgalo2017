import os
from logging.config import fileConfig


def init_logging():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    logging_conf = os.path.join(dir_path, '..', 'logging_config.ini')
    if not os.path.exists(logging_conf):
        raise FileNotFoundError("Expected logging_config.ini in src dir")

    fileConfig(logging_conf)


def init_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
