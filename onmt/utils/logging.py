from seutil import LoggingUtils

logger = LoggingUtils.get_logger(__name__)

def init_logger(log_file=None, log_file_level=None):
    return logger