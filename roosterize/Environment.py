from typing import *

from seutil import LoggingUtils


class Environment:

    logger = LoggingUtils.get_logger(__name__)

    # =====
    # Random seed
    random_seed: int = None
