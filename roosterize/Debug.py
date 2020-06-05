from typing import *

import collections


class Debug:
    """
    For holding some debugging variables.
    """
    is_debug = False

    global_print_counter = 0
    print_counter: Counter[str] = collections.Counter()
    seen_shapes: Set[str] = set()
