from typing import *

import copy
import importlib
import importlib.util
import json
import numpy as np
import os
from pathlib import Path
import sys
import time

from seutil import BashUtils, LoggingUtils


class Utils:
    """
    Some utilities that doesn't tie to a specific other file. TODO: move them into seutil at some point.
    """
    logger = LoggingUtils.get_logger(__name__)

    @classmethod
    def get_option_as_boolean(cls, options, opt, default=False) -> bool:
        if opt not in options:
            return default
        else:
            # Due to limitations of CliUtils...
            return str(options.get(opt, "false")).lower() != "false"
        # end if

    @classmethod
    def get_option_as_list(cls, options, opt, default=None) -> list:
        if opt not in options:
            return copy.deepcopy(default)
        else:
            l = options[opt]
            if not isinstance(l, list):  l = [l]
            return l
        # end if

    SUMMARIES_FUNCS: Dict[str, Callable[[Union[list, np.ndarray]], Union[int, float]]] = {
        "AVG": lambda l: np.mean(l) if len(l) > 0 else np.NaN,
        "SUM": lambda l: sum(l) if len(l) > 0 else np.NaN,
        "MAX": lambda l: max(l) if len(l) > 0 else np.NaN,
        "MIN": lambda l: min(l) if len(l) > 0 else np.NaN,
        "MEDIAN": lambda l: np.median(l) if len(l) > 0 and np.NaN not in l else np.NaN,
        "STDEV": lambda l: np.std(l) if len(l) > 0 else np.NaN,
    }

    SUMMARIES_PRESERVE_INT: Dict[str, bool] = {
        "AVG": False,
        "SUM": True,
        "MAX": True,
        "MIN": True,
        "MEDIAN": False,
        "STDEV": False,
    }

    @classmethod
    def tacc_get_num_jobs(cls) -> int:
        return int(BashUtils.run(f"squeue -u {os.getenv('USER')} | wc -l").stdout) - 1

    @classmethod
    def tacc_submit_jobs(cls, submit_script: Path, titles: List[str], scripts: List[Path], timeouts: List[str], output_dir: Path,
            submit_cd: int = 600, max_jobs: int = 4):
        job_i = 0
        while job_i < len(scripts):
            if cls.tacc_get_num_jobs() >= max_jobs:
                cls.logger.warning(f"Number of running jobs reach limit {max_jobs}, will retry after {submit_cd} seconds at {time.strftime('%a, %d %b %Y %H:%M:%S +0000', time.localtime(time.time()+submit_cd))}")
                time.sleep(submit_cd)
                continue
            # end if

            title = titles[job_i]
            script = scripts[job_i]
            timeout = timeouts[job_i]
            cls.logger.info(f"Submitting script {script}")

            try:
                BashUtils.run(f"{submit_script} \"{title}\" \"{output_dir}\" \"{script}\" \"{timeout}\"", expected_return_code=0)
            except KeyboardInterrupt:
                cls.logger.warning(f"Keyboard interrupt!")
                break
            except:
                cls.logger.warning(f"Failed to submit, will retry after {submit_cd} seconds at {time.strftime('%a, %d %b %Y %H:%M:%S +0000', time.localtime(time.time()+submit_cd))}")
                time.sleep(submit_cd)
                continue
            # end try

            # Submit successfully
            job_i += 1
        # end while
        return

    @classmethod
    def lod_to_dol(cls, list_of_dict: List[dict]) -> Dict[Any, List]:
        """
        Converts a list of dict to a dict of list.
        """
        keys = set.union(*[set(d.keys()) for d in list_of_dict])
        return {k: [d.get(k) for d in list_of_dict] for k in keys}

    @classmethod
    def counter_most_common_to_pretty_yaml(cls, most_common: List[Tuple[Any, int]]) -> str:
        s = "[\n"
        for x, c in most_common:
            s += f"[{json.dumps(x)}, {c}],\n"
        # end for
        s += "]\n"
        return s
