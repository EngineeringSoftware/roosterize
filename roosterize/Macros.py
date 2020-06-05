from typing import *

import os
from pathlib import Path


class Macros:
    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    project_dir: Path = this_dir.parent
    debug_dir: Path = project_dir / "debug"
    downloads_dir: Path = project_dir / "_downloads"

    DS_GROUP_T1 = "t1"
    DS_GROUP_T2 = "t2"
    DS_GROUP_T3 = "t3"
    DS_GROUP_LO = "lo"
    DS_GROUPS = [DS_GROUP_T1, DS_GROUP_T2, DS_GROUP_T3, DS_GROUP_LO]

    DS_GROUP_TA = "ta"  # = t1+t2+t3
    DS_GROUP_ALL = "allgroup"

    DS_TRAIN = "train"
    DS_VAL = "val"
    DS_TEST = "test"
    DS_TRAINEVALS = [DS_TRAIN, DS_VAL, DS_TEST]

    DS_TRAINEVAL_ALL = "all"

    DS_TRAIN_RATIO = 0.8
    DS_VAL_RATIO = 0.1
    DS_TEST_RATIO = 0.1
    DS_LO_TRAIN_RATIO = 0.4
    DS_LO_VAL_RATIO = 0.1
    DS_LO_TEST_RATIO = 0.5

    LEMMAS_DEPTH_CUTOFF = 0.75

    NUM_PICK_PRJS = 1
    NUM_PICK_FILES = 20
    NUM_PICK_EXAMPLES = 100
