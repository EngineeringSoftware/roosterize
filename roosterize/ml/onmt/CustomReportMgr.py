from typing import *

import pprint

from onmt.utils.report_manager import ReportMgrBase
from onmt.utils.statistics import Statistics

from seutil import LoggingUtils


class CustomReportMgr(ReportMgrBase):

    logger = LoggingUtils.get_logger(__name__)

    def __init__(self, report_every, start_time=-1.):
        super().__init__(report_every, start_time)

        self.training_history: List[dict] = list()
        self.step_history: List[dict] = list()
        return

    def _report_training(self,
            step: int,
            num_steps: int,
            learning_rate: float,
            report_stats: Statistics,
    ):
        new_history = {
            "step": step,
            "learning_rate": learning_rate,
            "accuracy": report_stats.accuracy(),
            "ppl": report_stats.ppl(),
            "xent": report_stats.xent(),
            "elapsed_time": report_stats.elapsed_time(),
        }
        self.logger.info(f"training reported: \n{pprint.pformat(new_history)}")
        self.training_history.append(new_history)
        return

    def _report_step(self,
            lr: float,
            step: int,
            train_stats: Optional[Statistics],
            valid_stats: Statistics,
    ):
        new_history = {
            "learning_rate": lr,
            "step": step,
            "accuracy": valid_stats.accuracy(),
            "ppl": valid_stats.ppl(),
            "xent": valid_stats.xent(),
            "elapsed_time": valid_stats.elapsed_time(),
        }
        self.logger.info(f"step reported: \n{pprint.pformat(new_history)}")
        self.step_history.append(new_history)
        return

    def get_joint_history(self):
        if len(self.training_history) != len(self.step_history):
            LoggingUtils.log_and_raise(self.logger, f"Cannot join two mismatch history!", Exception)
        # end if

        joint_history: List[dict] = list()
        for idx in range(len(self.training_history)):
            if self.training_history[idx]["step"] != self.step_history[idx]["step"]:
                LoggingUtils.log_and_raise(self.logger, f"Cannot join two mismatch history!", Exception)
            # end if
            joint_history.append({
                "step": self.training_history[idx]["step"],
                "elapsed_time": self.training_history[idx]["elapsed_time"],
                "learning_rate": self.training_history[idx]["learning_rate"],
                "train_accuracy": self.training_history[idx]["accuracy"],
                "train_ppl": self.training_history[idx]["ppl"],
                "train_xent": self.training_history[idx]["xent"],
                "val_accuracy": self.step_history[idx]["accuracy"],
                "val_ppl": self.step_history[idx]["ppl"],
                "val_xent": self.step_history[idx]["xent"],
            })
        # end for
        return joint_history
