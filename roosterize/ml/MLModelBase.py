from typing import *

import abc
from pathlib import Path
import time

from seutil import LoggingUtils, IOUtils

from roosterize.data.ModelSpec import ModelSpec


TConfig = TypeVar("TConfig")


class MLModelBase(Generic[TConfig]):
    """
    The base class for machine learning models for predicting code conventions for Coq documents.
    :param TConfig: the configuration class.
    """

    logger = LoggingUtils.get_logger(__name__)

    def __init__(self,
            model_spec: ModelSpec,
            config_clz: type,
    ):
        self.spec = model_spec
        self.config: TConfig = IOUtils.dejsonfy(model_spec.config_dict, config_clz) if model_spec.config_dict is not None else config_clz()

        self.logger.info(f"{type(self).__name__} {self.spec.model} created with config {self.config}")
        return

    @property
    def logging_prefix(self):
        return f"Model {self.spec.model}/{str(self.config)}: "

    @abc.abstractmethod
    def process_data_impl(self,
            data_dir: Path,
            output_processed_data_dir: Path,
    ) -> NoReturn:
        """
        Processes the input to the intermediate data.
        :param data_dir: the directory containing the raw data
        :param output_processed_data_dir: the directory to save the processed data
        """
        pass

    def preprocess_with_train_data(self,
            data_dir: Path,
            output_processed_data_dir: Path,
    ) -> NoReturn:
        """
        Pre-processes the training set. For example, an implementation may use it to build the vocabulary.

        :param data_dir: the directory containing the raw train data
        :param output_processed_data_dir: the directory to save the processed data
        """
        pass

    def process_data(self,
            data_dir: Path,
            output_processed_data_dir: Path,
            is_train: bool = False,
    ) -> NoReturn:
        """
        Processes the data to the intermediate format.
        """
        self.logger.info(self.logging_prefix + f"Processing data from {data_dir} to {output_processed_data_dir}")
        IOUtils.rm_dir(output_processed_data_dir)
        IOUtils.mk_dir(output_processed_data_dir)

        if is_train:
            # Preprocess with training data, if needed
            self.preprocess_with_train_data(data_dir, output_processed_data_dir)
        # end if

        self.process_data_impl(data_dir, output_processed_data_dir)
        return

    @abc.abstractmethod
    def train_impl(self,
            train_processed_data_dir: Path,
            val_processed_data_dir: Path,
            output_model_dir: Path,
    ) -> NoReturn:
        """
        Trains the data on the input processed data.

        :param train_processed_data_dir: the directory containing the processed train data
        :param val_processed_data_dir: the directory containing the processed val data
        :param output_model_dir: the directory to save the output model
        """
        pass

    TRAINING_COMPLETED_FILE_NAME = "training-completed.txt"

    def train(self,
            train_processed_data_dir: Path,
            val_processed_data_dir: Path,
            output_model_dir: Path,
            force_retrain: bool = False,
    ) -> NoReturn:
        """
        Trains the model on the training data.

        The trained model should be saved to output_dir.
        This function auto-saves a training-completed.txt as a proof of completion of training at the end.

        :param train_processed_data_dir: the directory containing the processed train data
        :param val_processed_data_dir: the directory containing the processed val data
        :param output_model_dir: the directory to save the output model
        :param force_retrain: if set to True, re-train the model even if it was already trained (will remove previously trained model)
        """
        if force_retrain or not self.is_training_completed(output_model_dir):
            self.logger.info(self.logging_prefix + f"Training model at {output_model_dir}; train: {train_processed_data_dir}, val: {val_processed_data_dir}")
            IOUtils.rm_dir(output_model_dir)
            IOUtils.mk_dir(output_model_dir)

            # Save spec & configs of this model
            IOUtils.dump(output_model_dir/"config-dict.json", IOUtils.jsonfy(self.config), IOUtils.Format.jsonPretty)
            IOUtils.dump(output_model_dir/"spec.json", IOUtils.jsonfy(self.spec), IOUtils.Format.jsonPretty)
            self.train_impl(train_processed_data_dir, val_processed_data_dir, output_model_dir)
            IOUtils.dump(output_model_dir / self.TRAINING_COMPLETED_FILE_NAME, str(time.time_ns()), IOUtils.Format.txt)
        # end if
        return

    def is_training_completed(self, model_dir: Path) -> bool:
        """
        Checks if there already existed a completely-trained model in model_dir.
        """
        return (model_dir/self.TRAINING_COMPLETED_FILE_NAME).is_file()

    @abc.abstractmethod
    def eval(self,
            processed_data_dir: Path,
            model_dir: Path,
            output_result_dir: Path,
    ) -> NoReturn:
        """
        Evaluates the model on the specified provided data.

        :param processed_data_dir: the directory to processed training data
        :param model_dir: the directory containing the trained model
        :param output_result_dir: the directory to save output result files
        """
        pass

    @abc.abstractmethod
    def combine_eval_results_trials(self,
            result_dirs: List[Path],
            output_result_dir: Path,
    ) -> NoReturn:
        """
        Combines the evaluation results on different trials.
        """
        pass

    @abc.abstractmethod
    def error_analyze(self,
            data_dir: Path,
            processed_data_dir: Path,
            result_dir: Path,
            output_report_dir: Path,
    ) -> NoReturn:
        """
        Performs additional error analysis for the data's evaluation results on the specified dataset part.
        Requires: both #eval and #combine_eval_results_trials are called.

        This function is implemented differently for models of different applications. The evaluation metrics shall be dumped onto the current trial_rel_path.

        :param data_dir: the directory containing the raw data
        :param processed_data_dir: the directory containing the processed data
        :param result_dir: the directory containing the output result files from eval
        :param output_report_dir: the directory to save output error analyze reports
        """
        pass

    @abc.abstractmethod
    def get_best_trial(self,
            result_dirs: List[Path],
    ) -> Path:
        pass
