import random
import sys
import time
from pathlib import Path

from seutil import CliUtils, IOUtils, LoggingUtils

from roosterize.Environment import Environment
from roosterize.Macros import Macros
from roosterize.Utils import Utils

logging_file = Macros.this_dir.parent / "experiment.log"
LoggingUtils.setup(filename=str(logging_file))

logger = LoggingUtils.get_logger(__name__)


# ==========
# Data Collection

def extract_data(**options):
    from roosterize.data.DataMiner import DataMiner

    project_path = Path(options["project"]).absolute()
    files = Utils.get_option_as_list(options, "files", None)
    exclude_files = Utils.get_option_as_list(options, "exclude-files", None)
    exclude_pattern = options.get("exclude-pattern", None)
    serapi_options = options.get("serapi-options", "")
    output_path = Path(options["output"]).absolute()


    DataMiner.extract_data_project(project_path, files, exclude_files, exclude_pattern, serapi_options, output_path)
    return


def extract_data_from_corpus(**options):
    from roosterize.data.DataMiner import DataMiner

    corpus_path = Path(options["corpus"]).absolute()
    trainevals = Utils.get_option_as_list(options, "trainevals", Macros.DS_TRAINEVALS)
    groups = Utils.get_option_as_list(options, "groups", [Macros.DS_GROUP_T1, Macros.DS_GROUP_TA])
    output_path = Path(options["output"]).absolute()

    DataMiner.extract_data_from_corpus(corpus_path, trainevals, groups, output_path)
    return


def collect_data(**options):
    from roosterize.data.DataMiner import DataMiner
    DataMiner.collect_data(**options)
    return


# ==========
# Training / Testing

def train_model(**options):
    from roosterize.data.ModelSpec import ModelSpec
    from roosterize.ml.MLModels import MLModels

    train_data_dir = Path(options["train"]).absolute()
    val_data_dir = Path(options["val"]).absolute()
    model_dir = Path(options["model-dir"]).absolute()
    output_dir = Path(options["output"]).absolute()

    force_retrain = Utils.get_option_as_boolean(options, "force-retrain")

    model_spec = ModelSpec.build_from_dict(options)

    # Get the ML model
    model = MLModels.get_model(model_dir, model_spec)

    # Process data
    model.process_data(train_data_dir, output_dir/"train-processed-data", is_train=True)
    model.process_data(val_data_dir, output_dir/"val-processed-data")

    # Train & eval the ML model on val set
    model.train(output_dir/"train-processed-data", output_dir/"val-processed-data", force_retrain=force_retrain)
    model.eval(output_dir/"val-processed-data", output_dir/"val-eval-result")
    return


def eval_model(**options):
    from roosterize.data.ModelSpec import ModelSpec
    from roosterize.ml.MLModels import MLModels

    data_dir = Path(options["data"]).absolute()
    model_dir = Path(options["model-dir"]).absolute()
    output_dir = Path(options["output"]).absolute()

    # Get the ML model
    model_spec = IOUtils.dejsonfy(IOUtils.load(model_dir/"spec.json", IOUtils.Format.json), ModelSpec)
    model = MLModels.get_model(model_dir, model_spec, is_eval=True)

    # Process data
    model.process_data(data_dir, output_dir/"eval-processed-data")

    # Eval
    model.eval(output_dir/"eval-processed-data", output_dir/"eval-result")
    return


def suggest_lemmas(**options):
    from roosterize.data.DataMiner import DataMiner
    from roosterize.data.ModelSpec import ModelSpec
    from roosterize.ml.MLModels import MLModels

    project_path = Path(options["project"]).absolute()
    files = Utils.get_option_as_list(options, "files", None)
    exclude_files = Utils.get_option_as_list(options, "exclude-files", None)
    exclude_pattern = options.get("exclude-pattern", None)
    serapi_options = options.get("serapi-options", "")
    output_dir = Path(options["output"]).absolute()
    model_dir = Path(options["model-dir"]).absolute()

    # Extract data
    print(">>>>> Extracting lemmas ...")
    DataMiner.extract_data_project(project_path, files, exclude_files, exclude_pattern, serapi_options, output_dir/"raw-data")

    # Get the ML model
    print(">>>>> Initializing model ...")
    model_spec = IOUtils.dejsonfy(IOUtils.load(model_dir/"spec.json", IOUtils.Format.json), ModelSpec)
    model = MLModels.get_model(model_dir, model_spec, is_eval=True)

    # Process data
    print(">>>>> Processing data ...")
    model.process_data(output_dir/"raw-data", output_dir/"eval-processed-data")

    # Eval
    print(">>>>> Applying model ...")
    model.eval(output_dir/"eval-processed-data", output_dir/"eval-result")

    # Print suggestions
    print(">>>>> Suggestions:")
    print(IOUtils.load(output_dir/"eval-result"/"suggestions.txt", IOUtils.Format.txt))
    return


# User interfaces

def download_global_model(**options):
    from roosterize.interface.CommandLineInterface import CommandLineInterface
    ui = CommandLineInterface()
    force_yes = Utils.get_option_as_boolean(options, "y")
    ui.download_global_model(force_yes)


def suggest_naming(**options):
    from roosterize.interface.CommandLineInterface import CommandLineInterface
    file_path = Path(options["file"])
    prj_root = options.get("project_root", None)
    if prj_root is not None:
        prj_root = Path(prj_root)
    ui = CommandLineInterface()
    ui.suggest_naming(file_path, prj_root)


def improve_project_model(**options):
    # TODO: future work
    raise NotImplementedError("Improve_project_model feature will be enabled in the future.")
    from roosterize.interface.CommandLineInterface import CommandLineInterface
    prj_root = options.get("project_root", None)
    if prj_root is not None:
        prj_root = Path(prj_root)
    ui = CommandLineInterface()
    ui.improve_project_model(prj_root)


def vscode_server(**options):
    from roosterize.interface.VSCodeServer import start_server
    start_server(**options)


def help(**options):
    print("""Usage: python -m roosterize.main ACTION OPTIONS
Below is a list of supported actions and options.

python -m roosterize.main suggest_lemmas
  Suggests lemma names for a Coq project.
  Options:
    --project         The path to the project
    --model-dir       The path to the trained model
    --output          The path to put outputs
    --serapi-options  (Optional) The SerAPI command line options for setting library paths
    --files           (Optional) Only process specified files if provided; repeat the option to provide multiple files
    --exclude-files   (Optional) Skip specified files if provided; repeat the option to provide multiple files
    --exclude-pattern (Optional) Skip files matching the specified regular expression pattern


python -m roosterize.main extract_data
  Extracts raw data (used for train/eval the model) from a Coq project.
  Options:
    --project         The path to the project
    --output          The path to put output raw data
    --serapi-options  (Optional) The SerAPI command line options for setting library paths
    --files           (Optional) Only process specified files if provided; repeat the option to provide multiple files
    --exclude-files   (Optional) Skip specified files if provided; repeat the option to provide multiple files
    --exclude-pattern (Optional) Skip files matching the specified regular expression pattern


python -m roosterize.main train_model
  Trains a lemma name suggestion neural network model.
  Options:
    --train           The path to the training set raw data
    --val             The path to the validation set raw data
    --model-dir       The path to put the trained model
    --output          The path to put outputs during training/validation
    --model           The name of model to train (LN-ONMTMS|LN-ONMTI)
    --config-file     (Optional) The path to the config file (see ./configs)
    --force-retrain   (Optional) If provided, re-training the model even if it exists


python -m roosterize.main eval_model
  Evaluates a lemma name suggestion model on extracted data.
  Options:
    --data            The path to the extracted raw data
    --model-dir       The path to the trained model
    --output          The path to put outputs during evaluation


python -m roosterize.main help
  Prints this manual page.
""")


# ==========
# Debugging

def check_gpu(**options):
    import torch

    is_gpu_available = torch.cuda.is_available()
    print(f"GPU available: {is_gpu_available}")

    if is_gpu_available:
        gpu_count = torch.cuda.device_count()
        print(f"GPU count: {gpu_count}")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_cap = torch.cuda.get_device_capability(i)
            gpu_prop = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: name: {gpu_name}; capacity: {gpu_cap}; properties: {gpu_prop}")
        # end for
    # end if
    return


# ==========
# Main

def normalize_options(opts: dict) -> dict:
    if "log_path" in opts:
        logger.info(f"Switching to log file {opts['log_path']}")
        LoggingUtils.setup(filename=opts['log_path'])
    # end if

    if "random_seed" in opts:
        seed = opts["random_seed"]
    else:
        seed = time.time_ns()
    # end if
    logger.info(f"Random seed is {seed}")
    random.seed(seed)
    Environment.random_seed = seed

    if "debug" in opts:
        from roosterize.Debug import Debug
        Debug.is_debug = True
        logger.debug(f"options: {opts}")
    # end if
    return opts

if __name__ == "__main__":
    CliUtils.main(sys.argv[1:], globals(), normalize_options)
