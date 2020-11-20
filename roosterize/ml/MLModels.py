from pathlib import Path
from typing import get_type_hints, Set

import recordclass
from seutil import IOUtils, LoggingUtils

from roosterize.data.ModelSpec import ModelSpec
from roosterize.ml.MLModelBase import MLModelBase
from roosterize.ml.naming import get_model_cls, get_model_config_cls

logger = LoggingUtils.get_logger(__name__)


class MLModels:

    @classmethod
    def get_model(
            cls,
            model_dir: Path,
            model_spec: ModelSpec,
            is_eval: bool = False,
    ) -> "MLModelBase":
        model_cls = get_model_cls(model_spec.model)

        if not is_eval:
            try:
                model_spec.load_config()
            except (ValueError, FileNotFoundError):
                pass

        return model_cls(model_dir, model_spec)

    @classmethod
    def generate_configs(cls, name: str, path: Path, **options):
        config_files: Set[str] = set()
        model_cls = get_model_cls(name)
        model_config_cls = get_model_config_cls(model_cls)
        config = model_config_cls()

        type_hints = get_type_hints(model_config_cls)

        model_path = path / name
        model_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Possible attrs and default values: {config.__dict__}")

        attrs_choices: dict = dict()
        attrs: list = list()

        for k, default_v in config.__dict__.items():
            attrs.append(k)
            if k not in options:
                attrs_choices[k] = [default_v]
            else:
                if type_hints[k] == bool:
                    attrs_choices[k] = [v == "True" for v in str(options[k]).split()]
                elif issubclass(type_hints[k], recordclass.mutabletuple):
                    attrs_choices[k] = [IOUtils.dejsonfy(v, type_hints[k]) if v != "None" else None for v in
                                        str(options[k]).split()]
                else:
                    attrs_choices[k] = [type_hints[k](v) for v in str(options[k]).split()]
                attrs_choices[k] = list(set(attrs_choices[k]))
                logger.debug(f"attr {k}, choices: {attrs_choices[k]}")
                options.pop(k)

        if len(options) > 0:
            logger.warning(f"These options are not recognized: {options.keys()}")

        candidate = [0] * len(attrs_choices)
        is_explore_finished = False
        while True:
            # Generate current candidate
            for i, attr in enumerate(attrs):
                config.__setattr__(attr, attrs_choices[attr][candidate[i]])
            if config.repOk():
                # Adjust batch size
                adjust_batch_size_func = getattr(config, "adjust_batch_size", None)
                if callable(adjust_batch_size_func):
                    adjust_batch_size_func()

                config_file = model_path / (str(config) + ".json")
                logger.info(f"Saving candidate to {config_file}: {config}")
                config_files.add(name + "/" + str(config) + ".json")
                IOUtils.dump(config_file, IOUtils.jsonfy(config), IOUtils.Format.jsonPretty)
            else:
                logger.info(f"Skipping invalid candidate: {config}")

            # To next candidate
            for i, attr in enumerate(attrs):
                candidate[i] += 1
                if candidate[i] >= len(attrs_choices[attr]):
                    candidate[i] = 0
                    if i == len(attrs) - 1:
                        is_explore_finished = True
                        break
                    else:
                        continue
                else:
                    break
            if is_explore_finished:
                break

        for config_file in config_files:
            print(f"- model: {name}")
            print(f"  config-file: {config_file}")
            print()

        return
