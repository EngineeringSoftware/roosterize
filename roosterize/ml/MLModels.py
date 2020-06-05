from typing import *

from pathlib import Path
import recordclass
from recordclass import RecordClass

from seutil import LoggingUtils, IOUtils

from roosterize.data.ModelSpec import ModelSpec
from roosterize.ml.naming.OpenNMTInterfaceForNaming import OpenNMTInterfaceForNaming, ONMTILNConfig
from roosterize.ml.naming.OpenNMTMultiSourceForNaming import OpenNMTMultiSourceForNaming, ONMTMSLNConfig
from roosterize.ml.MLModelBase import MLModelBase
from roosterize.ml.MLModelsConsts import MLModelsConsts


class MLModels:
    logger = LoggingUtils.get_logger(__name__)

    class MLModelClz(RecordClass):
        clz: type = None
        config_clz: type = None

    NAMES_MODELS: Dict[str, MLModelClz] = {
        MLModelsConsts.M_LN_ONMTI: MLModelClz(clz=OpenNMTInterfaceForNaming, config_clz=ONMTILNConfig),
        MLModelsConsts.M_LN_ONMTMS: MLModelClz(clz=OpenNMTMultiSourceForNaming, config_clz=ONMTMSLNConfig),
    }

    @classmethod
    def get_model(cls,
            model_spec: ModelSpec,
            is_eval: bool = False,
    ) -> "MLModelBase":
        ml_model_clz: MLModels.MLModelClz = cls.NAMES_MODELS[model_spec.model]

        if not is_eval:
            try:
                model_spec.load_config()
            except (ValueError, FileNotFoundError):
                pass
            # end try
        # end if

        return ml_model_clz.clz(model_spec)

    @classmethod
    def generate_configs(cls, name: str, path: Path, **options):
        config_files: Set[str] = set()
        ml_model_clz = cls.NAMES_MODELS[name]
        config = ml_model_clz.config_clz()

        type_hints = get_type_hints(ml_model_clz.config_clz)

        model_path = path/name
        model_path.mkdir(parents=True, exist_ok=True)

        cls.logger.info(f"Possible attrs and default values: {config.__dict__}")

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
                    attrs_choices[k] = [IOUtils.dejsonfy(v, type_hints[k]) if v != "None" else None for v in str(options[k]).split()]
                else:
                    attrs_choices[k] = [type_hints[k](v) for v in str(options[k]).split()]
                # end if
                attrs_choices[k] = list(set(attrs_choices[k]))
                cls.logger.debug(f"attr {k}, choices: {attrs_choices[k]}")
                options.pop(k)
            # end if
        # end for

        if len(options) > 0:
            cls.logger.warning(f"These options are not recognized: {options.keys()}")
        # end if

        candidate = [0] * len(attrs_choices)
        is_explore_finished = False
        while True:
            # Generate current candidate
            for i, attr in enumerate(attrs):
                config.__setattr__(attr, attrs_choices[attr][candidate[i]])
            # end for
            if config.repOk():
                # Adjust batch size
                adjust_batch_size_func = getattr(config, "adjust_batch_size", None)
                if callable(adjust_batch_size_func):
                    adjust_batch_size_func()
                # end if

                config_file = model_path / (str(config)+".json")
                cls.logger.info(f"Saving candidate to {config_file}: {config}")
                config_files.add(name + "/" + str(config) + ".json")
                IOUtils.dump(config_file, IOUtils.jsonfy(config), IOUtils.Format.jsonPretty)
            else:
                cls.logger.info(f"Skipping invalid candidate: {config}")
            # end if

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
                    # end if
                else:
                    break
                # end if
            # end for
            if is_explore_finished:  break
        # end while

        for config_file in config_files:
            print(f"- model: {name}")
            print(f"  config-file: {config_file}")
            print()
        # end for

        return
