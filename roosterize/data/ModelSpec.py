from typing import *

from recordclass import RecordClass

from seutil import IOUtils


class ModelSpec(RecordClass):
    name: str = ""
    model: str = ""
    config_file: str = None
    config_dict: dict = None

    def load_config(self) -> NoReturn:
        if self.config_file is not None:
            self.config_dict.update(IOUtils.load(self.config_file, IOUtils.Format.jsonPretty))
        else:
            raise ValueError("Config file not set!")
        # end if
        return

    def __hash__(self):
        return hash((self.name, self.model, self.config_file))

    @classmethod
    def build_from_dict(cls, d: dict) -> "ModelSpec":
        model_spec = ModelSpec(
            name=d.get("name", ""),
            model=d.get("model", "MultiSourceSeq2Seq"),
            config_file=d.get("config-file") if "config-file" in d else None,
            config_dict=dict(),
        )

        for k, v in d.items():
            if k.startswith("config-dict."):
                model_spec.config_dict[k.split(".", 1)[1]] = v
            # end if
        # end for

        return model_spec
