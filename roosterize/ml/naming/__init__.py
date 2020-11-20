from roosterize.ml.naming.MultiSourceSeq2Seq import MultiSourceSeq2Seq, MultiSourceSeq2SeqConfig

ALL_MODELS = [
    MultiSourceSeq2Seq,
]

MODEL_CONFIGS = {
    MultiSourceSeq2Seq: MultiSourceSeq2SeqConfig,
}


def get_model_cls(name: str) -> type:
    """
    Finds the model class based on the given name.
    """
    for model in ALL_MODELS:
        if name == model.__name__:
            return model
    else:
        raise ValueError(f"No model with name {name}")


def get_model_config_cls(model_cls: type) -> type:
    return MODEL_CONFIGS[model_cls]
