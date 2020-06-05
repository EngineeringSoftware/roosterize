from typing import *

from copy import deepcopy
from onmt.models.model_saver import ModelSaver
import torch
import torch.nn as nn

from seutil import LoggingUtils


class MultiSourceModelSaver(ModelSaver):

    logger = LoggingUtils.get_logger(__name__)

    @classmethod
    def build_model_saver(cls, src_types, model_opt, opt, model, fields, optim):
        model_saver = cls(
            src_types,
            opt.save_model,
            model,
            model_opt,
            fields,
            optim,
            opt.keep_checkpoint)
        return model_saver

    def __init__(self, src_types, base_path, model, model_opt, fields, optim, keep_checkpoint=-1):
        super().__init__(base_path, model, model_opt, fields, optim, keep_checkpoint)
        self.src_types = src_types
        return

    def _save(self, step, model):
        real_model = (model.module
                      if isinstance(model, nn.DataParallel)
                      else model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()

        # NOTE: We need to trim the vocab to remove any unk tokens that
        # were not originally here.

        vocab = deepcopy(self.fields)
        for side in [f"src.{src_type}" for src_type in self.src_types] + ["tgt"]:
            keys_to_pop = []
            if hasattr(vocab[side], "fields"):
                unk_token = vocab[side].fields[0][1].vocab.itos[0]
                for key, value in vocab[side].fields[0][1].vocab.stoi.items():
                    if value == 0 and key != unk_token:
                        keys_to_pop.append(key)
                for key in keys_to_pop:
                    vocab[side].fields[0][1].vocab.stoi.pop(key, None)

        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': vocab,
            'opt': self.model_opt,
            'optim': self.optim.state_dict(),
        }

        self.logger.info("Saving checkpoint %s_step_%d.pt" % (self.base_path, step))
        checkpoint_path = '%s_step_%d.pt' % (self.base_path, step)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path
