from typing import *

import onmt
from onmt.trainer import Trainer

from seutil import LoggingUtils

from roosterize.ml.onmt.CustomReportMgr import CustomReportMgr


class CustomTrainer(Trainer):

    logger = LoggingUtils.get_logger(__name__)

    @classmethod
    def build_trainer(cls, opt, device_id, model, fields, optim, model_saver=None):
        """
        Simplify `Trainer` creation based on user `opt`s*

        Args:
            opt (:obj:`Namespace`): user options (usually from argument parsing)
            model (:obj:`onmt.models.NMTModel`): the model to train
            fields (dict): dict of fields
            optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
            data_type (str): string describing the type of data
                e.g. "text", "img", "audio"
            model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
                used to save the model
        """

        tgt_field = dict(fields)["tgt"].base_field
        train_loss = onmt.utils.loss.build_loss_compute(model, tgt_field, opt)
        valid_loss = onmt.utils.loss.build_loss_compute(
            model, tgt_field, opt, train=False)

        trunc_size = opt.truncated_decoder  # Badly named...
        shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
        norm_method = opt.normalization
        accum_count = opt.accum_count
        accum_steps = opt.accum_steps
        n_gpu = opt.world_size
        average_decay = opt.average_decay
        average_every = opt.average_every
        dropout = opt.dropout
        dropout_steps = opt.dropout_steps
        if device_id >= 0:
            gpu_rank = opt.gpu_ranks[device_id]
        else:
            gpu_rank = 0
            n_gpu = 0
        gpu_verbose_level = opt.gpu_verbose_level

        earlystopper = onmt.utils.EarlyStopping(
            opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
            if opt.early_stopping > 0 else None

        # Customized report manager
        report_manager = CustomReportMgr(opt.report_every, start_time=-1)

        trainer = cls(model, train_loss, valid_loss, optim, trunc_size,
            shard_size, norm_method,
            accum_count, accum_steps,
            n_gpu, gpu_rank,
            gpu_verbose_level, report_manager,
            model_saver=model_saver if gpu_rank == 0 else None,
            average_decay=average_decay,
            average_every=average_every,
            model_dtype=opt.model_dtype,
            earlystopper=earlystopper,
            dropout=dropout,
            dropout_steps=dropout_steps)
        return trainer
