from typing import *

from copy import deepcopy
from roosterize.ml.onmt.CustomReportMgr import CustomReportMgr
from roosterize.ml.onmt.CustomTrainer import CustomTrainer
from roosterize.ml.onmt.MultiSourceCopyGenerator import MultiSourceCopyGeneratorLossCompute
import onmt
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.modules.sparse_activations import LogSparsemax
import onmt.utils
from onmt.utils.loss import LabelSmoothingLoss, NMTLossCompute
import torch
import torch.nn as nn
import traceback

from seutil import LoggingUtils


class MultiSourceTrainer(CustomTrainer):

    logger = LoggingUtils.get_logger(__name__)

    def __init__(self, src_types, model, train_loss, valid_loss, optim, trunc_size=0, shard_size=32, norm_method="sents", accum_count=[1], accum_steps=[0], n_gpu=1, gpu_rank=1, gpu_verbose_level=0, report_manager=None, model_saver=None, average_decay=0, average_every=1, model_dtype='fp32', earlystopper=None, dropout=[0.3], dropout_steps=[0]):
        super().__init__(model, train_loss, valid_loss, optim, trunc_size, shard_size, norm_method, accum_count, accum_steps, n_gpu, gpu_rank, gpu_verbose_level, report_manager, model_saver, average_decay, average_every, model_dtype, earlystopper, dropout, dropout_steps)

        self.src_types = src_types

        return

    @classmethod
    def build_loss_compute(cls, src_types, model, tgt_field, opt, train=True):
        """
        Returns a LossCompute subclass which wraps around an nn.Module subclass
        (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
        object allows this loss to be computed in shards and passes the relevant
        data to a Statistics object which handles training/validation logging.
        Currently, the NMTLossCompute class handles all loss computation except
        for when using a copy mechanism.
        """
        device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

        padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
        unk_idx = tgt_field.vocab.stoi[tgt_field.unk_token]

        if opt.lambda_coverage != 0:
            assert opt.coverage_attn, "--coverage_attn needs to be set in " \
                                      "order to use --lambda_coverage != 0"

        if opt.copy_attn:
            criterion = onmt.modules.CopyGeneratorLoss(
                len(tgt_field.vocab), opt.copy_attn_force,
                unk_index=unk_idx, ignore_index=padding_idx
            )
        elif opt.label_smoothing > 0 and train:
            criterion = LabelSmoothingLoss(
                opt.label_smoothing, len(tgt_field.vocab), ignore_index=padding_idx
            )
        elif isinstance(model.generator[-1], LogSparsemax):
            criterion = SparsemaxLoss(ignore_index=padding_idx, reduction='sum')
        else:
            criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

        # if the loss function operates on vectors of raw logits instead of
        # probabilities, only the first part of the generator needs to be
        # passed to the NMTLossCompute. At the moment, the only supported
        # loss function of this kind is the sparsemax loss.
        use_raw_logits = isinstance(criterion, SparsemaxLoss)
        loss_gen = model.generator[0] if use_raw_logits else model.generator
        if opt.copy_attn:
            compute = MultiSourceCopyGeneratorLossCompute(
                src_types, criterion, loss_gen, tgt_field.vocab, opt.copy_loss_by_seqlength,
                lambda_coverage=opt.lambda_coverage
            )
        else:
            compute = NMTLossCompute(
                criterion, loss_gen, lambda_coverage=opt.lambda_coverage)
        compute.to(device)

        return compute

    @classmethod
    def build_trainer(cls, src_types, opt, device_id, model, fields, optim, model_saver=None):
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
        train_loss = cls.build_loss_compute(src_types, model, tgt_field, opt)
        valid_loss = cls.build_loss_compute(src_types, model, tgt_field, opt, train=False)

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

        trainer = cls(src_types, model, train_loss, valid_loss, optim, trunc_size,
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

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            src_list = list()
            src_lengths_list = list()
            for src_type in self.src_types:
                batch_src = getattr(batch, f"src.{src_type}")
                src, src_lengths = batch_src if isinstance(batch_src, tuple) else (batch_src, None)
                if src_lengths is not None:
                    report_stats.n_src_words += src_lengths.sum().item()
                # end if
                src_list.append(src)
                src_lengths_list.append(src_lengths)
            # end for

            tgt_outer = batch.tgt

            bptt = False
            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()
                outputs, attns = self.model(src_list, tgt, src_lengths_list, bptt=bptt)
                bptt = True

                # 3. Compute loss.
                try:
                    loss, batch_stats = self.train_loss(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=j,
                        trunc_size=trunc_size)

                    if loss is not None:
                        self.optim.backward(loss)

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                except Exception:
                    traceback.print_exc()
                    self.logger.info("At step %d, we removed a batch - accum %d",
                                self.optim.training_step, k)

                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.model.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    self.optim.step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        if moving_average:
            valid_model = deepcopy(self.model)
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                param.data = avg.data
        else:
            valid_model = self.model

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src_list = list()
                src_lengths_list = list()
                for src_type in self.src_types:
                    batch_src = getattr(batch, f"src.{src_type}")
                    src, src_lengths = batch_src if isinstance(batch_src, tuple) else (batch_src, None)
                    src_list.append(src)
                    src_lengths_list.append(src_lengths)
                # end for
                tgt = batch.tgt

                # F-prop through the model.
                outputs, attns = valid_model(src_list, tgt, src_lengths_list)

                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)

                # Update statistics.
                stats.update(batch_stats)

        if moving_average:
            del valid_model
        else:
            # Set model back to training mode.
            valid_model.train()

        return stats

