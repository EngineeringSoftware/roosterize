from typing import *

from onmt.decoders import DecoderBase
from onmt.encoders import EncoderBase
import torch
import torch.nn as nn

from seutil import LoggingUtils


class MultiSourceNMTModel(nn.Module):

    logger = LoggingUtils.get_logger(__name__)

    def __init__(self,
            encoders: List[EncoderBase],
            decoder: DecoderBase,
    ):
        super().__init__()
        self.encoders = encoders
        for enc_i, encoder in enumerate(self.encoders):  self.add_module(f"encoder-{enc_i}", encoder)
        self.decoder = decoder
        return

    def forward(self,
            src_list: List[torch.Tensor],
            tgt: torch.LongTensor,
            lengths_list: List[torch.LongTensor],
            bptt: bool = False,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state_list: List = list()
        memory_bank_list: List = list()
        for enc_i, encoder in enumerate(self.encoders):
            enc_state, memory_bank, lengths = encoder(src_list[enc_i], lengths_list[enc_i])
            enc_state_list.append(enc_state)
            memory_bank_list.append(memory_bank)
            lengths_list[enc_i] = lengths
        # end for

        if bptt is False:
            self.decoder.init_state(src_list, memory_bank_list, enc_state_list)
        # end if
        dec_out, attns = self.decoder(tgt, memory_bank_list, memory_lengths_list=lengths_list)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
