from typing import *

from roosterize.ml.onmt.MultiSourceGlobalAttention import MultiSourceGlobalAttention
from onmt.decoders.decoder import DecoderBase
from onmt.models.stacked_rnn import StackedLSTM, StackedGRU
from onmt.modules import context_gate_factory, GlobalAttention
from onmt.utils.misc import aeq
import torch
import torch.nn as nn

from seutil import LoggingUtils


class MultiSourceInputFeedRNNDecoder(DecoderBase):

    logger = LoggingUtils.get_logger(__name__)

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general", attn_func="softmax",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False, copy_attn_type="general",
                 num_srcs: int = 1,
    ):
        super().__init__(attentional=attn_type != "none" and attn_type is not None)

        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.num_srcs = num_srcs

        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Hidden state merging
        self.hidden_merge_0 = nn.Linear(
            in_features=self.hidden_size * self.num_srcs,
            out_features=self.hidden_size,
        )
        if rnn_type == "LSTM":
            self.hidden_merge_1 = nn.Linear(
                in_features=self.hidden_size * self.num_srcs,
                out_features=self.hidden_size,
            )
        # end if

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )
        # end if

        # Set up the standard attention.
        assert not coverage_attn, "Coverage attention is not supported"
        self._coverage: bool = coverage_attn
        if not self.attentional:
            if self._coverage:  raise ValueError("Cannot use coverage term with no attention.")
            self.ms_attn = None
        else:
            self.ms_attn = MultiSourceGlobalAttention(
                hidden_size, coverage=coverage_attn,
                attn_type=attn_type, attn_func=attn_func
            )
        # end if

        # Copy attention
        if copy_attn and not reuse_copy_attn:
            if copy_attn_type == "none" or copy_attn_type is None:  raise ValueError("Cannot use copy_attn with copy_attn_type none")
            self.copy_ms_attn = MultiSourceGlobalAttention(
                hidden_size, attn_type=copy_attn_type, attn_func=attn_func
            )
        else:
            self.copy_ms_attn = None
        # end if

        self._reuse_copy_attn = reuse_copy_attn and copy_attn
        if self._reuse_copy_attn and not self.attentional:  raise ValueError("Cannot reuse copy attention with no attention.")
        return

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.coverage_attn,
            opt.context_gate,
            opt.copy_attn,
            opt.dropout[0] if type(opt.dropout) is list
            else opt.dropout,
            embeddings,
            opt.reuse_copy_attn,
            opt.copy_attn_type,
            opt.num_srcs,
        )

    def init_state(self,
            src_list: List,
            memory_bank_list: List,
            encoder_final_list: List,  # [srcs] x [layers*directions, batch, dim]
    ) -> NoReturn:
        """Initialize decoder state with last state of the encoder."""
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        if isinstance(encoder_final_list[0], tuple):  # LSTM
            self.state["hidden"] = (
                self.hidden_merge_0(torch.cat([_fix_enc_hidden(encoder_final[0]) for encoder_final in encoder_final_list], dim=2)),
                self.hidden_merge_1(torch.cat([_fix_enc_hidden(encoder_final[1]) for encoder_final in encoder_final_list], dim=2))
            )
        else:  # GRU
            self.state["hidden"] = (
                self.hidden_merge_0(torch.cat(_fix_enc_hidden(encoder_final_list), dim=2)),
            )
        # end if

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None
        return

    def map_state(self, fn):
        self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])
        self.state["input_feed"] = fn(self.state["input_feed"], 1)
        if self._coverage and self.state["coverage"] is not None:
            self.state["coverage"] = fn(self.state["coverage"], 1)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self,
            tgt: torch.LongTensor,  # [tgt_len, batch, nfeats]
            memory_bank_list: List[torch.FloatTensor],  # [srcs] x [src_len, batch, hidden]
            memory_lengths_list: List[torch.LongTensor] = None,  # [srcs] x [batch]
            step=None
    ) -> Tuple[List[torch.FloatTensor], Dict[str, List[torch.FloatTensor]]]:
        # dec_outs: [tgt_len, batch, hidden]
        # attns: Dict[.., [tgt_len, batch, src_len]]
        dec_state, dec_outs, attns = self._run_forward_pass(tgt, memory_bank_list, memory_lengths_list=memory_lengths_list)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):  dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)
        # end if

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:  attns[k] = torch.stack(attns[k])
            # end for
        # end if
        return dec_outs, attns

    def _run_forward_pass(self,
            tgt: torch.LongTensor,  # [tgt_len, batch, nfeats]
            memory_bank_list: List[torch.FloatTensor],  # [srcs] x [src_len, batch, hidden]
            memory_lengths_list: List[torch.LongTensor] = None,  # [srcs] x [batch]
    ) -> Tuple[torch.Tensor, List[torch.FloatTensor], Dict[str, List[torch.FloatTensor]]]:
        # dec_state
        # dec_outs
        # attns

        batch_size = tgt.size()[1]

        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        dec_outs: List[torch.FloatTensor] = []
        attns: Dict[str, List[torch.FloatTensor]] = {}
        if self.ms_attn is not None:  attns["std"] = []
        if self.copy_ms_attn is not None or self._reuse_copy_attn:  attns["copy"] = []
        if self._coverage:  attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]
        coverage = self.state["coverage"].squeeze(0) if self.state["coverage"] is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for emb_t in emb.split(1):
            decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)  # [batch, tgt_len, dim]
            if self.attentional:
                decoder_output, align_vectors = self.ms_attn(
                    rnn_output,
                    [mb.transpose(0, 1) for mb in memory_bank_list],
                    memory_lengths_list,
                )  # [tgt_len, batch, dim], [tgt_len, batch, num_srcs*src_len]
                attns["std"].append(align_vectors)
            else:
                decoder_output = rnn_output
            # end if
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            dec_outs += [decoder_output]

            # Update the coverage attention.
            # PN: disabled coverage attention for now
            # if self._coverage:
            #     coverage = p_attn if coverage is None else p_attn + coverage
            #     attns["coverage"] += [coverage]

            if self.copy_ms_attn is not None:
                _, copy_attn = self.copy_ms_attn(
                    decoder_output,
                    [mb.transpose(0, 1) for mb in memory_bank_list],
                    memory_lengths_list,
                )
                attns["copy"] += [copy_attn]
            elif self._reuse_copy_attn:
                for enc_i in range(self.num_srcs):
                    attns["copy"] = attns["std"]
                # end for
            # end if

        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert rnn_type != "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        stacked_cell = StackedLSTM if rnn_type == "LSTM" else StackedGRU
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    @property
    def _input_size(self):
        """Using input feed by concatenating input with attention vectors."""
        return self.embeddings.embedding_size + self.hidden_size

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.embeddings.update_dropout(dropout)
