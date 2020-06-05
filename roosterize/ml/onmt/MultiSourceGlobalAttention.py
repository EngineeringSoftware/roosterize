from typing import *

from onmt.modules.global_attention import GlobalAttention
from onmt.modules.sparse_activations import sparsemax
from onmt.utils.misc import aeq, sequence_mask
import torch
import torch.nn as nn
import torch.nn.functional as F

from seutil import LoggingUtils


class MultiSourceGlobalAttention(GlobalAttention):

    logger = LoggingUtils.get_logger(__name__)

    def forward(self,
            source: torch.FloatTensor,  # [batch, tgt_len, dim]
            memory_bank_list: List[torch.FloatTensor],   # [num_srcs] x [batch, src_len, dim]
            memory_lengths_list: List[torch.FloatTensor] = None,  # [num_srcs] x [batch]
            coverage=None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        assert coverage is None

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False
        # end if

        # Join memory bank
        memory_bank = torch.cat(memory_bank_list, dim=1)

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, source_l_ = coverage.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = torch.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank)

        if memory_lengths_list is not None:
            mask = torch.cat([sequence_mask(memory_lengths, max_len=memory_bank_list[src_i].size(1)) for src_i, memory_lengths in enumerate(memory_lengths_list)], dim=1)
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))
        # end if

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch*target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, source], 2).view(batch*target_l, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)
        # end if

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, source_l_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            # Check output sizes
            target_l_, batch_, dim_ = attn_h.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            target_l_, batch_, source_l_ = align_vectors.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(source_l, source_l_)
        # end if

        return attn_h, align_vectors
