from typing import *

import random
import torch
import torch.nn
import torch.nn.utils

__all__ = [
    "Encoder",
    "DecoderPlain", "DecoderWithAttention",
    "Seq2SeqPlain", "Seq2SeqWithAttention",
    "Attention",
]


# https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
class Encoder(torch.nn.Module):

    def get_dim_hidden_with_direction(self):  return self.dim_hidden * (2 if self.is_bidirectional else 1)

    def __init__(self, device,
            dim_embed: int,
            dim_hidden: int,
            rnn_num_layers: int,
            is_bidirectional: bool,
            dropout: float,
            vocab_input_size: int,
    ):
        super().__init__()
        self.device = device
        self.dim_embed: int = dim_embed
        self.dim_hidden: int = dim_hidden
        self.dropout: float = dropout
        self.vocab_input_size: int = vocab_input_size
        self.rnn_num_layers: int = rnn_num_layers
        self.is_bidirectional: bool = is_bidirectional

        ### Graph
        # Embed
        self.layer_embed = torch.nn.Embedding(
            num_embeddings=self.vocab_input_size,
            embedding_dim=self.dim_embed,
            padding_idx=0,
        )

        # RNN
        self.layer_rnn = torch.nn.GRU(
            input_size=self.dim_embed,
            hidden_size=self.dim_hidden,
            dropout=self.dropout,
            num_layers=self.rnn_num_layers,
            bidirectional=self.is_bidirectional,
        )

        # Dropout
        self.layer_dropout = torch.nn.Dropout(self.dropout)

        # Output fully connected layers, for merging bidirectional hidden states
        if self.is_bidirectional:
            self.layers_fc = [torch.nn.Linear(
                in_features=self.get_dim_hidden_with_direction(),
                out_features=self.dim_hidden,
            ) for _ in range(self.rnn_num_layers)]
            for i, layer_fc in enumerate(self.layers_fc):  self.add_module(f"layer_fc_{i}", layer_fc)
        # end if

        return

    def forward(self,
            inputs: torch.Tensor,  # [seq, batch]
            seq_lens,  # [batch]
    ):  # [seq, batch, dim_hidden*num_direction], [num_rnn_layer, batch, dim_hidden]
        embedded = self.layer_dropout(self.layer_embed(inputs))  # [seq, batch, dim_embed]
        embedded_packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, seq_lens, enforce_sorted=False)  # PackedSequence

        outputs, hidden = self.layer_rnn(embedded_packed)  # PackedSequence, [num_rnn, batch, dim_hidden] (a pair of this for LSTM)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # [seq, batch, dim_hidden*num_direction]

        if self.is_bidirectional:
            hidden = torch.stack([
                torch.tanh(self.layers_fc[i](
                    torch.cat((hidden[2*i,:,:], hidden[2*i+1,:,:]), dim=1)
                )) for i in range(self.rnn_num_layers)
            ], dim=0)
        # end if

        return outputs, hidden


# https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
class DecoderPlain(torch.nn.Module):

    def __init__(self, device,
            dim_embed: int,
            dim_hidden: int,
            rnn_num_layers: int,
            dropout: float,
            vocab_target_size: int,
    ):
        super().__init__()
        self.device = device
        self.vocab_target_size: int = vocab_target_size
        self.dim_embed: int = dim_embed
        self.dim_hidden: int = dim_hidden
        self.dropout: float = dropout
        self.rnn_num_layers: int = rnn_num_layers

        ### Graph
        # Embed
        self.layer_embed = torch.nn.Embedding(
            num_embeddings=self.vocab_target_size,
            embedding_dim=self.dim_embed,
            padding_idx=0,
        )

        # RNN
        self.layer_rnn = torch.nn.GRU(
            input_size=self.dim_embed,
            hidden_size=self.dim_hidden,
            dropout=self.dropout,
            num_layers=self.rnn_num_layers,
        )

        # Output Linear
        self.layer_output = torch.nn.Linear(
            in_features=self.dim_hidden,
            out_features=self.vocab_target_size,
        )

        # Dropout
        self.layer_dropout = torch.nn.Dropout(self.dropout)
        return

    def forward(self,
            input: torch.Tensor,  # [batch]
            hiddens_in: torch.Tensor,  # [num_rnn, batch, dim_hidden] (a pair of this for LSTM)
    ):
        input = input.unsqueeze(0)  # [seq(1), batch]
        embedded = self.layer_dropout(self.layer_embed(input))  # [seq(1), batch, dim_embed]
        output, hidden = self.layer_rnn(embedded, hiddens_in)  # [seq(1), batch, dim_hidden * num_direction], [num_rnn, batch, dim_hidden] (a pair of this for LSTM)

        prediction = self.layer_output(output.squeeze(0))  # [batch, dim_hidden * num_direction]
        return prediction, hidden


# https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
class Seq2SeqPlain(torch.nn.Module):

    def __init__(self, device,
            dim_embed: int,
            dim_hidden: int,
            rnn_num_layers: int,
            is_bidirectional: bool,
            dropout: float,
            teacher_forcing_ratio: float,
            vocab_input_size: int,
            vocab_target_size: int,
    ):
        super().__init__()
        self.device = device
        self.dim_embed: int = dim_embed
        self.dim_hidden: int = dim_hidden
        self.rnn_num_layers: int = rnn_num_layers
        self.is_bidirectional: bool = is_bidirectional
        self.dropout: float = dropout
        self.teacher_forcing_ratio: float = teacher_forcing_ratio
        self.vocab_input_size: int = vocab_input_size
        self.vocab_target_size: int = vocab_target_size

        self.encoder = Encoder(self.device,
            dim_embed=self.dim_embed,
            dim_hidden=self.dim_hidden,
            dropout=self.dropout,
            vocab_input_size=self.vocab_input_size,
            rnn_num_layers=self.rnn_num_layers,
            is_bidirectional=self.is_bidirectional,
        )

        self.decoder = DecoderPlain(self.device,
            dim_embed=self.dim_embed,
            dim_hidden=self.dim_hidden,
            dropout=self.dropout,
            vocab_target_size=self.vocab_target_size,
            rnn_num_layers=self.rnn_num_layers,
        )

        self.layer_decoder_softmax = torch.nn.LogSoftmax(dim=1)
        return

    def forward(self,
            inputs: torch.Tensor,  # [seq, batch]
            inputs_seq_lens,  # [batch]
            targets: torch.Tensor,  # [seq, batch], for teacher forcing
    ):  # [seq, batch]
        batch_size = targets.shape[1]
        max_len = targets.shape[0]

        # Tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, self.decoder.vocab_target_size, device=self.device)

        _, hidden = self.encoder(inputs, inputs_seq_lens)

        # First input: <BOS>
        input = targets[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            is_teacher_forcing = (random.random() < self.teacher_forcing_ratio) and self.training  # Teacher forcing only effective during training
            input = targets[t] if is_teacher_forcing else output.max(1)[1]
        # end for

        return outputs

    def beam_search(self,
            input: torch.Tensor,  # [seq]
            beam_search_size: int,
            bos: int,
            eos: int,
            max_len: int,
    ) -> List[Tuple[torch.Tensor, float]]:  # [beam]
        finished_candidates_logprobs = list()

        # Encode
        _, hidden = self.encoder(input.unsqueeze(1), [input.size(0)])  # [num_rnn, batch(1), dim_hidden] (a pair of this for LSTM)

        # Initial sequence: [<BOS>]
        candidate_init = torch.full([1], bos, dtype=torch.long, device=self.device)
        candidates_logprobs = [(candidate_init, hidden, 0)]

        # Beam search
        for t in range(1, max_len):
            if len(candidates_logprobs) == 0:  break

            input_beam = torch.tensor([c[-1] for c,_,_ in candidates_logprobs]).to(self.device)  # [batch]
            hidden_beam = torch.cat([h for _,h,_ in candidates_logprobs], dim=1)  # [num_rnn, batch, dim_hidden]
            output_beam, hidden_beam = self.decoder(input_beam, hidden_beam)  # [batch, vocab_target_size], [num_rnn, batch, dim_hidden]
            output_beam = self.layer_decoder_softmax(output_beam)

            next_candidates_logprobs = list()
            for i, (c, _, p) in enumerate(candidates_logprobs):
                logprobs, next_cs = torch.sort(output_beam[i], descending=True)
                for j in range(beam_search_size):
                    next_candidate = torch.cat([c, next_cs[j:j+1]])
                    next_logprob = p + logprobs[j].item()
                    if next_cs[j].item() == eos:
                        finished_candidates_logprobs.append((next_candidate, next_logprob))
                    else:
                        next_candidates_logprobs.append((next_candidate, hidden_beam[:,i:i+1,:], next_logprob))
                    # end if
                # end for
            # end for

            candidates_logprobs = sorted(next_candidates_logprobs, key=lambda chp: chp[2], reverse=True)[:beam_search_size]
        # end for

        return finished_candidates_logprobs


# https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
class Attention(torch.nn.Module):

    def get_dim_encoder_hidden_with_direction(self):  return self.dim_decoder_hidden * (2 if self.is_encoder_bidirectional else 1)

    def __init__(self, device,
            dim_encoder_hidden: int,
            is_encoder_bidirectional: bool,
            dim_decoder_hidden: int,
    ):
        super().__init__()

        self.device = device
        self.dim_encoder_hidden = dim_encoder_hidden
        self.is_encoder_bidirectional = is_encoder_bidirectional
        self.dim_decoder_hidden = dim_decoder_hidden

        self.layer_attn = torch.nn.Linear(
            in_features=self.get_dim_encoder_hidden_with_direction() + self.dim_decoder_hidden,
            out_features=self.dim_decoder_hidden,
        )
        self.v = torch.nn.Parameter(torch.rand(self.dim_decoder_hidden))  # [dim_decoder_hidden]
        return

    def forward(self,
            hidden: torch.Tensor,  # [batch, dim_decoder_hidden]
            encoder_outputs: torch.Tensor,  # [seq, batch, dim_encoder_hidden*num_encoder_direction]
    ):  # [batch, seq]
        batch_size = encoder_outputs.shape[1]
        seq_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq, dim_decoder_hidden]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch, seq, dim_encoder_hidden*num_encoder_direction]

        energy = torch.tanh(self.layer_attn(torch.cat(
            (hidden, encoder_outputs), dim=2
        )))  # [batch, seq, dim_decoder_hidden]
        energy = energy.permute(0, 2, 1)  # [batch, dim_decoder_hidden, seq]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch, 1, dim_decoder_hidden]

        attention = torch.bmm(v, energy).squeeze(1)  # [batch, seq]

        return torch.softmax(attention, dim=1)


# https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
class DecoderWithAttention(torch.nn.Module):

    def get_dim_encoder_hidden_with_direction(self):  return self.dim_decoder_hidden * (2 if self.is_encoder_bidirectional else 1)

    def __init__(self, device,
            dim_embed: int,
            dim_encoder_hidden: int,
            is_encoder_bidirectional: bool,
            dim_decoder_hidden: int,
            dropout: float,
            rnn_num_layers: int,
            vocab_target_size: int,
    ):
        super().__init__()

        self.device = device
        self.dim_embed = dim_embed
        self.dim_encoder_hidden = dim_encoder_hidden
        self.is_encoder_bidirectional = is_encoder_bidirectional
        self.dim_decoder_hidden = dim_decoder_hidden
        self.dropout = dropout
        self.rnn_num_layers = rnn_num_layers
        self.vocab_target_size = vocab_target_size

        self.layer_attention = Attention(self.device,
            dim_encoder_hidden=self.dim_encoder_hidden,
            is_encoder_bidirectional=self.is_encoder_bidirectional,
            dim_decoder_hidden=self.dim_decoder_hidden,
        )

        self.layer_embed = torch.nn.Embedding(
            num_embeddings=self.vocab_target_size,
            embedding_dim=self.dim_embed,
            padding_idx=0,
        )

        self.layer_rnn = torch.nn.GRU(
            input_size=self.get_dim_encoder_hidden_with_direction()+self.dim_embed,
            hidden_size=self.dim_decoder_hidden,
            num_layers=1,
        )

        self.layer_out = torch.nn.Linear(
            in_features=self.get_dim_encoder_hidden_with_direction()+self.dim_decoder_hidden+self.dim_embed,
            out_features=self.vocab_target_size,
        )

        self.layer_dropout = torch.nn.Dropout(self.dropout)
        return

    def forward(self,
            input: torch.Tensor,  # [batch]
            hidden: torch.Tensor,  # [batch, dim_decoder_hidden]
            encoder_outputs: torch.Tensor,  # [seq, batch, dim_encoder_hidden*num_direction]
    ):
        input = input.unsqueeze(0)  # [seq(1), batch]
        embedded = self.layer_dropout(self.layer_embed(input))  # [seq(1), batch, dim_embed]
        attn = self.layer_attention(hidden, encoder_outputs).unsqueeze(1)  # [batch, 1, seq]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch, seq, dim_encoder_hidden*num_direction]

        weighted = torch.bmm(attn, encoder_outputs)  # [batch, 1, dim_encoder_hidden*num_direction]
        weighted = weighted.permute(1, 0, 2)  # [1, batch, dim_encoder_hidden*num_direction]

        rnn_input = torch.cat((embedded, weighted), dim=2)  # [seq(1), batch, dim_embed + dim_encoder_hidden*num_direction]
        hidden = hidden.unsqueeze(0)  # [seq(1), batch, dim_decoder_hidden]
        output, hidden = self.layer_rnn(rnn_input, hidden)  # [seq(1), batch, dim_decoder_hidden], [1, batch, dim_decoder_hidden]

        embedded = embedded.squeeze(0)  # [batch, dim_embed]
        output = output.squeeze(0)  # [batch, dim_decoder_hidden]
        weighted = weighted.squeeze(0)  # [batch, dim_encoder_hidden*num_direction]

        output = self.layer_out(torch.cat(
            (output, weighted, embedded), dim=1))  # [batch, vocab_target_size]

        hidden = hidden.squeeze(0)  # [batch, dim_decoder_hidden]
        return output, hidden


# https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
class Seq2SeqWithAttention(torch.nn.Module):

    def __init__(self, device,
            dim_embed: int,
            dim_encoder_hidden: int,
            dim_decoder_hidden: int,
            rnn_num_layers: int,
            is_bidirectional: bool,
            dropout: float,
            teacher_forcing_ratio: float,
            vocab_input_size: int,
            vocab_target_size: int,
    ):
        super().__init__()
        self.device = device

        self.dim_embed = dim_embed
        self.dim_encoder_hidden = dim_encoder_hidden
        self.dim_decoder_hidden = dim_decoder_hidden
        self.rnn_num_layers = rnn_num_layers
        self.is_bidirectional = is_bidirectional
        self.dropout = dropout
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.vocab_input_size = vocab_input_size
        self.vocab_target_size = vocab_target_size

        self.encoder = Encoder(self.device,
            dim_embed=self.dim_embed,
            dim_hidden=self.dim_encoder_hidden,
            rnn_num_layers=self.rnn_num_layers,
            is_bidirectional=self.is_bidirectional,
            dropout=self.dropout,
            vocab_input_size=self.vocab_input_size,
        )

        self.decoder = DecoderWithAttention(self.device,
            dim_embed=self.dim_embed,
            dim_encoder_hidden=self.dim_encoder_hidden,
            is_encoder_bidirectional=self.is_bidirectional,
            dim_decoder_hidden=self.dim_decoder_hidden,
            dropout=self.dropout,
            rnn_num_layers=self.rnn_num_layers,
            vocab_target_size=self.vocab_target_size,
        )

        self.layer_fc_hidden = torch.nn.Linear(
            in_features=self.dim_encoder_hidden,
            out_features=self.dim_decoder_hidden,
        )

        self.layer_decoder_softmax = torch.nn.LogSoftmax(dim=1)
        return

    def forward(self,
            inputs: torch.Tensor,  # [seq, batch]
            inputs_seq_lens,  # [batch]
            targets: torch.Tensor,  # [seq, batch]
    ):  # [seq, batch, vocab_target_size]
        batch_size = inputs.shape[1]
        max_len = targets.shape[0]

        outputs = torch.zeros(max_len, batch_size, self.vocab_target_size, device=self.device)

        encoder_outputs, hidden = self.encoder(inputs, inputs_seq_lens)  # [seq, batch, dim_hidden*num_direction], [num_rnn_layer, batch, dim_encoder_hidden]
        hidden = self.layer_fc_hidden(hidden[-1])  # [batch, dim_decoder_hidden]

        decoder_input = targets[0, :]  # first input: <BOS>

        for t in range(1, max_len):
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[t] = output
            is_teacher_forcing = (random.random() < self.teacher_forcing_ratio) and self.training  # Teacher forcing only effective during training
            decoder_input = targets[t] if is_teacher_forcing else output.max(1)[1]
        # end for

        return outputs

    def beam_search(self,
            input: torch.Tensor,  # [seq]
            beam_search_size: int,
            bos: int,
            eos: int,
            max_len: int,
    ) -> List[Tuple[torch.Tensor, float]]:  # [beam]
        finished_candidates_logprobs = list()

        # Encode
        encoder_outputs, hidden = self.encoder(input.unsqueeze(1), [input.size(0)])  # [seq, batch(1), dim_hidden*num_direction], [num_rnn_layer, batch(1), dim_encoder_hidden]
        hidden = self.layer_fc_hidden(hidden[-1])  # [batch(1), dim_decoder_hidden]

        # Initial sequence: [<BOS>]
        candidate_init = torch.full([1], bos, dtype=torch.long, device=self.device)
        candidates_logprobs = [(candidate_init, hidden, 0)]

        # Beam search
        for t in range(1, max_len):
            if len(candidates_logprobs) == 0:  break

            input_beam = torch.tensor([c[-1] for c,_,_ in candidates_logprobs]).to(self.device)  # [batch]
            hidden_beam = torch.cat([h for _,h,_ in candidates_logprobs], dim=0)  # [batch, dim_decoder_hidden]
            batch_size = input_beam.shape[0]
            output_beam, hidden_beam = self.decoder(input_beam, hidden_beam, encoder_outputs.repeat(1, batch_size, 1))  # [batch, vocab_target_size], [batch, dim_decoder_hidden]
            output_beam = self.layer_decoder_softmax(output_beam)

            next_candidates_logprobs = list()
            for i, (c, _, p) in enumerate(candidates_logprobs):
                logprobs, next_cs = torch.sort(output_beam[i], descending=True)
                for j in range(beam_search_size):
                    next_candidate = torch.cat([c, next_cs[j:j+1]])
                    next_logprob = p + logprobs[j].item()
                    if next_cs[j].item() == eos:
                        finished_candidates_logprobs.append((next_candidate, next_logprob))
                    else:
                        next_candidates_logprobs.append((next_candidate, hidden_beam[i:i+1,:], next_logprob))
                    # end if
                # end for
            # end for

            candidates_logprobs = sorted(next_candidates_logprobs, key=lambda chp: chp[2], reverse=True)[:beam_search_size]
        # end for

        return finished_candidates_logprobs
