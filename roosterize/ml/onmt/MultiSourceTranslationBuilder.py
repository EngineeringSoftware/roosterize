import torch

from seutil import LoggingUtils

from roosterize.ml.onmt.MultiSourceTranslation import MultiSourceTranslation


class MultiSourceTranslationBuilder:

    logger = LoggingUtils.get_logger(__name__)

    def __init__(self, src_types, data, fields, n_best=1, replace_unk=False,
                 has_tgt=False, phrase_table=""):
        self.src_types = src_types
        self.data = data
        self.fields = fields
        self._has_text_src = True  # PN: all text for now
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.phrase_table = phrase_table
        self.has_tgt = has_tgt

    def _build_target_tokens(self, src_list, src_vocab, src_raw, pred, attn):
        tgt_field = dict(self.fields)["tgt"].base_field
        vocab = tgt_field.vocab
        tokens = []
        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)])
            # end if
            if tokens[-1] == tgt_field.eos_token:
                tokens = tokens[:-1]
                break
        if self.replace_unk and attn is not None and src_list[0] is not None:
            for i in range(len(tokens)):
                if tokens[i] == tgt_field.unk_token:
                    _, max_index = attn[i][:len(src_raw)].max(0)
                    tokens[i] = src_raw[max_index.item()]
                    if self.phrase_table != "":
                        with open(self.phrase_table, "r") as f:
                            for line in f:
                                if line.startswith(src_raw[max_index.item()]):
                                    tokens[i] = line.split('|||')[1].strip()
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, attn, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        # Sorting
        inds, perm = torch.sort(batch.indices)
        src_list = list()
        if self._has_text_src:
            for src_type in self.src_types:
                src_list.append(getattr(batch, f"src.{src_type}")[0][:, :, 0].index_select(1, perm))
        else:
            src_list = [None] * len(self.src_types)
        # end if
        tgt = batch.tgt[:, :, 0].index_select(1, perm) \
            if self.has_tgt else None

        translations = []
        for b in range(batch_size):
            src_raw_list = list()
            if self._has_text_src:
                src_vocab = self.data.src_vocabs[inds[b]] \
                    if self.data.src_vocabs else None
                for src_type in self.src_types:
                    src_raw_list.append(getattr(self.data.examples[inds[b]], f"src.{src_type}")[0])
                # end for
            else:
                src_vocab = [None] * len(self.src_types)
                src_raw_list = [None]
            pred_sents = [self._build_target_tokens(
                [src[:, b] if src is not None else None for src in src_list],
                src_vocab, src_raw_list,
                preds[b][n], attn[b][n])
                for n in range(self.n_best)]
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    [src[:, b] if src is not None else None for src in src_list],
                    src_vocab, src_raw_list,
                    tgt[1:, b] if tgt is not None else None, None)

            translation = MultiSourceTranslation(
                [src[:, b] if src is not None else None for src in src_list],
                src_raw_list, pred_sents, attn[b], pred_score[b],
                gold_sent, gold_score[b]
            )
            translations.append(translation)

        return translations


