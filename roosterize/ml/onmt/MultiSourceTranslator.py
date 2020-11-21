from typing import *

import codecs
from itertools import count
from roosterize.ml.onmt.CustomTranslator import CustomTranslator
from roosterize.ml.onmt.MultiSourceDataset import MultiSourceDataset
from roosterize.ml.onmt.MultiSourceInputter import MultiSourceInputter
from roosterize.ml.onmt.MultiSourceModelBuilder import MultiSourceModelBuilder
from roosterize.ml.onmt.MultiSourceTranslation import MultiSourceTranslation
from roosterize.ml.onmt.MultiSourceTranslationBuilder import MultiSourceTranslationBuilder
import onmt
import onmt.inputters as inputters
from onmt.modules.copy_generator import collapse_copy_scores
from onmt.translate.beam_search import BeamSearch
from onmt.utils.misc import tile
import os
import time
import torch

from seutil import LoggingUtils


class LoadedModel(NamedTuple):
    fields: any
    model: any
    model_opt: any


class MultiSourceTranslator(CustomTranslator):

    logger = LoggingUtils.get_logger(__name__)

    @classmethod
    def load_model(cls, src_types, opt) -> LoadedModel:
        fields, model, model_opt = MultiSourceModelBuilder.load_test_model(src_types, opt)
        return LoadedModel(fields, model, model_opt)

    @classmethod
    def build_translator(
            cls,
            src_types,
            opt,
            loaded_model: LoadedModel = None,
            report_score=True,
            logger=None,
            out_file=None,
    ):
        if out_file is None:
            out_file = codecs.open(opt.output, 'w+', 'utf-8')

        assert len(opt.models) == 1, "ensemble model is not supported"

        if loaded_model is None:
            fields, model, model_opt = MultiSourceModelBuilder.load_test_model(src_types, opt)
        else:
            fields, model, model_opt = loaded_model

        scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)

        translator = cls.from_opt(
            src_types,
            model,
            fields,
            opt,
            model_opt,
            global_scorer=scorer,
            out_file=out_file,
            report_score=report_score,
            logger=logger
        )
        return translator

    def __init__(self, src_types, model, fields, src_reader, tgt_reader, gpu=-1, n_best=1, min_length=0, max_length=100, ratio=0., beam_size=30, random_sampling_topk=1, random_sampling_temp=1, stepwise_penalty=None, dump_beam=False, block_ngram_repeat=0, ignore_when_blocking=frozenset(), replace_unk=False, phrase_table="", data_type="text", verbose=False, report_bleu=False, report_rouge=False, report_time=False, copy_attn=False, global_scorer=None, out_file=None, report_score=True, logger=None, seed=-1):
        super().__init__(model, fields, src_reader, tgt_reader, gpu, n_best, min_length, max_length, ratio, beam_size, random_sampling_topk, random_sampling_temp, stepwise_penalty, dump_beam, block_ngram_repeat, ignore_when_blocking, replace_unk, phrase_table, data_type, verbose, report_bleu, report_rouge, report_time, copy_attn, global_scorer, out_file, report_score, logger, seed)
        self.src_types = src_types
        return

    @classmethod
    def from_opt(cls,
            src_types,
            model,
            fields,
            opt,
            model_opt,
            global_scorer=None,
            out_file=None,
            report_score=True,
            logger=None):
        """Alternate constructor.

        Args:
            model (onmt.modules.NMTModel): See :func:`__init__()`.
            fields (dict[str, torchtext.data.Field]): See
                :func:`__init__()`.
            opt (argparse.Namespace): Command line options
            model_opt (argparse.Namespace): Command line options saved with
                the model checkpoint.
            global_scorer (onmt.translate.GNMTGlobalScorer): See
                :func:`__init__()`..
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            report_score (bool) : See :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """

        src_reader = inputters.str2reader["text"].from_opt(opt)
        tgt_reader = inputters.str2reader["text"].from_opt(opt)
        return cls(
            src_types,
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=opt.gpu,
            n_best=opt.n_best,
            min_length=opt.min_length,
            max_length=opt.max_length,
            ratio=opt.ratio,
            beam_size=opt.beam_size,
            random_sampling_topk=opt.random_sampling_topk,
            random_sampling_temp=opt.random_sampling_temp,
            stepwise_penalty=opt.stepwise_penalty,
            dump_beam=opt.dump_beam,
            block_ngram_repeat=opt.block_ngram_repeat,
            ignore_when_blocking=set(opt.ignore_when_blocking),
            replace_unk=opt.replace_unk,
            phrase_table=opt.phrase_table,
            data_type=opt.data_type,
            verbose=opt.verbose,
            report_bleu=opt.report_bleu,
            report_rouge=opt.report_rouge,
            report_time=opt.report_time,
            copy_attn=model_opt.copy_attn,
            global_scorer=global_scorer,
            out_file=out_file,
            report_score=report_score,
            logger=logger,
            seed=opt.seed)

    def translate(self,
            raw_data_shard: Dict,
            has_target: bool,
            src_dir=None,
            batch_size=None,
            attn_debug=False,
            phrase_table=""):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            src: See :func:`self.src_reader.read()`.
            tgt: See :func:`self.tgt_reader.read()`.
            src_dir: See :func:`self.src_reader.read()` (only relevant
                for certain types of data).
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        candidates_logprobs: List[List[Tuple[List[str], float]]] = list()

        if batch_size is None:
            raise ValueError("batch_size must be set")

        raw_data_keys = [f"src.{src_type}" for src_type in self.src_types] + (["tgt"] if has_target else [])
        tgt = raw_data_shard.get("tgt")

        data = MultiSourceDataset(
            self.src_types,
            self.fields,
            readers=([self.src_reader] * len(self.src_types) + ([self.tgt_reader] if self.tgt_reader else [])),
            data=[(k, raw_data_shard[k]) for k in raw_data_keys],
            dirs=[None] * len(raw_data_keys),
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred,
            can_copy=self.copy_attn,
        )

        data_iter = MultiSourceInputter.OrderedIterator(
            src_types=self.src_types,
            dataset=data,
            device=self._dev,
            batch_size=batch_size,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        xlation_builder = MultiSourceTranslationBuilder(
            self.src_types,
            data, self.fields, self.n_best, self.replace_unk, tgt,
            self.phrase_table
        )

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        start_time = time.time()

        for batch in data_iter:
            batch_data = self.translate_batch(
                batch, data.src_vocabs, attn_debug
            )
            translations = xlation_builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                all_predictions += [n_best_preds]

                candidates_logprobs.append([
                    (trans.pred_sents[idx], trans.pred_scores[idx].item())
                    for idx in range(self.n_best)
                ])
                self.out_file.write('\n'.join(n_best_preds) + '\n')
                self.out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode('utf-8'))

                if attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append('</s>')
                    attns = trans.attns[0].tolist()
                    if self.data_type == 'text':
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(attns[0]))]
                    header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
                    row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    output = header_format.format("", *srcs) + '\n'
                    for word, row in zip(preds, attns):
                        max_index = row.index(max(row))
                        row_format = row_format.replace(
                            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
                        row_format = row_format.replace(
                            "{:*>10.7f} ", "{:>10.7f} ", max_index)
                        output += row_format.format(word, *row) + '\n'
                        row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode('utf-8'))

        end_time = time.time()

        if self.report_score:
            msg = self._report_score('PRED', pred_score_total,
                                     pred_words_total)
            self._log(msg)
            if tgt is not None:
                msg = self._report_score('GOLD', gold_score_total,
                                         gold_words_total)
                self._log(msg)
                if self.report_bleu:
                    msg = self._report_bleu(tgt)
                    self._log(msg)
                if self.report_rouge:
                    msg = self._report_rouge(tgt)
                    self._log(msg)

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total translation time (s): %f" % total_time)
            self._log("Average translation time (s): %f" % (
                total_time / len(all_predictions)))
            self._log("Tokens per second: %f" % (
                pred_words_total / total_time))

        if self.dump_beam:
            import json
            json.dump(self.translator.beam_accum,
                      codecs.open(self.dump_beam, 'w', 'utf-8'))
        return all_scores, all_predictions, candidates_logprobs

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        with torch.no_grad():
            if self.beam_size == 1:
                return self._translate_random_sampling(
                    batch,
                    src_vocabs,
                    self.max_length,
                    min_length=self.min_length,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    return_attention=attn_debug or self.replace_unk)
            else:
                return self._translate_batch(
                    batch,
                    src_vocabs,
                    self.max_length,
                    min_length=self.min_length,
                    ratio=self.ratio,
                    n_best=self.n_best,
                    return_attention=attn_debug or self.replace_unk)

    def _translate_batch(
            self,
            batch,
            src_vocabs,
            max_length,
            min_length=0,
            ratio=0.,
            n_best=1,
            return_attention=False):
        # TODO: support these blacklisted features.
        assert not self.dump_beam

        # (0) Prep the components of the search.
        use_src_map = self.copy_attn
        beam_size = self.beam_size
        batch_size = batch.batch_size

        # (1) Run the encoder on the src.
        src_list, enc_states_list, memory_bank_list, src_lengths_list = self._run_encoder(batch)
        self.model.decoder.init_state(src_list, memory_bank_list, enc_states_list)

        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": self._gold_score(
                batch, memory_bank_list, src_lengths_list, src_vocabs, use_src_map,
                enc_states_list, batch_size, src_list)}

        # (2) Repeat src objects `beam_size` times.
        # We use batch_size x beam_size
        src_map_list = list()
        for src_type in self.src_types:
            src_map_list.append((tile(getattr(batch, f"src_map.{src_type}"), beam_size, dim=1) if use_src_map else None))
        # end for

        self.model.decoder.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))

        memory_lengths_list = list()
        for src_i in range(len(memory_bank_list)):
            if isinstance(memory_bank_list[src_i], tuple):
                memory_bank_list[src_i] = tuple(tile(x, beam_size, dim=1) for x in memory_bank_list[src_i])
                mb_device = memory_bank_list[src_i][0].device
            else:
                memory_bank_list[src_i] = tile(memory_bank_list[src_i], beam_size, dim=1)
                mb_device = memory_bank_list[src_i].device
            # end if

            memory_lengths_list.append(tile(src_lengths_list[src_i], beam_size))
        # end for

        # (0) pt 2, prep the beam object
        beam = BeamSearch(
            beam_size,
            n_best=n_best,
            batch_size=batch_size,
            global_scorer=self.global_scorer,
            pad=self._tgt_pad_idx,
            eos=self._tgt_eos_idx,
            bos=self._tgt_bos_idx,
            min_length=min_length,
            ratio=ratio,
            max_length=max_length,
            mb_device=mb_device,
            return_attention=return_attention,
            stepwise_penalty=self.stepwise_penalty,
            block_ngram_repeat=self.block_ngram_repeat,
            exclusion_tokens=self._exclusion_idxs,
            memory_lengths=memory_lengths_list)

        for step in range(max_length):
            decoder_input = beam.current_predictions.view(1, -1, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank_list,
                batch,
                src_vocabs,
                memory_lengths_list=memory_lengths_list,
                src_map_list=src_map_list,
                step=step,
                batch_offset=beam._batch_offset)

            beam.advance(log_probs, attn)
            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break

            select_indices = beam.current_origin

            if any_beam_is_finished:
                # Reorder states.
                for src_i in range(len(memory_bank_list)):
                    if isinstance(memory_bank_list[src_i], tuple):
                        memory_bank_list[src_i] = tuple(x.index_select(1, select_indices)
                                            for x in memory_bank_list[src_i])
                    else:
                        memory_bank_list[src_i] = memory_bank_list[src_i].index_select(1, select_indices)
                    # end if

                    memory_lengths_list[src_i] = memory_lengths_list[src_i].index_select(0, select_indices)
                # end for

                if use_src_map and src_map_list[0] is not None:
                    for src_i in range(len(src_map_list)):
                        src_map_list[src_i] = src_map_list[src_i].index_select(1, select_indices)
                    # end for
                # end if

            self.model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = beam.scores
        results["predictions"] = beam.predictions
        results["attention"] = beam.attention
        return results

    def _run_encoder(self, batch):
        src_list = list()
        src_lengths_list = list()
        for src_type in self.src_types:
            batch_src = getattr(batch, f"src.{src_type}")
            src, src_lengths = batch_src if isinstance(batch_src, tuple) else (batch_src, None)
            src_list.append(src)
            src_lengths_list.append(src_lengths)
        # end for

        enc_states_list = list()
        memory_bank_list = list()
        for enc_i, encoder in enumerate(self.model.encoders):
            enc_states, memory_bank, src_lengths = encoder(src_list[enc_i], src_lengths_list[enc_i])
            if src_lengths is None:
                assert not isinstance(memory_bank, tuple), \
                    'Ensemble decoding only supported for text data'
                src_lengths = torch.Tensor(batch.batch_size) \
                                   .type_as(memory_bank) \
                                   .long() \
                                   .fill_(memory_bank.size(0))
            # end if
            enc_states_list.append(enc_states)
            memory_bank_list.append(memory_bank)
            src_lengths_list[enc_i] = src_lengths
        # end for
        return src_list, enc_states_list, memory_bank_list, src_lengths_list

    def _decode_and_generate(
            self,
            decoder_in,
            memory_bank_list,
            batch,
            src_vocabs,
            memory_lengths_list,
            src_map_list=None,
            step=None,
            batch_offset=None):
        if self.copy_attn:
            # Turn any copied words into UNKs.
            decoder_in = decoder_in.masked_fill(
                decoder_in.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx
            )

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder(
            decoder_in, memory_bank_list, memory_lengths_list=memory_lengths_list, step=step
        )

        # Generator forward.
        if not self.copy_attn:
            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None
            log_probs = self.model.generator(dec_out.squeeze(0))
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        else:
            attn = dec_attn["copy"]
            scores = self.model.generator(dec_out.view(-1, dec_out.size(2)),
                                          attn.view(-1, attn.size(2)),
                                          src_map_list)
            # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
            if batch_offset is None:
                scores = scores.view(batch.batch_size, -1, scores.size(-1))
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            scores = collapse_copy_scores(
                scores,
                batch,
                self._tgt_vocab,
                src_vocabs,
                batch_dim=0,
                batch_offset=batch_offset
            )
            scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
            log_probs = scores.squeeze(0).log()
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        return log_probs, attn

    def _score_target(self, batch, memory_bank_list, src_lengths_list,
                      src_vocabs, src_map_list):
        tgt = batch.tgt
        tgt_in = tgt[:-1]

        log_probs, attn = self._decode_and_generate(
            tgt_in, memory_bank_list, batch, src_vocabs,
            memory_lengths_list=src_lengths_list, src_map_list=src_map_list)

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[1:]
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores

    def _gold_score(self, batch, memory_bank, src_lengths, src_vocabs,
                    use_src_map, enc_states, batch_size, src):
        if "tgt" in batch.__dict__:
            if use_src_map:
                src_map_list = list()
                for src_type in self.src_types:  src_map_list.append(getattr(batch, f"src_map.{src_type}"))
            else:
                src_map_list = None
            # end if
            gs = self._score_target(
                batch, memory_bank, src_lengths, src_vocabs,
                src_map_list)
            self.model.decoder.init_state(src, memory_bank, enc_states)
        else:
            gs = [0] * batch_size
        return gs
