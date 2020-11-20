import collections
import gc
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, NoReturn, Optional, Tuple

import numpy as np
import torch
import torch.nn
import torch.nn.utils
from recordclass import RecordClass
from seutil import IOUtils, LoggingUtils

from roosterize.data.Definition import Definition
from roosterize.data.Lemma import Lemma
from roosterize.data.LemmaBackendSexpTransformers import LemmaBackendSexpTransformers
from roosterize.data.LemmaForeendSexpTransformers import LemmaForeendSexpTransformers
from roosterize.data.ModelSpec import ModelSpec
from roosterize.Macros import Macros
from roosterize.ml.naming.NamingModelBase import NamingModelBase
from roosterize.ml.naming.SubTokenizer import SubTokenizer
from roosterize.Utils import Utils

logger = LoggingUtils.get_logger(__name__)


class Consts:

    ENCODER_RNN = "rnn"
    ENCODER_BRNN = "brnn"
    ENCODER_MEAN = "mean"
    ENCODER_TRANSFORMER = "transformer"
    ENCODER_CNN = "cnn"

    DECODER_RNN = "rnn"
    DECODER_TRANSFORMER = "transformer"
    DECODER_CNN = "cnn"

    INPUT_SEQ = "s"
    INPUT_BSEXP_ORIG = "bsexp"
    INPUT_BSEXP_L1 = "bsexpl1"
    INPUT_BSEXP_L2 = "bsexpl2"
    INPUT_BSEXP_NOPARA_ORIG = "bsexpnp"
    INPUT_BSEXP_NOPARA_L1 = "bsexpnpl1"
    INPUT_BSEXP_NOPARA_L2 = "bsexpnpl2"
    INPUT_FSEXP_L0 = "fsexp"
    INPUT_FSEXP_NOPARA_L0 = "fsexpnp"
    INPUT_FSEXP_L1 = "fsexpl1"
    INPUT_FSEXP_NOPARA_L1 = "fsexpnpl1"

    INPUT_BSEXP_DEPTH10 = "bsexpd10"
    INPUT_BSEXP_RAND = "bsexprnd"
    INPUT_BSEXP_L1x = "bsexpl1x"
    INPUT_FSEXP_DEPTH10 = "fsexpd10"
    INPUT_FSEXP_RAND = "fsexprnd"
    INPUT_FSEXP_L1x = "fsexpl1x"

    OUTPUT_CHAR = "c"
    OUTPUT_SUBTOKEN = "st"


class MultiSourceSeq2SeqConfig(RecordClass):
    # Encoder/Decoder
    encoder: str = Consts.ENCODER_BRNN
    decoder: str = Consts.DECODER_RNN

    # Dimensions
    dim_encoder_hidden: int = 500
    dim_decoder_hidden: int = 500
    dim_embed: int = 500

    # Input/output
    inputs: str = Consts.INPUT_SEQ
    input_max: int = 3000  # Fixed
    output: str = Consts.OUTPUT_SUBTOKEN

    # Vocab
    vocab_input_frequency_threshold: int = 5

    # Attention
    use_attn: bool = True

    # RNN
    rnn_num_layers: int = 2

    # Dropout
    dropout: float = 0.5

    # Copy
    use_copy: bool = False

    # Beam search
    beam_search_max_len_factor: float = 1.5  # Fixed; Will search for at most 1.5x of maximum length appeared in training set

    # Training configs
    learning_rate: float = 1e-3  # Fixed
    early_stopping_threshold: int = 3  # Fixed
    ckpt_keep_max: int = 3  # Fixed
    max_grad_norm: float = 5  # Fixed
    pretrain: str = None

    batch_size: int = 128

    def __str__(self, modulo_pretrain: bool = False):
        s = f"{self.encoder}-{self.inputs}-{self.decoder}-{self.output}-de_{self.dim_embed}-deh_{self.dim_encoder_hidden}-ddh_{self.dim_decoder_hidden}-rnl_{self.rnn_num_layers}-do_{self.dropout}-vift_{self.vocab_input_frequency_threshold}"
        if self.pretrain is not None and not modulo_pretrain:  s += f"-pt_{self.pretrain}"
        if self.use_attn:  s += "+attn"
        if self.use_copy:  s += "+copy"
        return s

    def repOk(self):
        # Check range
        if self.encoder not in [
            Consts.ENCODER_RNN,
            Consts.ENCODER_BRNN,
            Consts.ENCODER_MEAN,
            Consts.ENCODER_TRANSFORMER,
            Consts.ENCODER_CNN,
        ]:
            return False
        if self.decoder not in [
            Consts.DECODER_RNN,
            Consts.DECODER_TRANSFORMER,
            Consts.DECODER_CNN,
        ]:
            return False
        if not set(self.get_src_types()) <= {
            Consts.INPUT_SEQ,
            Consts.INPUT_BSEXP_ORIG,
            Consts.INPUT_BSEXP_NOPARA_ORIG,
            Consts.INPUT_BSEXP_L1,
            Consts.INPUT_BSEXP_NOPARA_L1,
            Consts.INPUT_BSEXP_L2,
            Consts.INPUT_BSEXP_NOPARA_L2,
            Consts.INPUT_FSEXP_L0,
            Consts.INPUT_FSEXP_NOPARA_L0,
            Consts.INPUT_FSEXP_L1,
            Consts.INPUT_FSEXP_NOPARA_L1,
            Consts.INPUT_BSEXP_DEPTH10,
            Consts.INPUT_BSEXP_RAND,
            Consts.INPUT_BSEXP_L1x,
            Consts.INPUT_FSEXP_DEPTH10,
            Consts.INPUT_FSEXP_RAND,
            Consts.INPUT_FSEXP_L1x,
        }:
            return False
        if len(self.get_src_types()) == 0:
            return False
        if self.output not in [
            Consts.OUTPUT_SUBTOKEN,
            Consts.OUTPUT_CHAR,
        ]:
            return False
        if self.dim_encoder_hidden <= 0 or self.dim_decoder_hidden <= 0 or self.dim_embed <= 0:
            return False
        if self.rnn_num_layers <= 0:
            return False
        if not 0 <= self.dropout <= 1:
            return False

        # Because of the way ONMT saves ckpts
        if self.ckpt_keep_max < self.early_stopping_threshold:
            return False

        # Encoder and decoder must have equal hidden state
        if self.dim_encoder_hidden != self.dim_decoder_hidden:
            return False

        # input types must be unique
        if len(set(self.get_src_types())) != len(self.get_src_types()):
            return False

        # Fixed configurations
        if self.early_stopping_threshold != 3\
                or self.learning_rate != 1e-3\
                or self.ckpt_keep_max != 3\
                or self.max_grad_norm != 5\
                or self.beam_search_max_len_factor != 1.5\
                or self.input_max != 3000:
            return False

        return True

    def get_src_types(self) -> List[str]:
        return sorted(self.inputs.split("+"))

    def adjust_batch_size(self):
        batch_size = 128
        for src_type in self.get_src_types():
            if src_type in {
                Consts.INPUT_BSEXP_ORIG,
                Consts.INPUT_BSEXP_NOPARA_ORIG,
                Consts.INPUT_FSEXP_L0,
                Consts.INPUT_FSEXP_NOPARA_L0,
            }:
                batch_size /= 16
            elif src_type in {
                Consts.INPUT_BSEXP_L1,
                Consts.INPUT_BSEXP_NOPARA_L1,
                Consts.INPUT_BSEXP_L2,
                Consts.INPUT_BSEXP_NOPARA_L2,
                Consts.INPUT_FSEXP_L1,
                Consts.INPUT_FSEXP_NOPARA_L1,
                Consts.INPUT_BSEXP_DEPTH10,
                Consts.INPUT_BSEXP_RAND,
                Consts.INPUT_BSEXP_L1x,
                Consts.INPUT_FSEXP_DEPTH10,
                Consts.INPUT_FSEXP_RAND,
                Consts.INPUT_FSEXP_L1x,
            }:
                batch_size /= 2
        if self.rnn_num_layers > 2:
            batch_size *= 2 / self.rnn_num_layers
        if self.dim_embed > 500:
            batch_size *= 500 / self.dim_embed
        if self.dim_encoder_hidden > 500:
            batch_size *= 500 / self.dim_encoder_hidden
        if self.dim_decoder_hidden > 500:
            batch_size *= 500 / self.dim_decoder_hidden
        self.batch_size = max(int(np.ceil(batch_size)), 1)
        return


class MultiSourceSeq2Seq(NamingModelBase[MultiSourceSeq2SeqConfig]):

    def __init__(self, model_dir: Path, model_spec: ModelSpec):
        super().__init__(model_dir, model_spec, MultiSourceSeq2SeqConfig)

        self.open_nmt_path = Macros.project_dir

        if not torch.cuda.is_available():
            self.logger.info("Cuda is not available")
        self.device_tag = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_tag)

        # Cache for processing data
        self.data_cache: dict = dict()
        # Cache for loaded model during translation
        self.loaded_model_cache = None
        return

    def get_input(
            self,
            lemma: Lemma,
            input_type: str,
            docs_sub_tokenizers: Optional[Dict[str, SubTokenizer]],
    ) -> List[str]:
        input: List[str]
        if input_type == Consts.INPUT_SEQ:
            input = [t.content for t in lemma.statement]
        else:
            if input_type in [
                Consts.INPUT_BSEXP_ORIG,
                Consts.INPUT_BSEXP_L1,
                Consts.INPUT_BSEXP_L2,
                Consts.INPUT_FSEXP_L0,
                Consts.INPUT_FSEXP_L1,
                Consts.INPUT_BSEXP_DEPTH10,
                Consts.INPUT_BSEXP_RAND,
                Consts.INPUT_BSEXP_L1x,
                Consts.INPUT_FSEXP_DEPTH10,
                Consts.INPUT_FSEXP_RAND,
                Consts.INPUT_FSEXP_L1x,
            ]:
                use_parathesis = True
            else:
                use_parathesis = False
            # end if

            if input_type.startswith("bsexp"):
                sexp = lemma.backend_sexp

                if input_type in [Consts.INPUT_BSEXP_L1, Consts.INPUT_BSEXP_NOPARA_L1]:
                    sexp = LemmaBackendSexpTransformers.transform(LemmaBackendSexpTransformers.LEVEL_1, sexp)
                elif input_type in [Consts.INPUT_BSEXP_L2, Consts.INPUT_BSEXP_NOPARA_L2]:
                    sexp = LemmaBackendSexpTransformers.transform(LemmaBackendSexpTransformers.LEVEL_2, sexp)
                elif input_type in [Consts.INPUT_BSEXP_DEPTH10]:
                    sexp = LemmaBackendSexpTransformers.transform(LemmaBackendSexpTransformers.DEPTH_10, sexp)
                elif input_type in [Consts.INPUT_BSEXP_RAND]:
                    sexp = LemmaBackendSexpTransformers.transform(LemmaBackendSexpTransformers.RANDOM, sexp)
                elif input_type in [Consts.INPUT_BSEXP_L1x]:
                    sexp = LemmaBackendSexpTransformers.transform(LemmaBackendSexpTransformers.LEVEL_1x, sexp)
            else:
                sexp = lemma.ast_sexp

                if input_type in [Consts.INPUT_FSEXP_L0, Consts.INPUT_FSEXP_NOPARA_L0]:
                    sexp = LemmaForeendSexpTransformers.transform(LemmaForeendSexpTransformers.LEVEL_0, sexp)
                elif input_type in [Consts.INPUT_FSEXP_L1, Consts.INPUT_FSEXP_NOPARA_L1]:
                    sexp = LemmaForeendSexpTransformers.transform(LemmaForeendSexpTransformers.LEVEL_1, sexp)
                elif input_type in [Consts.INPUT_FSEXP_DEPTH10]:
                    sexp = LemmaForeendSexpTransformers.transform(LemmaForeendSexpTransformers.DEPTH_10, sexp)
                elif input_type in [Consts.INPUT_FSEXP_RAND]:
                    sexp = LemmaForeendSexpTransformers.transform(LemmaForeendSexpTransformers.RANDOM, sexp)
                elif input_type in [Consts.INPUT_FSEXP_L1x]:
                    sexp = LemmaForeendSexpTransformers.transform(LemmaForeendSexpTransformers.LEVEL_1x, sexp)

            input = sexp.forward_depth_first_sequence(use_parathesis=use_parathesis)

        # Always sub tokenize input
        sub_tokenizer = docs_sub_tokenizers[lemma.data_index]
        input = [
            st for t in input
            for st in (sub_tokenizer.sub_tokenize(t) if SubTokenizer.can_tokenize(t) else [t])
        ]

        return input[:self.config.input_max]

    def get_all_inputs(
            self,
            lemmas: List[Lemma],
            docs_sub_tokenizers: Optional[Dict[str, SubTokenizer]],
    ) -> Dict[str, List[List[str]]]:
        all_inputs: Dict[str, List[List[str]]] = dict()
        input_types = self.config.get_src_types()
        for input_type in input_types:
            all_inputs[input_type] = list()

        for lemma in lemmas:
            for input_type in input_types:
                all_inputs[input_type].append(self.get_input(lemma, input_type, docs_sub_tokenizers))

        return all_inputs

    def get_output(
            self,
            lemma: Lemma,
            docs_sub_tokenizers: Optional[Dict[str, SubTokenizer]],
    ) -> List[str]:
        if self.config.output == Consts.OUTPUT_CHAR:
            return [c for c in lemma.name]
        elif self.config.output == Consts.OUTPUT_SUBTOKEN:
            sub_tokenizer = docs_sub_tokenizers[lemma.data_index]
            return sub_tokenizer.sub_tokenize(lemma.name)
        else:
            raise ValueError

    def process_data_impl(
            self,
            data_dir: Path,
            output_processed_data_dir: Path,
    ) -> NoReturn:
        lemmas: List[Lemma] = IOUtils.dejsonfy(IOUtils.load(data_dir/"lemmas.json", IOUtils.Format.json), List[Lemma])
        definitions: List[Definition] = IOUtils.dejsonfy(IOUtils.load(data_dir/"definitions.json", IOUtils.Format.json), List[Definition])

        docs_sub_tokenizers = SubTokenizer.get_docs_sub_tokenizers(lemmas, definitions)

        # Inputs
        all_inputs: Dict[str, List[List[str]]] = self.get_all_inputs(lemmas, docs_sub_tokenizers)
        for input_type, src_sentences in all_inputs.items():
            IOUtils.dump(output_processed_data_dir/f"src.{input_type}.txt", src_sentences, IOUtils.Format.txtList)

        # Outputs
        IOUtils.dump(
            output_processed_data_dir/f"tgt.txt",
            "".join([" ".join(self.get_output(lemma, docs_sub_tokenizers)) + "\n" for lemma in lemmas]),
            IOUtils.Format.txt,
        )

        super().process_data_impl(data_dir, output_processed_data_dir)
        return

    def preprocess(
            self,
            train_processed_data_dir: Path,
            val_processed_data_dir: Path,
    ) -> NoReturn:
        from roosterize.ml.onmt.MultiSourceInputter import MultiSourceInputter
        import onmt.inputters as inputters
        from preprocess import _get_parser as preprocess_get_parser
        from preprocess import check_existing_pt_files

        # Create opt (although the X_src files aren't correct and are not used, they're required by parse_args)
        parser = preprocess_get_parser()
        opt = parser.parse_args(
            f" -train_src {train_processed_data_dir}/src.txt"
            f" -train_tgt {train_processed_data_dir}/tgt.txt"
            f" -valid_src {val_processed_data_dir}/src.txt"
            f" -valid_tgt {val_processed_data_dir}/tgt.txt"
            f" -save_data {self.model_dir}/processed-data"
        )
        opt.src_seq_length = self.config.input_max
        opt.src_words_min_frequency = self.config.vocab_input_frequency_threshold
        if self.config.use_copy:  opt.dynamic_dict = True

        # Always check for existing pt files
        check_existing_pt_files(opt)

        # We always have 0 additional features (not counting the tokens themselves)
        src_nfeats = 0
        tgt_nfeats = 0

        self.logger.info("Building `Fields` object...")
        fields = MultiSourceInputter.get_fields(
            self.config.get_src_types(),
            src_nfeats,
            tgt_nfeats,
            dynamic_dict=opt.dynamic_dict,
            src_truncate=opt.src_seq_length_trunc,
            tgt_truncate=opt.tgt_seq_length_trunc,
        )

        src_reader = inputters.str2reader["text"].from_opt(opt)
        tgt_reader = inputters.str2reader["text"].from_opt(opt)

        logger.info("Building & saving training data...")
        self.build_save_dataset(train_processed_data_dir, 'train', fields, src_reader, tgt_reader, True, opt)

        if opt.valid_src and opt.valid_tgt:
            self.logger.info("Building & saving validation data...")
            self.build_save_dataset(val_processed_data_dir, 'valid', fields, src_reader, tgt_reader, True, opt)
        return

    def build_save_dataset(
            self,
            processed_data_dir: Path,
            corpus_type: str, fields, src_reader, tgt_reader, has_target: bool, opt
    ):
        import onmt.inputters as inputters
        from onmt.utils.misc import split_corpus
        from roosterize.ml.onmt.MultiSourceInputter import MultiSourceInputter
        from roosterize.ml.onmt.MultiSourceDataset import MultiSourceDataset

        assert corpus_type in ['train', 'valid']

        raw_data_keys = [f"src.{src_type}" for src_type in self.config.get_src_types()]\
                        + (["tgt"] if has_target else [])
        raw_data_paths: Dict[str, str] = {
            k: f"{processed_data_dir}/{k}.txt"
            for k in raw_data_keys
        }

        if corpus_type == 'train':
            counters = collections.defaultdict(collections.Counter)

        # for src, tgt, maybe_id in zip(srcs, tgts, ids):
        logger.info(f"Reading source and target files: {raw_data_paths.values()}")

        raw_data_shards: Dict[str, list] = {
            k: list(split_corpus(p, opt.shard_size))
            for k, p in raw_data_paths.items()
        }
        # src_shards = split_corpus(src, opt.shard_size)
        # tgt_shards = split_corpus(tgt, opt.shard_size)
        # shard_pairs = zip(src_shards, tgt_shards)
        dataset_paths = []
        if (corpus_type == "train" or opt.filter_valid) and has_target:
            filter_pred = partial(
                MultiSourceInputter.filter_example,
                src_types=self.config.get_src_types(),
                use_src_len=opt.data_type == "text",
                max_src_len=opt.src_seq_length,
                max_tgt_len=opt.tgt_seq_length,
            )
        else:
            filter_pred = None

        if corpus_type == "train":
            existing_fields = None
            if opt.src_vocab != "":
                try:
                    logger.info("Using existing vocabulary...")
                    existing_fields = torch.load(opt.src_vocab)
                except torch.serialization.pickle.UnpicklingError:
                    logger.info("Building vocab from text file...")
                    src_vocab, src_vocab_size = MultiSourceInputter.load_vocab(
                        opt.src_vocab,
                        "src",
                        counters,
                        opt.src_words_min_frequency,
                    )
            else:
                src_vocab = None

            if opt.tgt_vocab != "":
                tgt_vocab, tgt_vocab_size = MultiSourceInputter.load_vocab(
                    opt.tgt_vocab,
                    "tgt",
                    counters,
                    opt.tgt_words_min_frequency,
                )
            else:
                tgt_vocab = None

        for i in range(len(list(raw_data_shards.values())[0])):
        # for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        #     assert len(src_shard) == len(tgt_shard)
            logger.info("Building shard %d." % i)
            dataset = MultiSourceDataset(
                self.config.get_src_types(),
                fields,
                readers=([src_reader] * len(self.config.get_src_types()) + ([tgt_reader] if tgt_reader else [])),
                data=[(k, raw_data_shards[k][i]) for k in raw_data_keys],
                dirs=[None] * len(raw_data_keys),
                sort_key=inputters.str2sortkey[opt.data_type],
                filter_pred=filter_pred,
                can_copy=self.config.use_copy,
            )
            if corpus_type == "train" and existing_fields is None:
                for ex in dataset.examples:
                    for name, field in fields.items():
                        try:
                            f_iter = iter(field)
                        except TypeError:
                            f_iter = [(name, field)]
                            all_data = [getattr(ex, name, None)]
                        else:
                            all_data = getattr(ex, name)
                        for (sub_n, sub_f), fd in zip(f_iter, all_data):
                            has_vocab = (sub_n == 'src' and src_vocab is not None) or \
                                        (sub_n == 'tgt' and tgt_vocab is not None)
                            if (hasattr(sub_f, 'sequential') and sub_f.sequential and not has_vocab):
                                val = fd
                                counters[sub_n].update(val)

            # if maybe_id:
            #     shard_base = corpus_type + "_" + maybe_id
            # else:
            shard_base = corpus_type
            data_path = "{:s}.{:s}.{:d}.pt".format(opt.save_data, shard_base, i)
            dataset_paths.append(data_path)

            logger.info(" * saving %sth %s data shard to %s." % (i, shard_base, data_path))

            dataset.save(data_path)

            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

        if corpus_type == "train":
            vocab_path = opt.save_data + '.vocab.pt'
            if existing_fields is None:
                fields = MultiSourceInputter.build_fields_vocab(
                    self.config.get_src_types(),
                    fields, counters,
                    opt.share_vocab, opt.vocab_size_multiple,
                    opt.src_vocab_size, opt.src_words_min_frequency,
                    opt.tgt_vocab_size, opt.tgt_words_min_frequency)
            else:
                fields = existing_fields
            torch.save(fields, vocab_path)
        return

    def train_impl(
            self,
            train_processed_data_dir: Path,
            val_processed_data_dir: Path,
    ) -> NoReturn:
        self.preprocess(train_processed_data_dir, val_processed_data_dir)

        from train import _get_parser as train_get_parser
        from train import ErrorHandler, batch_producer
        from roosterize.ml.onmt.MultiSourceInputter import MultiSourceInputter
        from onmt.inputters.inputter import old_style_vocab, load_old_vocab
        import onmt.utils.distributed
        from onmt.utils.parse import ArgumentParser

        with IOUtils.cd(self.open_nmt_path):
            parser = train_get_parser()
            opt = parser.parse_args(
                f" -data {self.model_dir}/processed-data"
                f" -save_model {self.model_dir}/models/ckpt"
            )
            opt.gpu_ranks = [0]
            opt.early_stopping = self.config.early_stopping_threshold
            opt.report_every = 200
            opt.valid_steps = 200
            opt.save_checkpoint_steps = 200
            opt.keep_checkpoint_max = self.config.ckpt_keep_max

            opt.optim = "adam"
            opt.learning_rate = self.config.learning_rate
            opt.max_grad_norm = self.config.max_grad_norm
            opt.batch_size = self.config.batch_size

            opt.encoder_type = self.config.encoder
            opt.decoder_type = self.config.decoder
            opt.dropout = [self.config.dropout]
            opt.src_word_vec_size = self.config.dim_embed
            opt.tgt_word_vec_size = self.config.dim_embed
            opt.layers = self.config.rnn_num_layers
            opt.enc_rnn_size = self.config.dim_encoder_hidden
            opt.dec_rnn_size = self.config.dim_decoder_hidden
            opt.__setattr__("num_srcs", len(self.config.get_src_types()))
            if self.config.use_attn:
                opt.global_attention = "general"
            else:
                opt.global_attention = "none"
            if self.config.use_copy:
                opt.copy_attn = True
                opt.copy_attn_type = "general"

            # train.main
            ArgumentParser.validate_train_opts(opt)
            ArgumentParser.update_model_opts(opt)
            ArgumentParser.validate_model_opts(opt)

            # Load checkpoint if we resume from a previous training.
            if opt.train_from:
                self.logger.info('Loading checkpoint from %s' % opt.train_from)
                checkpoint = torch.load(opt.train_from, map_location=lambda storage, loc: storage)
                self.logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
                vocab = checkpoint['vocab']
            else:
                vocab = torch.load(opt.data + '.vocab.pt')

            # check for code where vocab is saved instead of fields
            # (in the future this will be done in a smarter way)
            if old_style_vocab(vocab):
                fields = load_old_vocab(vocab, opt.model_type, dynamic_dict=opt.copy_attn)
            else:
                fields = vocab

            if len(opt.data_ids) > 1:
                train_shards = []
                for train_id in opt.data_ids:
                    shard_base = "train_" + train_id
                    train_shards.append(shard_base)
                train_iter = MultiSourceInputter.build_dataset_iter_multiple(
                    self.config.get_src_types(), train_shards, fields, opt
                )
            else:
                if opt.data_ids[0] is not None:
                    shard_base = "train_" + opt.data_ids[0]
                else:
                    shard_base = "train"
                train_iter = MultiSourceInputter.build_dataset_iter(
                    self.config.get_src_types(), shard_base, fields, opt
                )

            nb_gpu = len(opt.gpu_ranks)

            if opt.world_size > 1:
                queues = []
                mp = torch.multiprocessing.get_context('spawn')
                semaphore = mp.Semaphore(opt.world_size * opt.queue_size)
                # Create a thread to listen for errors in the child processes.
                error_queue = mp.SimpleQueue()
                error_handler = ErrorHandler(error_queue)
                # Train with multiprocessing.
                procs = []
                for device_id in range(nb_gpu):
                    q = mp.Queue(opt.queue_size)
                    queues += [q]

                    def run(opt, device_id, error_queue, batch_queue, semaphore):
                        """ run process """
                        try:
                            gpu_rank = onmt.utils.distributed.multi_init(opt, device_id)
                            if gpu_rank != opt.gpu_ranks[device_id]:
                                raise AssertionError("An error occurred in Distributed initialization")
                            self.train_single(opt, device_id, batch_queue, semaphore)
                        except KeyboardInterrupt:
                            pass  # killed by parent, do nothing
                        except Exception:
                            # propagate exception to parent process, keeping original traceback
                            import traceback
                            error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))

                    procs.append(mp.Process(target=run, args=(opt, device_id, error_queue, q, semaphore), daemon=True))
                    procs[device_id].start()
                    self.logger.info(" Starting process pid: %d  " % procs[device_id].pid)
                    error_handler.add_child(procs[device_id].pid)
                producer = mp.Process(target=batch_producer,args=(train_iter, queues, semaphore, opt,), daemon=True)
                producer.start()
                error_handler.add_child(producer.pid)

                for p in procs:  p.join()
                producer.terminate()

            elif nb_gpu == 1:  # case 1 GPU only
                self.train_single(opt, 0)
            else:  # case only CPU
                self.train_single(opt, -1)

        # Delete unneeded model checkpoints
        best_step = IOUtils.load(self.model_dir / "best-step.json", IOUtils.Format.json)
        for f in (self.model_dir / "models").glob("ckpt_step_*.pt"):
            if f.name != f"ckpt_step_{best_step}.pt":
                f.unlink()

        # Make model cache invalid
        self.loaded_model_cache = None
        return

    def train_single(self, opt, device_id, batch_queue=None, semaphore=None):
        from roosterize.ml.onmt.MultiSourceInputter import MultiSourceInputter
        from roosterize.ml.onmt.MultiSourceModelBuilder import MultiSourceModelBuilder
        from roosterize.ml.onmt.MultiSourceModelSaver import MultiSourceModelSaver
        from roosterize.ml.onmt.MultiSourceTrainer import MultiSourceTrainer
        from onmt.inputters.inputter import load_old_vocab, old_style_vocab
        from onmt.train_single import configure_process, _tally_parameters, _check_save_model_path
        from onmt.utils.optimizers import Optimizer
        from onmt.utils.parse import ArgumentParser

        configure_process(opt, device_id)
        assert len(opt.accum_count) == len(opt.accum_steps), 'Number of accum_count values must match number of accum_steps'
        # Load checkpoint if we resume from a previous training.
        if opt.train_from:
            self.logger.info('Loading checkpoint from %s' % opt.train_from)
            checkpoint = torch.load(opt.train_from, map_location=lambda storage, loc: storage)
            model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
            ArgumentParser.update_model_opts(model_opt)
            ArgumentParser.validate_model_opts(model_opt)
            self.logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
            vocab = checkpoint['vocab']
        else:
            checkpoint = None
            model_opt = opt
            vocab = torch.load(opt.data + '.vocab.pt')

        # check for code where vocab is saved instead of fields
        # (in the future this will be done in a smarter way)
        if old_style_vocab(vocab):
            fields = load_old_vocab(vocab, opt.model_type, dynamic_dict=opt.copy_attn)
        else:
            fields = vocab

        # Report src and tgt vocab sizes, including for features
        data_keys = [f"src.{src_type}" for src_type in self.config.get_src_types()] + ["tgt"]
        for side in data_keys:
            f = fields[side]
            try:
                f_iter = iter(f)
            except TypeError:
                f_iter = [(side, f)]
            for sn, sf in f_iter:
                if sf.use_vocab:  self.logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

        # Build model
        model = MultiSourceModelBuilder.build_model(self.config.get_src_types(), model_opt, opt, fields, checkpoint)
        n_params, enc, dec = _tally_parameters(model)
        self.logger.info('encoder: %d' % enc)
        self.logger.info('decoder: %d' % dec)
        self.logger.info('* number of parameters: %d' % n_params)
        _check_save_model_path(opt)

        # Build optimizer.
        optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

        # Build model saver
        model_saver = MultiSourceModelSaver.build_model_saver(self.config.get_src_types(), model_opt, opt, model, fields, optim)

        trainer = MultiSourceTrainer.build_trainer(self.config.get_src_types(), opt, device_id, model, fields, optim, model_saver=model_saver)

        if batch_queue is None:
            if len(opt.data_ids) > 1:
                train_shards = []
                for train_id in opt.data_ids:
                    shard_base = "train_" + train_id
                    train_shards.append(shard_base)
                train_iter = MultiSourceInputter.build_dataset_iter_multiple(self.config.get_src_types(), train_shards, fields, opt)
            else:
                if opt.data_ids[0] is not None:
                    shard_base = "train_" + opt.data_ids[0]
                else:
                    shard_base = "train"
                train_iter = MultiSourceInputter.build_dataset_iter(self.config.get_src_types(), shard_base, fields, opt)
        else:
            assert semaphore is not None, "Using batch_queue requires semaphore as well"

            def _train_iter():
                while True:
                    batch = batch_queue.get()
                    semaphore.release()
                    yield batch
                # end while
            # end def

            train_iter = _train_iter()
        # end if

        valid_iter = MultiSourceInputter.build_dataset_iter(self.config.get_src_types(), "valid", fields, opt, is_train=False)

        if len(opt.gpu_ranks):
            self.logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
        else:
            self.logger.info('Starting training on CPU, could be very slow')
        train_steps = opt.train_steps
        if opt.single_pass and train_steps > 0:
            self.logger.warning("Option single_pass is enabled, ignoring train_steps.")
            train_steps = 0

        trainer.train(
            train_iter,
            train_steps,
            save_checkpoint_steps=opt.save_checkpoint_steps,
            valid_iter=valid_iter,
            valid_steps=opt.valid_steps,
        )
        time_begin = trainer.report_manager.start_time
        time_end = time.time()

        if opt.tensorboard:  trainer.report_manager.tensorboard_writer.close()

        # Dump train metrics
        train_history = trainer.report_manager.get_joint_history()
        train_metrics = {
            "time_begin": time_begin,
            "time_end": time_end,
            "time": time_end - time_begin,
            "train_history": train_history,
        }
        IOUtils.dump(self.model_dir/"train-metrics.json", train_metrics, IOUtils.Format.jsonNoSort)

        # Get the best step, depending on the lowest val_xent (cross entropy)
        best_loss = min([th["val_xent"] for th in train_history])
        best_step = [th["step"] for th in train_history if th["val_xent"] == best_loss][-1]  # Take the last if multiple
        IOUtils.dump(self.model_dir/"best-step.json", best_step, IOUtils.Format.json)
        return

    def eval_impl(
            self,
            processed_data_dir: Path,
            beam_search_size: int,
            k: int
    ) -> List[List[Tuple[str, float]]]:
        from roosterize.ml.onmt.MultiSourceTranslator import MultiSourceTranslator
        from onmt.utils.misc import split_corpus
        from onmt.utils.parse import ArgumentParser
        from translate import _get_parser as translate_get_parser

        src_path = processed_data_dir/"src.txt"
        tgt_path = processed_data_dir/"tgt.txt"

        best_step = IOUtils.load(self.model_dir/"best-step.json", IOUtils.Format.json)
        self.logger.info(f"Taking best step at {best_step}")

        candidates_logprobs: List[List[Tuple[List[str], float]]] = list()

        with IOUtils.cd(self.open_nmt_path):
            parser = translate_get_parser()
            opt = parser.parse_args(
                f" -model {self.model_dir}/models/ckpt_step_{best_step}.pt"
                f" -src {src_path}"
                f" -tgt {tgt_path}"
            )
            opt.output = f"{self.model_dir}/last-pred.txt"
            opt.beam_size = beam_search_size
            opt.gpu = 0 if torch.cuda.is_available() else -1
            opt.n_best = k
            opt.block_ngram_repeat = 1
            opt.ignore_when_blocking = ["_"]

            # translate.main
            ArgumentParser.validate_translate_opts(opt)

            # Cached model loading
            if self.loaded_model_cache is None:
                self.loaded_model_cache = MultiSourceTranslator.load_model(self.config.get_src_types(), opt)
            translator = MultiSourceTranslator.build_translator(
                self.config.get_src_types(),
                opt,
                loaded_model=self.loaded_model_cache,
                report_score=False,
            )

            has_target = True
            raw_data_keys = [f"src.{src_type}" for src_type in self.config.get_src_types()] + (["tgt"] if has_target else [])
            raw_data_paths: Dict[str, str] = {
                k: f"{processed_data_dir}/{k}.txt"
                for k in raw_data_keys
            }
            raw_data_shards: Dict[str, list] = {
                k: list(split_corpus(p, opt.shard_size))
                for k, p in raw_data_paths.items()
            }

            # src_shards = split_corpus(opt.src, opt.shard_size)
            # tgt_shards = split_corpus(opt.tgt, opt.shard_size) if opt.tgt is not None else repeat(None)
            # shard_pairs = zip(src_shards, tgt_shards)

            for i in range(len(list(raw_data_shards.values())[0])):
                self.logger.info("Translating shard %d." % i)
                _, _, candidates_logprobs_shard = translator.translate(
                    {k: v[i] for k, v in raw_data_shards.items()},
                    has_target,
                    src_dir=None,
                    batch_size=opt.batch_size,
                    attn_debug=opt.attn_debug
                )
                candidates_logprobs.extend(candidates_logprobs_shard)

        # Reformat candidates
        candidates_logprobs: List[List[Tuple[str, float]]] = [[("".join(c), l) for c, l in cl] for cl in candidates_logprobs]

        return candidates_logprobs
