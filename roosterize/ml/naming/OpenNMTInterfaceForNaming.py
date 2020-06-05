from typing import *

from itertools import repeat
import numpy as np
from pathlib import Path
from recordclass import RecordClass
import sys
import time
import torch
import torch.nn
import torch.nn.utils

from seutil import LoggingUtils, IOUtils

from roosterize.data.Definition import Definition
from roosterize.data.ModelSpec import ModelSpec
from roosterize.data.Lemma import Lemma
from roosterize.data.LemmaBackendSexpTransformers import LemmaBackendSexpTransformers
from roosterize.data.LemmaForeendSexpTransformers import LemmaForeendSexpTransformers
from roosterize.Macros import Macros
from roosterize.ml.naming.NamingModelBase import NamingModelBase
from roosterize.ml.naming.SubTokenizer import SubTokenizer
from roosterize.Utils import Utils


class ONMTILNConsts:

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

    OUTPUT_CHAR = "c"
    OUTPUT_SUBTOKEN = "st"


class ONMTILNConfig(RecordClass):
    # Encoder/Decoder
    encoder: str = ONMTILNConsts.ENCODER_BRNN
    decoder: str = ONMTILNConsts.DECODER_RNN

    # Dimensions
    dim_encoder_hidden: int = 500
    dim_decoder_hidden: int = 500
    dim_embed: int = 500

    # Input/output
    input: str = ONMTILNConsts.INPUT_SEQ
    input_max: int = 3000  # Fixed
    output: str = ONMTILNConsts.OUTPUT_SUBTOKEN

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
        s = f"{self.encoder}-{self.input}-{self.decoder}-{self.output}-de_{self.dim_embed}-deh_{self.dim_encoder_hidden}-ddh_{self.dim_decoder_hidden}-rnl_{self.rnn_num_layers}-do_{self.dropout}-vift_{self.vocab_input_frequency_threshold}"
        if self.pretrain is not None and not modulo_pretrain:  s += f"-pt_{self.pretrain}"
        if self.use_attn:  s += "+attn"
        if self.use_copy:  s += "+copy"
        return s

    def repOk(self):
        # Check range
        if self.encoder not in [ONMTILNConsts.ENCODER_RNN, ONMTILNConsts.ENCODER_BRNN, ONMTILNConsts.ENCODER_MEAN, ONMTILNConsts.ENCODER_TRANSFORMER, ONMTILNConsts.ENCODER_CNN]:
            return False
        if self.decoder not in [ONMTILNConsts.DECODER_RNN, ONMTILNConsts.DECODER_TRANSFORMER, ONMTILNConsts.DECODER_CNN]:
            return False
        if self.input not in [
            ONMTILNConsts.INPUT_SEQ,
            ONMTILNConsts.INPUT_BSEXP_ORIG, ONMTILNConsts.INPUT_BSEXP_NOPARA_ORIG,
            ONMTILNConsts.INPUT_BSEXP_L1, ONMTILNConsts.INPUT_BSEXP_NOPARA_L1,
            ONMTILNConsts.INPUT_BSEXP_L2, ONMTILNConsts.INPUT_BSEXP_NOPARA_L2,
            ONMTILNConsts.INPUT_FSEXP_L0, ONMTILNConsts.INPUT_FSEXP_NOPARA_L0,
            ONMTILNConsts.INPUT_FSEXP_L1, ONMTILNConsts.INPUT_FSEXP_NOPARA_L1,
        ]:
            return False
        if self.output not in [ONMTILNConsts.OUTPUT_SUBTOKEN, ONMTILNConsts.OUTPUT_CHAR]:
            return False
        if self.dim_encoder_hidden <= 0 or self.dim_decoder_hidden <= 0 or self.dim_embed <= 0:
            return False
        if self.rnn_num_layers <=0:
            return False
        if not 0 <= self.dropout <= 1:
            return False

        # Because of the way ONMT saves ckpts
        if self.ckpt_keep_max < self.early_stopping_threshold:
            return False

        # Encoder and decoder must have equal hidden state
        if self.dim_encoder_hidden != self.dim_decoder_hidden:
            return False

        # Fixed configurations
        if self.early_stopping_threshold != 3 or self.learning_rate != 1e-3 or self.ckpt_keep_max != 3 or self.max_grad_norm != 5 or self.beam_search_max_len_factor != 1.5 or self.input_max != 3000:
            return False

        return True

    def adjust_batch_size(self):
        batch_size = 128
        if self.input in [ONMTILNConsts.INPUT_BSEXP_ORIG, ONMTILNConsts.INPUT_BSEXP_NOPARA_ORIG, ONMTILNConsts.INPUT_FSEXP_L0, ONMTILNConsts.INPUT_FSEXP_NOPARA_L0]:  batch_size /= 16
        if self.input in [ONMTILNConsts.INPUT_BSEXP_L1, ONMTILNConsts.INPUT_BSEXP_NOPARA_L1, ONMTILNConsts.INPUT_BSEXP_L2, ONMTILNConsts.INPUT_BSEXP_NOPARA_L2, ONMTILNConsts.INPUT_FSEXP_L1, ONMTILNConsts.INPUT_FSEXP_NOPARA_L1]:  batch_size /= 2
        if self.rnn_num_layers > 2:  batch_size *= 2 / self.rnn_num_layers
        if self.dim_embed > 500:  batch_size *= 500 / self.dim_embed
        if self.dim_encoder_hidden > 500:  batch_size *= 500 / self.dim_encoder_hidden
        if self.dim_decoder_hidden > 500:  batch_size *= 500 / self.dim_decoder_hidden
        self.batch_size = max(int(np.ceil(batch_size)), 1)
        return


class OpenNMTInterfaceForNaming(NamingModelBase[ONMTILNConfig]):

    logger = LoggingUtils.get_logger(__name__)

    def __init__(self, model_spec: ModelSpec):
        super().__init__(model_spec, ONMTILNConfig)

        self.open_nmt_path = Macros.project_dir

        # Hack OpenNMT logging
        Utils.modify_and_import("onmt.utils.logging", None,
            lambda x: f"from seutil import LoggingUtils\n"
                      f"logger = LoggingUtils.get_logger(__name__)\n"
                      f"def init_logger(log_file=None, log_file_level=None):\n"
                      f"  return logger\n"
        )
        module_names = list(sys.modules.keys())
        for module_name in module_names:
            if (module_name.startswith("onmt.") or module_name == "onmt") and module_name != "onmt.utils.logging":  sys.modules.pop(module_name)
        # end for

        if not torch.cuda.is_available():  self.logger.warning("Cuda is not available")
        self.device_tag = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_tag)

        self.data_cache: dict = dict()  # For caching some data during data processing
        return

    def get_input(self, lemma: Lemma, docs_sub_tokenizers: Optional[Dict[str, SubTokenizer]]) -> List[str]:
        input: List[str]
        if self.config.input == ONMTILNConsts.INPUT_SEQ:
            input = [t.content for t in lemma.statement]
        else:
            if self.config.input in [
                ONMTILNConsts.INPUT_BSEXP_ORIG,
                ONMTILNConsts.INPUT_BSEXP_L1,
                ONMTILNConsts.INPUT_BSEXP_L2,
                ONMTILNConsts.INPUT_FSEXP_L0,
                ONMTILNConsts.INPUT_FSEXP_L1,
            ]:
                use_parathesis = True
            else:
                use_parathesis = False
            # end if

            if self.config.input.startswith("bsexp"):
                sexp = lemma.backend_sexp

                if self.config.input in [ONMTILNConsts.INPUT_BSEXP_L1, ONMTILNConsts.INPUT_BSEXP_NOPARA_L1]:
                    sexp = LemmaBackendSexpTransformers.transform(LemmaBackendSexpTransformers.LEVEL_1, sexp)
                elif self.config.input in [ONMTILNConsts.INPUT_BSEXP_L2, ONMTILNConsts.INPUT_BSEXP_NOPARA_L2]:
                    sexp = LemmaBackendSexpTransformers.transform(LemmaBackendSexpTransformers.LEVEL_2, sexp)
                # end if
            else:
                sexp = lemma.ast_sexp

                if self.config.input in [ONMTILNConsts.INPUT_FSEXP_L0, ONMTILNConsts.INPUT_FSEXP_NOPARA_L0]:
                    sexp = LemmaForeendSexpTransformers.transform(LemmaForeendSexpTransformers.LEVEL_0, sexp)
                elif self.config.input in [ONMTILNConsts.INPUT_FSEXP_L1, ONMTILNConsts.INPUT_FSEXP_NOPARA_L1]:
                    sexp = LemmaForeendSexpTransformers.transform(LemmaForeendSexpTransformers.LEVEL_1, sexp)
                # end if
            # end if

            input = sexp.forward_depth_first_sequence(use_parathesis=use_parathesis)
        # end if

        # Always sub tokenize input
        sub_tokenizer = docs_sub_tokenizers[lemma.data_index]
        input = [st for t in input for st in (sub_tokenizer.sub_tokenize(t) if SubTokenizer.can_tokenize(t) else [t])]

        return input[:self.config.input_max]

    def get_output(self, lemma: Lemma, docs_sub_tokenizers: Optional[Dict[str, SubTokenizer]]) -> List[str]:
        if self.config.output == ONMTILNConsts.OUTPUT_CHAR:
            return [c for c in lemma.name]
        elif self.config.output == ONMTILNConsts.OUTPUT_SUBTOKEN:
            sub_tokenizer = docs_sub_tokenizers[lemma.data_index]
            return sub_tokenizer.sub_tokenize(lemma.name)
        else:
            raise ValueError
        # end if

    def process_data_impl(self,
            data_dir: Path,
            output_processed_data_dir: Path,
    ) -> NoReturn:
        lemmas: List[Lemma] = IOUtils.dejsonfy(IOUtils.load(data_dir/"lemmas.json", IOUtils.Format.json), List[Lemma])
        definitions: List[Definition] = IOUtils.dejsonfy(IOUtils.load(data_dir/"definitions.json", IOUtils.Format.json), List[Definition])

        docs_sub_tokenizers = SubTokenizer.get_docs_sub_tokenizers(lemmas, definitions)

        # Put data in serialized files
        IOUtils.dump(output_processed_data_dir/f"src.txt",
            "".join([" ".join(self.get_input(lemma, docs_sub_tokenizers)) + "\n" for lemma in lemmas]),
            IOUtils.Format.txt)
        IOUtils.dump(output_processed_data_dir/f"tgt.txt",
            "".join([" ".join(self.get_output(lemma, docs_sub_tokenizers)) + "\n" for lemma in lemmas]),
            IOUtils.Format.txt)
        return

    def preprocess(self,
            train_processed_data_dir: Path,
            val_processed_data_dir: Path,
            output_model_dir: Path
    ) -> NoReturn:
        # Call OpenNMT preprocess
        with IOUtils.cd(self.open_nmt_path):
            from preprocess import _get_parser as preprocess_get_parser
            from preprocess import main as preprocess_main
            parser = preprocess_get_parser()
            opt = parser.parse_args(
                f" -train_src {train_processed_data_dir}/src.txt"
                f" -train_tgt {train_processed_data_dir}/tgt.txt"
                f" -valid_src {val_processed_data_dir}/src.txt"
                f" -valid_tgt {val_processed_data_dir}/tgt.txt"
                f" -save_data {output_model_dir}/processed-data"
            )
            opt.src_seq_length = self.config.input_max
            opt.src_words_min_frequency = self.config.vocab_input_frequency_threshold
            if self.config.use_copy:  opt.dynamic_dict = True
            preprocess_main(opt)
        # end with
        return

    def train_impl(self,
            train_processed_data_dir: Path,
            val_processed_data_dir: Path,
            output_model_dir: Path,
    ) -> NoReturn:
        from train import _get_parser as train_get_parser
        from train import ErrorHandler, batch_producer
        from onmt.inputters.inputter import old_style_vocab, load_old_vocab, build_dataset_iter, build_dataset_iter_multiple
        import onmt.utils.distributed
        from onmt.utils.parse import ArgumentParser

        with IOUtils.cd(self.open_nmt_path):
            parser = train_get_parser()
            opt = parser.parse_args(
                f" -data {output_model_dir}/processed-data"
                f" -save_model {output_model_dir}/models/ckpt"
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
            if self.config.use_attn:
                opt.global_attention = "general"
            else:
                opt.global_attention = "none"
            # end if
            if self.config.use_copy:
                opt.copy_attn = True
                opt.copy_attn_type = "general"
            # end if

            # train.main, one gpu case
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
            # end if

            # check for code where vocab is saved instead of fields
            # (in the future this will be done in a smarter way)
            if old_style_vocab(vocab):
                fields = load_old_vocab(vocab, opt.model_type, dynamic_dict=opt.copy_attn)
            else:
                fields = vocab
            # end if

            if len(opt.data_ids) > 1:
                train_shards = []
                for train_id in opt.data_ids:
                    shard_base = "train_" + train_id
                    train_shards.append(shard_base)
                # end for
                train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
            else:
                if opt.data_ids[0] is not None:
                    shard_base = "train_" + opt.data_ids[0]
                else:
                    shard_base = "train"
                # end if
                train_iter = build_dataset_iter(shard_base, fields, opt)
            # end if

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
                        # end try
                    # end def

                    procs.append(mp.Process(target=run, args=(opt, device_id, error_queue, q, semaphore), daemon=True))
                    procs[device_id].start()
                    self.logger.info(" Starting process pid: %d  " % procs[device_id].pid)
                    error_handler.add_child(procs[device_id].pid)
                # end for
                producer = mp.Process(target=batch_producer,args=(train_iter, queues, semaphore, opt,), daemon=True)
                producer.start()
                error_handler.add_child(producer.pid)

                for p in procs:  p.join()
                producer.terminate()

            elif nb_gpu == 1:  # case 1 GPU only
                self.train_single(output_model_dir, opt, 0)
            else:  # case only CPU
                self.train_single(output_model_dir, opt, -1)
            # end if
        # end with
        return

    def train_single(self, output_model_dir: Path, opt, device_id, batch_queue=None, semaphore=None):
        from roosterize.ml.onmt.CustomTrainer import CustomTrainer
        from onmt.inputters.inputter import build_dataset_iter, load_old_vocab, old_style_vocab, build_dataset_iter_multiple
        from onmt.model_builder import build_model
        from onmt.train_single import configure_process, _tally_parameters, _check_save_model_path
        from onmt.models import build_model_saver
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
        # end if

        # check for code where vocab is saved instead of fields
        # (in the future this will be done in a smarter way)
        if old_style_vocab(vocab):
            fields = load_old_vocab(vocab, opt.model_type, dynamic_dict=opt.copy_attn)
        else:
            fields = vocab
        # end if

        # Report src and tgt vocab sizes, including for features
        for side in ['src', 'tgt']:
            f = fields[side]
            try:
                f_iter = iter(f)
            except TypeError:
                f_iter = [(side, f)]
            # end try
            for sn, sf in f_iter:
                if sf.use_vocab:  self.logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))
            # end for

        # Build model
        model = build_model(model_opt, opt, fields, checkpoint)
        n_params, enc, dec = _tally_parameters(model)
        self.logger.info('encoder: %d' % enc)
        self.logger.info('decoder: %d' % dec)
        self.logger.info('* number of parameters: %d' % n_params)
        _check_save_model_path(opt)

        # Build optimizer.
        optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

        # Build model saver
        model_saver = build_model_saver(model_opt, opt, model, fields, optim)

        trainer = CustomTrainer.build_trainer(opt, device_id, model, fields, optim, model_saver=model_saver)

        if batch_queue is None:
            if len(opt.data_ids) > 1:
                train_shards = []
                for train_id in opt.data_ids:
                    shard_base = "train_" + train_id
                    train_shards.append(shard_base)
                # end for
                train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
            else:
                if opt.data_ids[0] is not None:
                    shard_base = "train_" + opt.data_ids[0]
                else:
                    shard_base = "train"
                # end if
                train_iter = build_dataset_iter(shard_base, fields, opt)
            # end if
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

        valid_iter = build_dataset_iter("valid", fields, opt, is_train=False)

        if len(opt.gpu_ranks):
            self.logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
        else:
            self.logger.info('Starting training on CPU, could be very slow')
        # end if
        train_steps = opt.train_steps
        if opt.single_pass and train_steps > 0:
            self.logger.warning("Option single_pass is enabled, ignoring train_steps.")
            train_steps = 0
        # end if

        trainer.train(
            train_iter,
            train_steps,
            save_checkpoint_steps=opt.save_checkpoint_steps,
            valid_iter=valid_iter,
            valid_steps=opt.valid_steps)
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
        IOUtils.dump(output_model_dir/"train-metrics.json", train_metrics, IOUtils.Format.jsonNoSort)

        # Get the best step, depending on the lowest val_xent (cross entropy)
        best_loss = min([th["val_xent"] for th in train_history])
        best_step = [th["step"] for th in train_history if th["val_xent"] == best_loss][-1]  # Take the last if multiple
        IOUtils.dump(output_model_dir/"best-step.json", best_step, IOUtils.Format.json)
        return

    def eval_impl(self,
            processed_data_dir: Path,
            model_dir: Path,
            beam_search_size: int,
            k: int
    ) -> List[List[Tuple[str, float]]]:
        from roosterize.ml.onmt.CustomTranslator import CustomTranslator
        from onmt.utils.misc import split_corpus
        from onmt.utils.parse import ArgumentParser
        from translate import _get_parser as translate_get_parser

        src_path = processed_data_dir/"src.txt"
        tgt_path = processed_data_dir/"tgt.txt"

        best_step = IOUtils.load(model_dir/"best-step.json", IOUtils.Format.json)
        self.logger.info(f"Taking best step at {best_step}")

        candidates_logprobs: List[List[Tuple[List[str], float]]] = list()

        with IOUtils.cd(self.open_nmt_path):
            parser = translate_get_parser()
            opt = parser.parse_args(
                f" -model {model_dir}/models/ckpt_step_{best_step}.pt"
                f" -src {src_path}"
                f" -tgt {tgt_path}"
            )
            opt.output = f"{model_dir}/last-pred.txt"
            opt.beam_size = beam_search_size
            opt.gpu = 0 if torch.cuda.is_available() else -1
            opt.n_best = k
            opt.block_ngram_repeat = 1
            opt.ignore_when_blocking = ["_"]

            # translate.main
            ArgumentParser.validate_translate_opts(opt)

            translator = CustomTranslator.build_translator(opt, report_score=False)
            src_shards = split_corpus(opt.src, opt.shard_size)
            tgt_shards = split_corpus(opt.tgt, opt.shard_size) if opt.tgt is not None else repeat(None)
            shard_pairs = zip(src_shards, tgt_shards)

            for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
                self.logger.info("Translating shard %d." % i)
                _, _, candidates_logprobs_shard = translator.translate(
                    src=src_shard,
                    tgt=tgt_shard,
                    src_dir=opt.src_dir,
                    batch_size=opt.batch_size,
                    attn_debug=opt.attn_debug
                )
                candidates_logprobs.extend(candidates_logprobs_shard)
            # end for
        # end with

        # Reformat candidates
        candidates_logprobs: List[List[Tuple[str, float]]] = [[("".join(c), l) for c, l in cl] for cl in candidates_logprobs]

        return candidates_logprobs
