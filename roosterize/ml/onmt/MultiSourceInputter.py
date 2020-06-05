from typing import *

import codecs
import collections
import glob
from itertools import chain, cycle
import math
from onmt.inputters.text_dataset import text_fields, TextMultiField
import os
import torch
import torchtext.data
from torchtext.data import Field, RawField
from torchtext.data.utils import RandomShuffler
from torchtext.vocab import Vocab

from seutil import LoggingUtils


# monkey-patch to make torchtext Vocab's pickleable
def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = collections.defaultdict(lambda: 0, self.stoi)


Vocab.__getstate__ = _getstate
Vocab.__setstate__ = _setstate


class MultiSourceInputter:
    """
    Modified version of onmt.inputters.inputter, to support multi source.
    """

    logger = LoggingUtils.get_logger(__name__)

    @classmethod
    def make_src(cls, data, vocab):
        src_size = max([t.size(0) for t in data])
        src_vocab_size = max([t.max() for t in data]) + 1
        alignment = torch.zeros(src_size, len(data), src_vocab_size)
        for i, sent in enumerate(data):
            for j, t in enumerate(sent):
                alignment[j, i, t] = 1
        return alignment

    @classmethod
    def make_tgt(cls, data, vocab):
        tgt_size = max([t.size(0) for t in data])
        alignment = torch.zeros(tgt_size, len(data)).long()
        for i, sent in enumerate(data):
            alignment[:sent.size(0), i] = sent
        return alignment

    @classmethod
    def get_fields(cls,
            src_types: List[str],
            n_src_feats: int,
            n_tgt_feats: int,
            pad: str = '<blank>',
            bos: str = '<s>',
            eos: str = '</s>',
            dynamic_dict: bool = False,
            src_truncate: Optional[int] = None,
            tgt_truncate: Optional[int] = None,
    ) -> Dict[str, Union[Field, TextMultiField]]:
        """
        Args:
            src_data_type: type of the source input. Options are [text|img|audio].
            n_src_feats (int): the number of source features (not counting tokens)
                to create a :class:`torchtext.data.Field` for. (If
                ``src_data_type=="text"``, these fields are stored together
                as a ``TextMultiField``).
            n_tgt_feats (int): See above.
            pad (str): Special pad symbol. Used on src and tgt side.
            bos (str): Special beginning of sequence symbol. Only relevant
                for tgt.
            eos (str): Special end of sequence symbol. Only relevant
                for tgt.
            dynamic_dict (bool): Whether or not to include source map and
                alignment fields.
            src_truncate: Cut off src sequences beyond this (passed to
                ``src_data_type``'s data reader - see there for more details).
            tgt_truncate: Cut off tgt sequences beyond this (passed to
                :class:`TextDataReader` - see there for more details).

        Returns:
            A dict mapping names to fields. These names need to match
            the dataset example attributes.
        """
        # PN: here I removed data types other than "text", to make things easier
        # assert src_data_type == 'text', "Only text is supported in multi-source"
        # assert not dynamic_dict or src_data_type == 'text', 'it is not possible to use dynamic_dict with non-text input'
        fields: Dict = {}

        for src_type in src_types:
            src_field_kwargs = {"n_feats": n_src_feats,
                "include_lengths": True,
                "pad": pad, "bos": None, "eos": None,
                "truncate": src_truncate,
                "base_name": "src"}
            fields[f"src.{src_type}"] = text_fields(**src_field_kwargs)
        # end for

        tgt_field_kwargs = {"n_feats": n_tgt_feats,
            "include_lengths": False,
            "pad": pad, "bos": bos, "eos": eos,
            "truncate": tgt_truncate,
            "base_name": "tgt"}
        fields["tgt"] = text_fields(**tgt_field_kwargs)

        indices = Field(use_vocab=False, dtype=torch.long, sequential=False)
        fields["indices"] = indices

        if dynamic_dict:
            for src_type in src_types:
                src_map = Field(
                    use_vocab=False, dtype=torch.float,
                    postprocessing=cls.make_src, sequential=False)
                fields[f"src_map.{src_type}"] = src_map
            # end for

            src_ex_vocab = RawField()
            fields["src_ex_vocab"] = src_ex_vocab

            align = Field(
                use_vocab=False, dtype=torch.long,
                postprocessing=cls.make_tgt, sequential=False)
            fields["alignment"] = align
        # end if

        return fields

    # TODO load_old_vocab
    # TODO _old_style_vocab
    # TODO _old_style_nesting
    # TODO _old_style_field_list
    # TODO old_style_vocab

    @classmethod
    def filter_example(cls,
            ex,
            src_types: List[str],
            use_src_len=True,
            use_tgt_len=True,
            min_src_len=1,
            max_src_len=float('inf'),
            min_tgt_len=1,
            max_tgt_len=float('inf')
    ):
        """Return whether an example is an acceptable length.

        If used with a dataset as ``filter_pred``, use :func:`partial()`
        for all keyword arguments.

        Args:
            ex (torchtext.data.Example): An object with a ``src`` and ``tgt``
                property.
            use_src_len (bool): Filter based on the length of ``ex.src``.
            use_tgt_len (bool): Similar to above.
            min_src_len (int): A non-negative minimally acceptable length
                (examples of exactly this length will be included).
            min_tgt_len (int): Similar to above.
            max_src_len (int or float): A non-negative (possibly infinite)
                maximally acceptable length (examples of exactly this length
                will be included).
            max_tgt_len (int or float): Similar to above.
        """

        src_lens = [len(ex.__getattribute__(f"src.{src_type}")[0]) for src_type in src_types]
        tgt_len = len(ex.tgt[0])
        return all([(not use_src_len or min_src_len <= src_len <= max_src_len) for src_len in src_lens]) and \
               (not use_tgt_len or min_tgt_len <= tgt_len <= max_tgt_len)


    @classmethod
    def pad_vocab_to_multiple(cls,
            vocab: Vocab,
            multiple: int,
    ) -> Vocab:
        # PN: original name was _pad_vocab_to_multiple
        vocab_size = len(vocab)
        if vocab_size % multiple == 0:
            return vocab
        target_size = int(math.ceil(vocab_size / multiple)) * multiple
        padding_tokens = [
            "averyunlikelytoken%d" % i for i in range(target_size - vocab_size)]
        vocab.extend(Vocab(Counter(), specials=padding_tokens))
        return vocab

    @classmethod
    def build_field_vocab(cls,
            field: Field,
            counter: Counter,
            size_multiple: int = 1,
            **kwargs
    ) -> NoReturn:
        # PN: original name was _build_field_vocab
        # this is basically copy-pasted from torchtext.
        all_specials = [
            field.unk_token, field.pad_token, field.init_token, field.eos_token
        ]
        specials = [tok for tok in all_specials if tok is not None]
        field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)
        if size_multiple > 1:  cls.pad_vocab_to_multiple(field.vocab, size_multiple)
        return

    @classmethod
    def load_vocab(cls,
            vocab_path: str,
            name: str,
            counters: Dict[Any, Counter],
            min_freq: int,
    ) -> Tuple[List[str], int]:
        # PN: original name was _load_vocab
        # counters changes in place
        vocab: List[str] = cls.read_vocab_file(vocab_path, name)
        vocab_size: int = len(vocab)
        cls.logger.info('Loaded %s vocab has %d tokens.' % (name, vocab_size))
        for i, token in enumerate(vocab):
            # keep the order of tokens specified in the vocab file by
            # adding them to the counter with decreasing counting values
            counters[name][token] = vocab_size - i + min_freq
        # end for
        return vocab, vocab_size

    @classmethod
    def build_fv_from_multifield(cls,
            multifield,
            counters,
            build_fv_args,
            size_multiple=1
    ):
        # PN: original name was _build_fv_from_multifield
        for name, field in multifield:
            cls.build_field_vocab(
                field,
                counters[name],
                size_multiple=size_multiple,
                **build_fv_args[name])
            cls.logger.info(" * %s vocab size: %d." % (name, len(field.vocab)))

    @classmethod
    def build_fields_vocab(cls,
            src_types: List[str],
            fields,
            counters,
            share_vocab,
            vocab_size_multiple,
            src_vocab_size,
            src_words_min_frequency,
            tgt_vocab_size,
            tgt_words_min_frequency
    ):
        # PN: original name was _build_fields_vocab

        # TODO share_vocab is not supported for now
        assert not share_vocab

        build_fv_args = collections.defaultdict(dict)
        for src_type in src_types:
            build_fv_args[f"src.{src_type}"] = dict(max_size=src_vocab_size, min_freq=src_words_min_frequency)
        # end for
        build_fv_args["tgt"] = dict(max_size=tgt_vocab_size, min_freq=tgt_words_min_frequency)
        tgt_multifield = fields["tgt"]
        cls.build_fv_from_multifield(
            tgt_multifield,
            counters,
            build_fv_args,
            size_multiple=vocab_size_multiple if not share_vocab else 1)

        for src_type in src_types:
            src_multifield = fields[f"src.{src_type}"]
            cls.build_fv_from_multifield(
                src_multifield,
                counters,
                build_fv_args,
                size_multiple=vocab_size_multiple if not share_vocab else 1)
        # end for

        # TODO share_vocab is not supported for now
        # if share_vocab:
        #     # `tgt_vocab_size` is ignored when sharing vocabularies
        #     cls.logger.info(" * merging src and tgt vocab...")
        #     src_field = src_multifield.base_field
        #     tgt_field = tgt_multifield.base_field
        #     cls.merge_field_vocabs(
        #         src_field, tgt_field, vocab_size=src_vocab_size,
        #         min_freq=src_words_min_frequency,
        #         vocab_size_multiple=vocab_size_multiple)
        #     cls.logger.info(" * merged vocab size: %d." % len(src_field.vocab))

        return fields

    # TODO build_vocab

    @classmethod
    def merge_field_vocabs(cls,
            src_field,
            tgt_field,
            vocab_size,
            min_freq,
            vocab_size_multiple
    ) -> NoReturn:
        # PN: original name was merge_field_vocabs
        # in the long run, shouldn't it be possible to do this by calling
        # build_vocab with both the src and tgt data?
        specials = [tgt_field.unk_token, tgt_field.pad_token,
            tgt_field.init_token, tgt_field.eos_token]
        merged = sum(
            [src_field.vocab.freqs, tgt_field.vocab.freqs], Counter()
        )
        merged_vocab = Vocab(
            merged, specials=specials,
            max_size=vocab_size, min_freq=min_freq
        )
        if vocab_size_multiple > 1:
            cls.pad_vocab_to_multiple(merged_vocab, vocab_size_multiple)
        src_field.vocab = merged_vocab
        tgt_field.vocab = merged_vocab
        assert len(src_field.vocab) == len(tgt_field.vocab)
        return

    @classmethod
    def read_vocab_file(cls, vocab_path: str, tag: str) -> List[str]:
        """Loads a vocabulary from the given path.

        Args:
            vocab_path (str): Path to utf-8 text file containing vocabulary.
                Each token should be on a line by itself. Tokens must not
                contain whitespace (else only before the whitespace
                is considered).
            tag (str): Used for logging which vocab is being read.
        """
        # PN: original name was _read_vocab_file

        cls.logger.info("Loading {} vocabulary from {}".format(tag, vocab_path))

        if not os.path.exists(vocab_path):
            raise RuntimeError("{} vocabulary not found at {}".format(tag, vocab_path))
        else:
            with codecs.open(vocab_path, 'r', 'utf-8') as f:
                return [line.strip().split()[0] for line in f if line.strip()]

    @classmethod
    def batch_iter(cls, data, batch_size, batch_size_fn=None, batch_size_multiple=1):
        """Yield elements from data in chunks of batch_size, where each chunk size
        is a multiple of batch_size_multiple.

        This is an extended version of torchtext.data.batch.
        """
        if batch_size_fn is None:
            def batch_size_fn(new, count, sofar):
                return count
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
            if size_so_far >= batch_size:
                overflowed = 0
                if size_so_far > batch_size:
                    overflowed += 1
                if batch_size_multiple > 1:
                    overflowed += (
                            (len(minibatch) - overflowed) % batch_size_multiple)
                if overflowed == 0:
                    yield minibatch
                    minibatch, size_so_far = [], 0
                else:
                    if overflowed == len(minibatch):
                        cls.logger.warning(
                            "An example was ignored, more tokens"
                            " than allowed by tokens batch_size")
                    else:
                        yield minibatch[:-overflowed]
                        minibatch = minibatch[-overflowed:]
                        size_so_far = 0
                        for i, ex in enumerate(minibatch):
                            size_so_far = batch_size_fn(ex, i + 1, size_so_far)
        if minibatch:
            yield minibatch

    @classmethod
    def pool(cls, data, batch_size, batch_size_fn, batch_size_multiple,
            sort_key, random_shuffler, pool_factor):
        # PN: original name was _pool
        for p in torchtext.data.batch(
                data, batch_size * pool_factor,
                batch_size_fn=batch_size_fn):
            p_batch = list(cls.batch_iter(
                sorted(p, key=sort_key),
                batch_size,
                batch_size_fn=batch_size_fn,
                batch_size_multiple=batch_size_multiple))
            for b in random_shuffler(p_batch):
                yield b

    @classmethod
    def text_sort_key(cls, ex, src_types):
        t = tuple([len(getattr(ex, f"src.{src_type}")[0]) for src_type in src_types] + ([len(getattr(ex, "tgt")[0])] if hasattr(ex, "tgt") else []))
        return t

    class OrderedIterator(torchtext.data.Iterator):

        def __init__(self,
                src_types,
                dataset,
                batch_size,
                pool_factor=1,
                batch_size_multiple=1,
                yield_raw_example=False,
                **kwargs):
            # Fix dataset.sort_key
            dataset.sort_key = lambda ex: MultiSourceInputter.text_sort_key(ex, src_types)

            super().__init__(dataset, batch_size, **kwargs)
            self.batch_size_multiple = batch_size_multiple
            self.yield_raw_example = yield_raw_example
            self.dataset = dataset
            self.pool_factor = pool_factor

        def create_batches(self):
            if self.train:
                if self.yield_raw_example:
                    self.batches = MultiSourceInputter.batch_iter(
                        self.data(),
                        1,
                        batch_size_fn=None,
                        batch_size_multiple=1)
                else:
                    self.batches = MultiSourceInputter.pool(
                        self.data(),
                        self.batch_size,
                        self.batch_size_fn,
                        self.batch_size_multiple,
                        self.sort_key,
                        self.random_shuffler,
                        self.pool_factor)
            else:
                self.batches = []
                for b in MultiSourceInputter.batch_iter(
                        self.data(),
                        self.batch_size,
                        batch_size_fn=self.batch_size_fn,
                        batch_size_multiple=self.batch_size_multiple):
                    self.batches.append(sorted(b, key=self.sort_key))

        def __iter__(self):
            """
            Extended version of the definition in torchtext.data.Iterator.
            Added yield_raw_example behaviour to yield a torchtext.data.Example
            instead of a torchtext.data.Batch object.
            """
            while True:
                self.init_epoch()
                for idx, minibatch in enumerate(self.batches):
                    # fast-forward if loaded from state
                    if self._iterations_this_epoch > idx:
                        continue
                    self.iterations += 1
                    self._iterations_this_epoch += 1
                    if self.sort_within_batch:
                        # NOTE: `rnn.pack_padded_sequence` requires that a
                        # minibatch be sorted by decreasing order, which
                        #  requires reversing relative to typical sort keys
                        if self.sort:
                            minibatch.reverse()
                        else:
                            minibatch.sort(key=self.sort_key, reverse=True)
                    if self.yield_raw_example:
                        yield minibatch[0]
                    else:
                        yield torchtext.data.Batch(
                            minibatch,
                            self.dataset,
                            self.device)
                if not self.repeat:
                    return

    class MultipleDatasetIterator(object):
        """
        This takes a list of iterable objects (DatasetLazyIter) and their
        respective weights, and yields a batch in the wanted proportions.
        """

        def __init__(self,
                src_types,
                train_shards,
                fields,
                device,
                opt):
            self.index = -1
            self.iterables = []
            for shard in train_shards:
                self.iterables.append(
                    MultiSourceInputter.build_dataset_iter(src_types, shard, fields, opt, multi=True))
            self.init_iterators = True
            self.weights = opt.data_weights
            self.batch_size = opt.batch_size
            self.batch_size_fn = MultiSourceInputter.max_tok_len \
                if opt.batch_type == "tokens" else None
            self.batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1
            self.device = device
            # Temporarily load one shard to retrieve sort_key for data_type
            temp_dataset = torch.load(self.iterables[0]._paths[0])
            self.sort_key = temp_dataset.sort_key
            self.random_shuffler = RandomShuffler()
            self.pool_factor = opt.pool_factor
            del temp_dataset

        def _iter_datasets(self):
            if self.init_iterators:
                self.iterators = [iter(iterable) for iterable in self.iterables]
                self.init_iterators = False
            for weight in self.weights:
                self.index = (self.index + 1) % len(self.iterators)
                for i in range(weight):
                    yield self.iterators[self.index]

        def _iter_examples(self):
            for iterator in cycle(self._iter_datasets()):
                yield next(iterator)

        def __iter__(self):
            while True:
                for minibatch in MultiSourceInputter.pool(
                        self._iter_examples(),
                        self.batch_size,
                        self.batch_size_fn,
                        self.batch_size_multiple,
                        self.sort_key,
                        self.random_shuffler,
                        self.pool_factor):
                    minibatch = sorted(minibatch, key=self.sort_key, reverse=True)
                    yield torchtext.data.Batch(minibatch,
                        self.iterables[0].dataset,
                        self.device)

    class DatasetLazyIter(object):
        """Yield data from sharded dataset files.

        Args:
            dataset_paths: a list containing the locations of dataset files.
            fields (dict[str, Field]): fields dict for the
                datasets.
            batch_size (int): batch size.
            batch_size_fn: custom batch process function.
            device: See :class:`OrderedIterator` ``device``.
            is_train (bool): train or valid?
        """

        def __init__(self, src_types, dataset_paths, fields, batch_size, batch_size_fn,
                batch_size_multiple, device, is_train, pool_factor,
                repeat=True, num_batches_multiple=1, yield_raw_example=False):
            self.src_types = src_types
            self._paths = dataset_paths
            self.fields = fields
            self.batch_size = batch_size
            self.batch_size_fn = batch_size_fn
            self.batch_size_multiple = batch_size_multiple
            self.device = device
            self.is_train = is_train
            self.repeat = repeat
            self.num_batches_multiple = num_batches_multiple
            self.yield_raw_example = yield_raw_example
            self.pool_factor = pool_factor

        def _iter_dataset(self, path):
            MultiSourceInputter.logger.info('Loading dataset from %s' % path)
            cur_dataset = torch.load(path)
            MultiSourceInputter.logger.info('number of examples: %d' % len(cur_dataset))
            cur_dataset.fields = self.fields
            cur_iter = MultiSourceInputter.OrderedIterator(
                src_types=self.src_types,
                dataset=cur_dataset,
                batch_size=self.batch_size,
                pool_factor=self.pool_factor,
                batch_size_multiple=self.batch_size_multiple,
                batch_size_fn=self.batch_size_fn,
                device=self.device,
                train=self.is_train,
                sort=False,
                sort_within_batch=True,
                repeat=False,
                yield_raw_example=self.yield_raw_example
            )
            for batch in cur_iter:
                self.dataset = cur_iter.dataset
                yield batch

            # NOTE: This is causing some issues for consumer/producer,
            # as we may still have some of those examples in some queue
            # cur_dataset.examples = None
            # gc.collect()
            # del cur_dataset
            # gc.collect()

        def __iter__(self):
            num_batches = 0
            paths = self._paths
            if self.is_train and self.repeat:
                # Cycle through the shards indefinitely.
                paths = cycle(paths)
            for path in paths:
                for batch in self._iter_dataset(path):
                    yield batch
                    num_batches += 1
            if self.is_train and not self.repeat and \
                    num_batches % self.num_batches_multiple != 0:
                # When the dataset is not repeated, we might need to ensure that
                # the number of returned batches is the multiple of a given value.
                # This is important for multi GPU training to ensure that all
                # workers have the same number of batches to process.
                for path in paths:
                    for batch in self._iter_dataset(path):
                        yield batch
                        num_batches += 1
                        if num_batches % self.num_batches_multiple == 0:
                            return

    @classmethod
    def max_tok_len(cls, new, count, sofar):
        """
        In token batching scheme, the number of sequences is limited
        such that the total number of src/tgt tokens (including padding)
        in a batch <= batch_size
        """
        # Maintains the longest src and tgt length in the current batch
        global max_src_in_batch, max_tgt_in_batch  # this is a hack
        # Reset current longest length at a new batch (count=1)
        if count == 1:
            max_src_in_batch = 0
            max_tgt_in_batch = 0
        # Src: [<bos> w1 ... wN <eos>]
        max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
        # Tgt: [w1 ... wM <eos>]
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt[0]) + 1)
        src_elements = count * max_src_in_batch
        tgt_elements = count * max_tgt_in_batch
        return max(src_elements, tgt_elements)

    @classmethod
    def build_dataset_iter(cls, src_types, corpus_type, fields, opt, is_train=True, multi=False):
        """
        This returns user-defined train/validate data iterator for the trainer
        to iterate over. We implement simple ordered iterator strategy here,
        but more sophisticated strategy like curriculum learning is ok too.
        """
        dataset_paths = list(sorted(
            glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt')))
        if not dataset_paths:
            if is_train:
                raise ValueError('Training data %s not found' % opt.data)
            else:
                return None
        if multi:
            batch_size = 1
            batch_fn = None
            batch_size_multiple = 1
        else:
            batch_size = opt.batch_size if is_train else opt.valid_batch_size
            batch_fn = cls.max_tok_len \
                if is_train and opt.batch_type == "tokens" else None
            batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1

        device = "cuda" if opt.gpu_ranks else "cpu"

        return cls.DatasetLazyIter(
            src_types,
            dataset_paths,
            fields,
            batch_size,
            batch_fn,
            batch_size_multiple,
            device,
            is_train,
            opt.pool_factor,
            repeat=not opt.single_pass,
            num_batches_multiple=max(opt.accum_count) * opt.world_size,
            yield_raw_example=multi)

    @classmethod
    def build_dataset_iter_multiple(cls, src_types, train_shards, fields, opt):
        return cls.MultipleDatasetIterator(
            src_types, train_shards, fields, "cuda" if opt.gpu_ranks else "cpu", opt)
