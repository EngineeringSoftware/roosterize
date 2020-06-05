from typing import *

import collections
from itertools import chain, starmap
import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example, Field
from torchtext.vocab import Vocab


def _join_dicts(*args):
    """
    Args:
        dictionaries with disjoint keys.

    Returns:
        a single dictionary that has the union of these keys.
    """

    return dict(chain(*[d.items() for d in args]))


def _dynamic_dict(
        example: dict,
        src_types: List[str],
        src_types_fields: Dict[str, Field],
        tgt_field: Field,
) -> Tuple[Vocab, dict]:
    """Create copy-vocab and numericalize with it.

    In-place adds ``"src_map"`` to ``example``. That is the copy-vocab
    numericalization of the tokenized ``example["src"]``. If ``example``
    has a ``"tgt"`` key, adds ``"alignment"`` to example. That is the
    copy-vocab numericalization of the tokenized ``example["tgt"]``. The
    alignment has an initial and final UNK token to match the BOS and EOS
    tokens.

    Args:
        example (dict): An example dictionary with a ``"src"`` key and
            maybe a ``"tgt"`` key. (This argument changes in place!)
        src_field (torchtext.data.Field): Field object.
        tgt_field (torchtext.data.Field): Field object.

    Returns:
        torchtext.data.Vocab and ``example``, changed as described.
    """
    # src_ex_vocab_list = list()
    unk = None
    pad = None
    src_counter: Counter = collections.Counter()

    for src_type in src_types:
        src = src_types_fields[src_type].tokenize(example[f"src.{src_type}"])

        # add into counter
        src_counter.update(src)

        # update or match unk, pad
        unk_ = src_types_fields[src_type].unk_token
        if unk is None:
            unk = unk_
        else:
            assert unk == unk_
        # end if

        pad_ = src_types_fields[src_type].pad_token
        if pad is None:
            pad = pad_
        else:
            assert pad == pad_
        # end if
    # end for

    # Build src_ex_vocab  (shared among all srcs)
    src_ex_vocab = Vocab(src_counter, specials=[unk, pad])
    unk_idx = src_ex_vocab.stoi[unk]

    # Map source tokens to indices in the dynamic dict.
    for src_type in src_types:
        src = src_types_fields[src_type].tokenize(example[f"src.{src_type}"])
        src_map = torch.LongTensor([src_ex_vocab.stoi[w] for w in src])
        example[f"src_map.{src_type}"] = src_map
    # end for

    example[f"src_ex_vocab"] = src_ex_vocab

    if "tgt" in example:
        tgt = tgt_field.tokenize(example["tgt"])
        mask = torch.LongTensor(
            [unk_idx] + [src_ex_vocab.stoi[w] for w in tgt] + [unk_idx])
        example["alignment"] = mask
    # end if
    return src_ex_vocab, example


class MultiSourceDataset(TorchtextDataset):
    """Contain data and process it. Allows multiple source sentences.

    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of :class:`torchtext.data.Example` objects.
    torchtext's iterators then know how to use these examples to make batches.

    Args:
        fields (dict[str, Field]): a dict with the structure
            returned by :func:`onmt.inputters.get_fields()`. Usually
            that means the dataset side, ``"src.*"`` or ``"tgt"``. Keys match
            the keys of items yielded by the ``readers``, while values
            are lists of (name, Field) pairs. An attribute with this
            name will be created for each :class:`torchtext.data.Example`
            object and its value will be the result of applying the Field
            to the data that matches the key. The advantage of having
            sequences of fields for each piece of raw input is that it allows
            the dataset to store multiple "views" of each input, which allows
            for easy implementation of token-level features, mixed word-
            and character-level models, and so on. (See also
            :class:`onmt.inputters.TextMultiField`.)
        readers (Iterable[onmt.inputters.DataReaderBase]): Reader objects
            for disk-to-dict. The yielded dicts are then processed
            according to ``fields``.
        data (Iterable[Tuple[str, Any]]): (name, ``data_arg``) pairs
            where ``data_arg`` is passed to the ``read()`` method of the
            reader in ``readers`` at that position. (See the reader object for
            details on the ``Any`` type.)
        dirs (Iterable[str or NoneType]): A list of directories where
            data is contained. See the reader object for more details.
        sort_key (Callable[[torchtext.data.Example], Any]): A function
            for determining the value on which data is sorted (i.e. length).
        filter_pred (Callable[[torchtext.data.Example], bool]): A function
            that accepts Example objects and returns a boolean value
            indicating whether to include that example in the dataset.

    Attributes:
        src_vocabs (List[List[torchtext.data.Vocab]]): Used with dynamic dict/copy
            attention. There is a very short vocab for each src example.
            It contains just the source words, e.g. so that the generator can
            predict to copy them.
    """

    def __init__(self,
            src_types: List[str],
            fields,
            readers: List,
            data: List[Tuple[str, Any]],
            dirs: List[str],
            sort_key,
            filter_pred=None,
            can_copy: bool = False,
    ):
        self.sort_key = sort_key

        read_iters = [r.read(dat[1], dat[0], dir_) for r, dat, dir_
                      in zip(readers, data, dirs)]

        # self.src_vocabs is used in collapse_copy_scores and Translator.py
        self.src_vocabs = []
        examples = []
        for ex_dict in starmap(_join_dicts, zip(*read_iters)):
            if can_copy:
                tgt_field = fields['tgt']
                src_types_fields = {
                    src_type: fields[f"src.{src_type}"].base_field
                    for src_type in src_types
                }
                src_ex_vocab, ex_dict = _dynamic_dict(
                    ex_dict, src_types, src_types_fields, tgt_field.base_field)
                self.src_vocabs.append(src_ex_vocab)
            # end if
            ex_fields = {k: [(k, v)] for k, v in fields.items() if
                         k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        # fields needs to have only keys that examples have as attrs
        fields = []
        for _, nf_list in ex_fields.items():
            assert len(nf_list) == 1
            fields.append(nf_list[0])

        super(MultiSourceDataset, self).__init__(examples, fields, filter_pred)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)
