from typing import *

import collections
import numpy as np
import re

from seutil import LoggingUtils

from roosterize.data.Definition import Definition
from roosterize.data.Lemma import Lemma


class SubTokenizer:

    logger = LoggingUtils.get_logger(__name__)

    RE_FIRST_PASS_SPLITTER = re.compile(r"(?<=[_'])(?!$)|(?<!^)(?=[_'])|(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z])(?=[A-Z0-9])|(?<=[0-9])(?=[a-zA-Z0-9])")
    SUFFIXES = [
        "x", "g", "m", "n", "l", "r", "b", "f", "z", "r",
        "le", "lt", "ge", "gt", "ne", "eq", "Pn", "Vn", "Mn", "Un",
    ]
    SUFFIXES.sort(key=len, reverse=True)  # Favor longer suffixes over shorter ones
    RESERVE_WORDS = [
    ]
    ATOMIC_WORDS = [
        "nat", "group", "type", "elem", "core", "fix", "norm", "get", "exp", "shr", "None",
        "all", "max", "id", "set", "order", "comm", "sqrt", "subseq", "ext", "roots", "continuous",
        "coeff", "int", "left", "cons", "min", "right", "bits", "neq", "ret", "sign", "proper",
        "sigma", "tau", "kappa", "alpha", "beta", "pi", "horner", "uniqueness", "rect", "notin",
        "const", "join", "JOIN", "bitseq", "mul", "simple", "mult", "not", "rel", "refl", "col",
        "morph", "series", "lift", "perm", "prod", "coset", "trans", "sym", "pair", "cycle", "gcd",
        "block", "div", "can", "subgroup", "pred", "sign", "tree", "val", "mx", "real", "char",
        "size", "mut", "forall", "exists", "leq", "sub", "valid", "regular", "reg", "aux", "asym",
        "deriv", "face", "free", "field", "meet", "margin", "widen", "frac", "ring", "eval", "term",
        "mset", "imset", "fset", "add", "irr", "cat", "seq",
    ]

    CONTEXT_THRESHOLD = 3

    def __init__(self):
        self.context: Counter[str] = collections.Counter()
        return

    @classmethod
    def get_docs_sub_tokenizers(cls, named_commands: List[Lemma], definitions: List[Definition]) -> Dict[str, "SubTokenizer"]:
        docs_sub_tokenizers: Dict[str, SubTokenizer] = dict()
        docs_named_commands: Dict[str, List[Lemma]] = collections.defaultdict(list)
        docs_definitions: Dict[str, List[Definition]] = collections.defaultdict(list)

        for named_command in named_commands:
            docs_named_commands[named_command.data_index].append(named_command)
        # end for

        for definition in definitions:
            docs_definitions[definition.data_index].append(definition)
        # end for

        for doc, named_commands in docs_named_commands.items():
            docs_sub_tokenizers[doc] = SubTokenizer()
            docs_sub_tokenizers[doc].update_context(
                names=[nc.name for nc in named_commands],
                definition_names=[d.name for d in definitions],
                module_names=[qnp for nc in named_commands for qnp in nc.qname.split(".")[:-1]]
            )
        # end for
        return docs_sub_tokenizers

    def clear_context(self):
        self.context.clear()
        for t in self.RESERVE_WORDS:
            self.context[t] = self.CONTEXT_THRESHOLD
        # end for
        for t in self.ATOMIC_WORDS:
            self.context[t] = np.PINF
        # end for
        return

    def update_context(self, names: List[str], definition_names: List[str], module_names: List[str]):
        """
        Roughly sub-tokenizes the tokens and build a local context, to help precise sub-tokenization.
        """
        self.clear_context()

        for st in module_names:
            self.context[st.lower()] = np.PINF
        # end for

        for t in definition_names:
            rough_sub_tokens: List[str] = self.RE_FIRST_PASS_SPLITTER.split(t)

            for st in rough_sub_tokens:
                if len(st) == 2:  self.context[st.lower()] += 0.6
                if len(st) >= 3:  self.context[st.lower()] += 3
            # end for
        # end for

        for t in names:
            rough_sub_tokens: List[str] = self.RE_FIRST_PASS_SPLITTER.split(t)

            after_first_underscore = False
            after_second_underscore = False
            maybe_suffix = False
            for st in rough_sub_tokens:
                if st == "_":
                    if not after_first_underscore:
                        after_first_underscore = True
                    elif not after_second_underscore:
                        after_second_underscore = True
                    # end if
                    maybe_suffix = False
                    continue
                # end if

                if after_second_underscore and not st[0].islower():  maybe_suffix = True

                if not maybe_suffix:
                    if len(st) == 2:  self.context[st.lower()] += 0.2
                    if len(st) >= 3:  self.context[st.lower()] += 1
                    maybe_suffix = True
                # end if
            # end for
        # end for

        current_context_items = self.context.items()
        for st, cnt in current_context_items:
            if cnt >= np.PINF:  continue
            # Favor shorter names
            if len(st) >= 3 and st[:-1] in self.context:  self.context[st[:-1]] += cnt / 2
            if len(st) >= 4 and st[:-2] in self.context:  self.context[st[:-2]] += cnt / 3
            if len(st) >= 5 and st[:-3] in self.context:  self.context[st[:-3]] += cnt / 5
        # end for
        return

    @classmethod
    def can_tokenize(cls, token: str) -> bool:
        return len(token) > 0 and token[0].isalpha() and all([c.isalnum() or c in "_'" for c in token])

    def sub_tokenize(self,
            token: str,
    ) -> List[str]:
        """
        Sub-tokenizes the Coq identifier using a bunch of heuristics.
        :param token: the token to be sub-tokenized.
        """
        # Refuse to handle tokens with characters other than alnum & _'
        if not self.can_tokenize(token):  return [token]

        sub_tokens: List[str] = None

        # First pass - snake_case, CamelCase, digits
        sub_tokens_first_pass: List[str] = self.RE_FIRST_PASS_SPLITTER.split(token)

        # Special case: pure CamelCase, skip second pass
        if not "_" in sub_tokens_first_pass and sub_tokens_first_pass[0][0].isupper():
            sub_tokens = sub_tokens_first_pass
        # end if

        # Second pass - utilizing context
        if sub_tokens is None:
            sub_tokens_second_pass = list()
            after_first_underscore = False
            after_second_underscore = False
            maybe_suffix = False
            for st in sub_tokens_first_pass:
                if st == "_":
                    maybe_suffix = False
                    if not after_first_underscore:
                        after_first_underscore = True
                    elif not after_second_underscore:
                        after_second_underscore = True
                    # end if
                    sub_tokens_second_pass.append(st)
                    continue
                # end if

                if after_second_underscore and not st[0].islower():  maybe_suffix = True

                if len(st) == 1:
                    if not maybe_suffix:  maybe_suffix = True
                    sub_tokens_second_pass.append(st)
                    continue
                # end if

                if not maybe_suffix:
                    # Carefully identify additional suffixes, only if the new core part is more frequent than the current one
                    fractions = list()
                    core = st
                    while len(core) > 0:
                        # Keep atomic words
                        if self.context[core] == np.PINF:  break
                        for suffix in self.SUFFIXES:
                            if core.endswith(suffix):
                                fractions.insert(0, suffix)
                                core = core[:-len(suffix)]
                                break
                            # end if
                        else:
                            break
                        # end for-else
                    # end while
                    if len(core) > 0:  fractions.insert(0, core)

                    while len(fractions) > 1:
                        if self.context[fractions[0].lower()] >= self.context[st.lower()]:
                            # Adopt the new split
                            break
                        else:
                            fractions[0] = fractions[0]+fractions[1]
                            del fractions[1]
                        # end if
                    # end while

                    # Prefix checking (one character)
                    if len(fractions[0]) > 1 and fractions[0][0] and self.context[fractions[0][1:].lower()] >= self.context[fractions[0].lower()]:
                        # Take out the first char as prefix
                        fractions.insert(0, fractions[0][0])
                        fractions[1] = fractions[1][1:]
                    # end if

                    sub_tokens_second_pass.extend(fractions)

                    maybe_suffix = True
                else:
                    # Splits the suffix into small pieces, unless the word exists in context
                    if self.context[st.lower()] >= self.CONTEXT_THRESHOLD:
                        sub_tokens_second_pass.append(st)
                    else:
                        fractions = list()
                        remain = st
                        while len(remain) > 0:
                            # Try full match in context
                            if self.context[remain.lower()] >= self.CONTEXT_THRESHOLD:
                                fractions.insert(0, remain)
                                remain = ""
                                break
                            else:
                                # Try to find a suffix
                                for suffix in self.SUFFIXES:
                                    if remain.endswith(suffix):
                                        fractions.insert(0, suffix)
                                        remain = remain[:-len(suffix)]
                                        break
                                    # end if
                                else:
                                    # Try to find a suffix match in context
                                    for length in range(1, len(remain) + 1):
                                        if self.context[remain[-length:].lower()] >= self.CONTEXT_THRESHOLD:
                                            fractions.insert(0, remain[-length:])
                                            remain = remain[:-length]
                                            break
                                        # end if
                                    else:
                                        # If this is a CamelCase, leave as is; else take the last char out
                                        if remain[0].isupper():
                                            fractions.insert(0, remain)
                                            break
                                        else:
                                            fractions.insert(0, remain[-1])
                                            remain = remain[:-1]
                                        # end if
                                    # end for-else
                                # end for-else
                            # end if
                        # end while
                        sub_tokens_second_pass.extend(fractions)
                    # end if
                # end if
            # end for

            sub_tokens = sub_tokens_second_pass
        # end if

        return sub_tokens
