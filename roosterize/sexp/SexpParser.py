from typing import *

import string

from seutil import LoggingUtils

from roosterize.sexp.SexpList import SexpList
from roosterize.sexp.SexpNode import SexpNode
from roosterize.sexp.SexpString import SexpString


class SexpParser:

    logger = LoggingUtils.get_logger(__name__)

    # non_par_printables = "".join(c for c in pyparsing.printables if c not in "()")

    c_quote = '"'
    c_escape = '\\'
    c_lpar = '('
    c_rpar = ')'

    @classmethod
    def parse_list(cls, sexp_list_str: str) -> List[SexpNode]:
        """
        Parses a string of a list of s-expressions.
        """
        sexp_list: List[SexpNode] = list()
        sexp_list_str = sexp_list_str.strip()
        cur_pos = 0
        while cur_pos < len(sexp_list_str):
            sexp, cur_pos = cls.parse_recur(sexp_list_str, cur_pos)
            sexp_list.append(sexp)
        # end while

        return sexp_list


    @classmethod
    def parse(cls, sexp_str: str) -> SexpNode:
        """
        Parses a string of s-expression to structured s-expression.
        """
        sexp, end_pos = cls.parse_recur(sexp_str, 0)
        if end_pos != len(sexp_str):
            cls.logger.warning(f"Parsing did not terminate at the last character! ({end_pos}/{len(sexp_str)})")
        # end if

        return sexp

    @classmethod
    def parse_recur(cls, sexp_str: str, cur_pos: int) -> Tuple[SexpNode, int]:
        try:
            cur_char = None

            # Find the next non-whitespace char
            def parse_ws():
                nonlocal cur_char, sexp_str, cur_pos
                cur_char = sexp_str[cur_pos]
                while cur_char in string.whitespace:
                    cur_pos += 1
                    cur_char = sexp_str[cur_pos]
                # end while
                return
            # end def

            parse_ws()

            if cur_char == cls.c_lpar:
                # Start SexpList
                child_sexps: List[SexpNode] = list()
                cur_pos += 1

                while True:
                    parse_ws()
                    cur_char = sexp_str[cur_pos]
                    if cur_char == cls.c_rpar:
                        break
                    else:
                        child_sexp, cur_pos = cls.parse_recur(sexp_str, cur_pos)
                        child_sexps.append(child_sexp)
                    # end if
                # end while

                return SexpList(child_sexps), cur_pos + 1  # Consume the ending par
            elif cur_char == cls.c_quote:
                # Start string literal
                cur_token = cur_char
                cur_pos += 1
                while True:
                    cur_char = sexp_str[cur_pos]
                    if cur_char == cls.c_quote:
                        # End string literal
                        cur_token += cur_char
                        break
                    elif cur_char == cls.c_escape:
                        # Goto and escape the next char
                        cur_pos += 1
                        cur_char = ("\\" + sexp_str[cur_pos]).encode().decode("unicode-escape")
                    # end if
                    cur_token += cur_char
                    cur_pos += 1
                # end while

                return SexpString(cur_token[1:-1]), cur_pos + 1  # Consume the ending quote
            else:
                # Start a normal token
                cur_token = cur_char
                cur_pos += 1
                while True:
                    cur_char = sexp_str[cur_pos]
                    if cur_char == cls.c_lpar or cur_char == cls.c_rpar or cur_char == cls.c_quote or cur_char in string.whitespace:
                        break
                    # end if
                    cur_token += cur_char
                    cur_pos += 1
                # end while

                return SexpString(cur_token), cur_pos  # Does not consume the stopping char
            # end if
        except IndexError as e:
            raise ValueError("Malformed sexp") from e

    @classmethod
    def from_python_ds(cls, python_ds: Union[str, Iterable]) -> SexpNode:
        if isinstance(python_ds, str):
            return SexpString(python_ds)
        else:
            return SexpList([cls.from_python_ds(child) for child in python_ds])
        # end if
