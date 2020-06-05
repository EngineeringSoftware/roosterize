from typing import *

import re


class ParserUtils:
    @classmethod
    def get_unicode_offsets(cls, code: str) -> List[int]:
        # Get the offsets sequence to represent how many additional bytes (than 1 byte) a character at the given offsets need in UTF-8 embedding.
        # The index is repeated multiple times, as many as the number of additional bytes needed.
        return [i for i in range(len(code)) for _ in range(len(code[i].encode("UTF-8"))-1) if not code[i].isascii()]

    @classmethod
    def coq_charno_to_actual_charno(cls, coq_charno: int, unicode_offsets: List[int]) -> int:
        return coq_charno - len([i for i, offset in enumerate(unicode_offsets) if i+offset < coq_charno])

    @classmethod
    def actual_charno_to_coq_charno_bp(cls, actual_charno: int, unicode_offsets: List[int]) -> int:
        return actual_charno + len([offset for offset in unicode_offsets if offset < actual_charno])

    @classmethod
    def actual_charno_to_coq_charno_ep(cls, actual_charno: int, unicode_offsets: List[int]) -> int:
        return actual_charno + len([offset for offset in unicode_offsets if offset <= actual_charno])

    @classmethod
    def is_ws_or_comment(cls, s: str):
        s_no_ws = s.strip()
        if len(s_no_ws) == 0:  return True
        if s_no_ws[:2] == "(*" and s_no_ws[-2:] == "*)":  return True

        return False

    REGEX_COMMENT = re.compile(r"\s*(?P<com>\(\*[\s\S]*\*\))\s*")

    @classmethod
    def find_comment(cls, s: str) -> Optional[Tuple[int, int]]:
        m = cls.REGEX_COMMENT.fullmatch(s)
        if m is None:
            return None
        else:
            return m.start("com"), m.end("com")
        # end if
