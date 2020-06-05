from typing import *

from recordclass import RecordClass

from roosterize.data.LanguageId import LanguageId


class TokenConsts:
    CONTENT_UNK = "<UNK>"

    KIND_PAD = "<PAD>"
    KIND_UNK = "<UNK>"
    KIND_ID = "ID"
    KIND_KEYWORD = "KW"
    KIND_SYMBOL = "SYM"
    KIND_NUMBER = "NUM"
    KIND_STR = "STR"
    KIND_COMMENT = "COM"
    KIND_STR_IN_COMMENT = "STR_in_COM"
    KIND_BOS = "<BOS>"
    KIND_EOS = "<EOS>"

    OFFSET_UNSET = -2
    OFFSET_INVALID = -1
    OFFSET_BOS = -3
    OFFSET_EOS = -4

    LOC_UNSET = -2
    LOC_INVALID = -1

    KINDS_EMBEDDINGS: Dict[str, int] = {
        KIND_PAD: 0,
        KIND_UNK: 1,
        KIND_ID: 2,
        KIND_KEYWORD: 3,
        KIND_SYMBOL: 4,
        KIND_NUMBER: 5,
        KIND_STR: 6,
        KIND_COMMENT: 7,
        KIND_BOS: 8,
        KIND_EOS: 9,
    }


class Spacing(RecordClass):
    loffset: int = TokenConsts.OFFSET_UNSET
    coffset: int = TokenConsts.OFFSET_UNSET
    indentation: int = TokenConsts.OFFSET_UNSET

    def __str__(self):
        if self.coffset >= 0:
            return f"{self.coffset}s"
        else:
            return f"{self.loffset}l{self.indentation}s"

    def __hash__(self):
        return hash((self.loffset, self.coffset, self.indentation))

    def __eq__(self, other):
        if isinstance(other, Spacing):
            return self.loffset == other.loffset and self.coffset == other.coffset and self.indentation == other.indentation
        else:
            return False

    def describe(self):
        if self.coffset >= 0:
            return f"{self.coffset} space(s)"
        else:
            return f"{self.loffset} newline(s) and {self.indentation} space(s)"


class Token(RecordClass):
    content: str = TokenConsts.CONTENT_UNK
    kind: str = TokenConsts.CONTENT_UNK
    loffset: int = TokenConsts.OFFSET_UNSET
    coffset: int = TokenConsts.OFFSET_UNSET
    indentation: int = TokenConsts.OFFSET_UNSET

    lang_id: LanguageId = LanguageId.Unknown

    beg_charno: int = TokenConsts.LOC_UNSET
    end_charno: int = TokenConsts.LOC_UNSET
    lineno: int = TokenConsts.LOC_UNSET

    is_one_token_gallina: bool = False

    def str_with_space(self):
        return self.get_space() + self.content

    def get_space(self):
        if self.coffset >= 0:
            return " " * self.coffset
        elif self.indentation >= 0:
            return "\n" * self.loffset + " " * self.indentation
        else:
            # Default spacing
            return " "
        # end if

    def get_spacing(self) -> Spacing:
        return Spacing(self.loffset, self.coffset, self.indentation)

    def clear_spacing(self):
        self.loffset = TokenConsts.OFFSET_UNSET
        self.coffset = TokenConsts.OFFSET_UNSET
        self.indentation = TokenConsts.OFFSET_UNSET
        return

    def apply_spacing(self, spacing: Spacing):
        self.loffset = spacing.loffset
        self.coffset = spacing.coffset
        self.indentation = spacing.indentation
        return

    def clear_naming(self):
        self.content = TokenConsts.CONTENT_UNK
        return

    def is_ignored(self):
        return self.kind not in [TokenConsts.KIND_ID, TokenConsts.KIND_NUMBER, TokenConsts.KIND_STR, TokenConsts.KIND_KEYWORD, TokenConsts.KIND_SYMBOL]
