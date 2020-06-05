from typing import *

from recordclass import RecordClass

from roosterize.data.Token import Token
from roosterize.sexp import *


class Lemma(RecordClass):
    data_index: str = ""

    vernac_command: List[Token] = None
    name: str = ""
    qname: str = ""

    statement: List[Token] = None
    ast_sexp: SexpNode = None
    backend_sexp: SexpNode = None

    uid: int = -1  # Used only for indexing in this dataset

    def vernac_command_with_space(self):
        return ''.join([t.str_with_space() for t in self.vernac_command])

    def statement_with_space(self):
        return ''.join([t.str_with_space() for t in self.statement])

    def __str__(self):
        s = ""
        s += f"data_index: {self.data_index}\n"
        s += f"name: {self.name}\n"
        s += f"qname: {self.qname}\n"
        s += f"vernac_command: {self.vernac_command_with_space()}\n"
        s += f"statement: {self.statement_with_space()}\n"
        return s

    def __repr__(self):
        return self.__str__()
