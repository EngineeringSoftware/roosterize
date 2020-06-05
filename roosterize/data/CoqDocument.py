from typing import *

import copy
from recordclass import RecordClass

from roosterize.data.LanguageId import LanguageId
from roosterize.data.Token import Token


class VernacularSentence(RecordClass):
    tokens: List[Token] = None

    def __copy__(self):
        return VernacularSentence(
            tokens=copy.deepcopy(self.tokens),
        )

    def classify_lid(self) -> LanguageId:
        if all([t.lang_id == LanguageId.Comment for t in self.tokens]):
            return LanguageId.Comment
        if any([t.lang_id == LanguageId.Ltac for t in self.tokens]):
            if any([t.lang_id == LanguageId.Gallina and not t.is_one_token_gallina for t in self.tokens]):
                return LanguageId.LtacMixedWithGallina
            else:
                return LanguageId.Ltac
            # end if
        elif any([t.lang_id == LanguageId.Gallina and not t.is_one_token_gallina for t in self.tokens]):
            return LanguageId.VernacMixedWithGallina
        else:
            return LanguageId.Vernac
        # end if

    def str_with_space(self):
        return "".join([t.str_with_space() for t in self.tokens])


class CoqDocument(RecordClass):
    # tokens: List[Token] = None
    sentences: List[VernacularSentence] = None
    file_name: str = ""
    project_name: str = ""
    revision: str = ""

    def get_all_tokens(self) -> List[Token]:
        return [t for s in self.sentences for t in s.tokens]

    def get_data_index(self):
        return f"{self.project_name}/{self.file_name}"

    def __copy__(self):
        return CoqDocument(
            sentences=copy.deepcopy(self.sentences),
            file_name=self.file_name,
            project_name=self.project_name,
            revision=self.revision,
        )

    def debug_repr(self) -> str:
        s = f"File: {self.file_name}\n"
        s += f"Project: {self.project_name}\n"
        s += f"Revision: {self.revision}\n"
        s += f"#sentences: {len(self.sentences)}\n"
        s += f"#tokens: {len([t for sent in self.sentences for t in sent.tokens])}\n"
        s += "\n"

        for sent in self.sentences:
            for t in sent.tokens:
                s += f"<{t.content}:{t.lang_id.debug_repr()}{'ot' if t.is_one_token_gallina else ''}:{t.kind}:{t.loffset}:{t.coffset}:{t.indentation}> "
            # end for
            s += "\n"
        # end for

        return s

    def str_with_space(self):
        return "".join([s.str_with_space() for s in self.sentences])
