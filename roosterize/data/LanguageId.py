from typing import *

from enum import Enum


class LanguageId(Enum):
    Unknown = -1
    Vernac = 1
    Gallina = 2
    Ltac = 3
    Comment = 4

    # code-mixed lids are only assigned on sentences but not tokens
    LtacMixedWithGallina = 11
    VernacMixedWithGallina = 12

    def debug_repr(self) -> str:
        return self.__repr__()

    def __repr__(self):
        return {
            LanguageId.Unknown: "UNK",
            LanguageId.Vernac: "V",
            LanguageId.Gallina: "G",
            LanguageId.Ltac: "L",
            LanguageId.Comment: "C",
            LanguageId.LtacMixedWithGallina: "LG",
            LanguageId.VernacMixedWithGallina: "VG",
        }[self]

    def __str__(self):
        return self.__repr__()

    @property
    def base_lid(self) -> "LanguageId":
        """
        :return the base lid of a code-mixed lid; if self is not a code-mixed lid, return self.
        """
        if self == LanguageId.LtacMixedWithGallina:
            return LanguageId.Ltac
        elif self == LanguageId.VernacMixedWithGallina:
            return LanguageId.Vernac
        else:
            return self
        # end if
