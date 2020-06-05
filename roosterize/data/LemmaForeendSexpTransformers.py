from typing import *

from roosterize.data.TreeTransformation import TreeTransformation, TreeTransformationConsts
from roosterize.data.TreeTransformer import TreeTransformer, Depth10Transformer, RandomTransformer
from roosterize.sexp import *


class Level0Transformer(TreeTransformer):
    """
    Removes the lemma name.
    """

    def __init__(self):
        super().__init__()
        return

    def transform(self, sexp: SexpNode) -> SexpNode:
        # ( VernacExpr () ( X X (  (   (    ( v ( Id <lemma_name> ) ) ( loc ... ) ) ...
        #   R          R    R R                   R-------------------------------------^ K...
        #   0          1  2     22 220 2200 22000
        rm = SexpString(TreeTransformationConsts.REMOVE)
        keep = SexpString(TreeTransformationConsts.KEEP)
        return SexpList([
            rm,
            rm,
            SexpList([
                rm,
                rm,
                SexpList([
                    SexpList([
                        SexpList([rm] + [keep] * (len(sexp[2][2][0][0])-1))
                    ] + [keep] * (len(sexp[2][2][0])-1))
                ] + [keep] * (len(sexp[2][2])-1))
            ])
        ])


class Level1Transformer(TreeTransformer):
    """
    Aggressive cutting to remove all unhelpful stuff from the tree.
    - Removes loc constructs;
    - Removes all name components indicators;
    - Removes all except the last name component for each qualified name;
    - Changes some words, e.g., CApp, CNotation ... to <CApp>, <CNotation> ..., to protect them from sub-tokenization.
    """

    def __init__(self):
        super().__init__()
        return

    def transform(self, sexp: SexpNode) -> SexpNode:
        # ( ( v <X> ) ( loc () ) )
        #     R +     R
        if sexp.is_list() and len(sexp) == 2 and \
                sexp[0].is_list() and len(sexp[0]) == 2 and \
                sexp[0][0].is_string() and sexp[0][0].content == "v" and \
                sexp[1].is_list() and len(sexp[1]) == 2 and \
                sexp[1][0].is_string() and sexp[1][0].content == "loc" and \
                sexp[1][1].is_list():
            # then
            return SexpList([
                SexpList([
                    SexpString(TreeTransformationConsts.REMOVE),
                    self.transform(sexp[0][1]),
                ]),
                SexpString(TreeTransformationConsts.REMOVE),
            ])
        # end if

        # ( <A> <X> )  where <A> in [Id, ConstRef, Name]
        #   R   +
        if sexp.is_list() and len(sexp) == 2 and \
                sexp[0].is_string() and sexp[0].content in ["Id", "ConstRef", "Name"]:
            # then
            return SexpList([
                SexpString(TreeTransformationConsts.REMOVE),
                self.transform(sexp[1]),
            ])
        # end if

        # ( <A> ( <Xs> ) <Ys> )  where <A> in [MPdot, DirPath, Constant, MutInd, MPfile]
        #   R   R        R. +
        if sexp.is_list() and len(sexp) > 2 and \
                sexp[0].is_string() and sexp[0].content in ["MPdot", "DirPath", "Constant", "MutInd", "MPfile"] and \
                sexp[1].is_list() and len(sexp[1]) > 0:
            # then
            return SexpList([SexpString(TreeTransformationConsts.REMOVE)] * (len(sexp)-1) + [self.transform(sexp[-1])])
        # end if

        # ( <A> ( <Xs> ) )  where <A> in [MPdot, DirPath, Constant, MutInd, MPfile]
        #   R     R. +
        if sexp.is_list() and len(sexp) == 2 and \
                sexp[0].is_string() and sexp[0].content in ["MPdot", "DirPath", "Constant", "MutInd", "MPfile"] and \
                sexp[1].is_list() and len(sexp[1]) > 0:
            # then
            return SexpList([
                SexpString(TreeTransformationConsts.REMOVE),
                SexpList([SexpString(TreeTransformationConsts.REMOVE)] * (len(sexp[1])-1) + [self.transform(sexp[1][-1])]),
            ])
        # end if

        # ( <A> <X> () )  where <A> in [CRef]
        #   R   +   R
        if sexp.is_list() and len(sexp) == 3 and \
                sexp[0].is_string() and sexp[0].content in ["CRef"] and \
                sexp[2].is_list() and len(sexp[2]) == 0:
            # then
            return SexpList([
                SexpString(TreeTransformationConsts.REMOVE),
                self.transform(sexp[1]),
                SexpString(TreeTransformationConsts.REMOVE),
            ])
        # end if

        # ( IndRef ( ( <Xs> ) <n> ) )
        #   R          R. +   R
        if sexp.is_list() and len(sexp) == 2 and \
                sexp[0].is_string() and sexp[0].content == "IndRef" and \
                sexp[1].is_list() and len(sexp[1]) == 2 and \
                sexp[1][0].is_list() and len(sexp[1][0]) > 0:
            # then
            return SexpList([
                SexpString(TreeTransformationConsts.REMOVE),
                SexpList([
                    SexpList([SexpString(TreeTransformationConsts.REMOVE)] * (len(sexp[1][0])-1) + [self.transform(sexp[1][0][-1])]),
                    SexpString(TreeTransformationConsts.REMOVE),
                ]),
            ])
        # end if

        # ( ConstructRef ( ( ( <Xs> ) <n> ) <m> ) )
        #   R                  R. +   R     R
        if sexp.is_list() and len(sexp) == 2 and \
                sexp[0].is_string() and sexp[0].content == "ConstructRef" and \
                sexp[1].is_list() and len(sexp[1]) == 2 and \
                sexp[1][0].is_list() and len(sexp[1][0]) == 2 and \
                sexp[1][0][0].is_list() and len(sexp[1][0][0]) > 0:
            # then
            return SexpList([
                SexpString(TreeTransformationConsts.REMOVE),
                SexpList([
                    SexpList([
                        SexpList([SexpString(TreeTransformationConsts.REMOVE)] * (len(sexp[1][0][0])-1) + [self.transform(sexp[1][0][0][-1])]),
                        SexpString(TreeTransformationConsts.REMOVE),
                    ]),
                    SexpString(TreeTransformationConsts.REMOVE),
                ]),
            ])
        # end if

        # ( <A>      ... )  where A in [CXxx]
        #   U'<'A'>' +
        if sexp.is_list() and len(sexp) >= 1 and \
                sexp[0].is_string() and sexp[0].content in ["CRef", "CFix", "CCoFix", "CProdN", "CLambdaN", "CLetIn", "CAppExpl", "CApp", "CRecord", "CCases", "CLetTuple", "CIf", "CHole", "CPatVar", "CEvar", "CSort", "CCast", "CNotation", "CGeneralization", "CPrim", "CDelimiters", "CLocal"]:
            # then
            return SexpList([
                SexpString(TreeTransformationConsts.UPDATE + "<" + sexp[0].content + ">"),
            ] + [self.transform(x) for x in sexp[1:]])
        # end if

        # Default
        if sexp.is_list():
            return SexpList([self.transform(c) for c in sexp.get_children()])
        else:
            return SexpString(TreeTransformationConsts.KEEP)
        # end if


class Level1XTransformer(TreeTransformer):
    """
    Aggressive cutting to remove all unhelpful stuff from the tree.
    - Removes loc constructs;
    - Removes all name components indicators;
    - Removes all except the last name component for each qualified name;
    - Changes some words, e.g., CApp, CNotation ... to <CApp>, <CNotation> ..., to protect them from sub-tokenization.
    """

    def __init__(self):
        super().__init__()
        return

    def transform(self, sexp: SexpNode) -> SexpNode:
        # ( ( v <X> ) ( loc () ) )
        #     R +     R
        if sexp.is_list() and len(sexp) == 2 and \
                sexp[0].is_list() and len(sexp[0]) == 2 and \
                sexp[0][0].is_string() and sexp[0][0].content == "v" and \
                sexp[1].is_list() and len(sexp[1]) == 2 and \
                sexp[1][0].is_string() and sexp[1][0].content == "loc" and \
                sexp[1][1].is_list():
            # then
            return SexpList([
                SexpList([
                    SexpString(TreeTransformationConsts.REMOVE),
                    self.transform(sexp[0][1]),
                ]),
                SexpString(TreeTransformationConsts.REMOVE),
            ])
        # end if

        # ( <A> <X> )  where <A> in [Id, <del>ConstRef,</del> Name]
        #   R   +
        if sexp.is_list() and len(sexp) == 2 and \
                sexp[0].is_string() and sexp[0].content in ["Id", "Name"]:
            # then
            return SexpList([
                SexpString(TreeTransformationConsts.REMOVE),
                self.transform(sexp[1]),
            ])
        # end if

        # ( <A> ( <Xs> ) <Ys> )  where <A> in [MPdot, DirPath, <del>Constant,</del> MutInd, MPfile]
        #   R   R        R. +
        if sexp.is_list() and len(sexp) > 2 and \
                sexp[0].is_string() and sexp[0].content in ["MPdot", "DirPath", "MutInd", "MPfile"] and \
                sexp[1].is_list() and len(sexp[1]) > 0:
            # then
            return SexpList([SexpString(TreeTransformationConsts.REMOVE)] * (len(sexp)-1) + [self.transform(sexp[-1])])
        # end if

        # ( <A> ( <Xs> ) )  where <A> in [MPdot, DirPath, <del>Constant,</del> MutInd, MPfile]
        #   R     R. +
        if sexp.is_list() and len(sexp) == 2 and \
                sexp[0].is_string() and sexp[0].content in ["MPdot", "DirPath", "MutInd", "MPfile"] and \
                sexp[1].is_list() and len(sexp[1]) > 0:
            # then
            return SexpList([
                SexpString(TreeTransformationConsts.REMOVE),
                SexpList([SexpString(TreeTransformationConsts.REMOVE)] * (len(sexp[1])-1) + [self.transform(sexp[1][-1])]),
            ])
        # end if

        # ( <A> <X> () )  where <A> in [CRef]
        #   R   +   R
        if sexp.is_list() and len(sexp) == 3 and \
                sexp[0].is_string() and sexp[0].content in ["CRef"] and \
                sexp[2].is_list() and len(sexp[2]) == 0:
            # then
            return SexpList([
                SexpString(TreeTransformationConsts.REMOVE),
                self.transform(sexp[1]),
                SexpString(TreeTransformationConsts.REMOVE),
            ])
        # end if

        # # ( IndRef ( ( <Xs> ) <n> ) )
        # #   R          R. +   R
        # if sexp.is_list() and len(sexp) == 2 and \
        #         sexp[0].is_string() and sexp[0].content == "IndRef" and \
        #         sexp[1].is_list() and len(sexp[1]) == 2 and \
        #         sexp[1][0].is_list() and len(sexp[1][0]) > 0:
        #     # then
        #     return SexpList([
        #         SexpString(TreeTransformationConsts.REMOVE),
        #         SexpList([
        #             SexpList([SexpString(TreeTransformationConsts.REMOVE)] * (len(sexp[1][0])-1) + [self.transform(sexp[1][0][-1])]),
        #             SexpString(TreeTransformationConsts.REMOVE),
        #         ]),
        #     ])
        # # end if
        #
        # # ( ConstructRef ( ( ( <Xs> ) <n> ) <m> ) )
        # #   R                  R. +   R     R
        # if sexp.is_list() and len(sexp) == 2 and \
        #         sexp[0].is_string() and sexp[0].content == "ConstructRef" and \
        #         sexp[1].is_list() and len(sexp[1]) == 2 and \
        #         sexp[1][0].is_list() and len(sexp[1][0]) == 2 and \
        #         sexp[1][0][0].is_list() and len(sexp[1][0][0]) > 0:
        #     # then
        #     return SexpList([
        #         SexpString(TreeTransformationConsts.REMOVE),
        #         SexpList([
        #             SexpList([
        #                 SexpList([SexpString(TreeTransformationConsts.REMOVE)] * (len(sexp[1][0][0])-1) + [self.transform(sexp[1][0][0][-1])]),
        #                 SexpString(TreeTransformationConsts.REMOVE),
        #             ]),
        #             SexpString(TreeTransformationConsts.REMOVE),
        #         ]),
        #     ])
        # # end if

        # ( <A>      ... )  where A in [CXxx]
        #   U'<'A'>' +
        if sexp.is_list() and len(sexp) >= 1 and \
                sexp[0].is_string() and sexp[0].content in ["CRef", "CFix", "CCoFix", "CProdN", "CLambdaN", "CLetIn", "CAppExpl", "CApp", "CRecord", "CCases", "CLetTuple", "CIf", "CHole", "CPatVar", "CEvar", "CSort", "CCast", "CNotation", "CGeneralization", "CPrim", "CDelimiters", "CLocal"]:
            # then
            return SexpList([
                SexpString(TreeTransformationConsts.UPDATE + "<" + sexp[0].content + ">"),
            ] + [self.transform(x) for x in sexp[1:]])
        # end if

        # Default
        if sexp.is_list():
            return SexpList([self.transform(c) for c in sexp.get_children()])
        else:
            return SexpString(TreeTransformationConsts.KEEP)
        # end if


class ForeendSexpRandomTransformer(RandomTransformer):
    TARGET_SIZE_FRACTION = 0.092

    def __init__(self):
        super().__init__(target_size_fraction=self.TARGET_SIZE_FRACTION)
        return


class LemmaForeendSexpTransformers:

    LEVEL_0 = "l0"
    LEVEL_1 = "l1"
    DEPTH_10 = "d10"
    RANDOM = "rnd"
    LEVEL_1x = "l1x"

    LEVELS = [
        LEVEL_0,
        LEVEL_1,
    ]

    SPECIALS = [
        DEPTH_10,
        RANDOM,
        LEVEL_1x,
    ]

    TWO_STEPS = [
        LEVEL_0,
        LEVEL_1,
        LEVEL_1x,
    ]

    TRANSFORMERS = {
        LEVEL_0: Level0Transformer,
        LEVEL_1: Level1Transformer,
        DEPTH_10: Depth10Transformer,
        RANDOM: ForeendSexpRandomTransformer,
        LEVEL_1x: Level1XTransformer,
    }

    @classmethod
    def transform(cls, tr_name: str, sexp: SexpNode) -> SexpNode:
        transformer = cls.TRANSFORMERS[tr_name]()
        if tr_name in cls.TWO_STEPS:
            transformation = TreeTransformation(transformer.transform(sexp))
            transformed = transformation.transform_sexp(sexp)
        else:
            transformed = transformer.transform(sexp)
        # end if
        return transformed
