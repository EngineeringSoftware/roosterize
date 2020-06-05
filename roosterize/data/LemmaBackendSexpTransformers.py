from typing import *

from roosterize.data.TreeTransformation import TreeTransformation, TreeTransformationConsts
from roosterize.data.TreeTransformer import TreeTransformer, Depth10Transformer, RandomTransformer
from roosterize.sexp import *


class Level1Transformer(TreeTransformer):
    """
    Aggressive cutting to remove all unhelpful stuff from the tree.
    - Removes loc constructs;
    - Removes all name components indicators;
    - Removes all except the last name component for each qualified name;
    - Removes "Explicit" in GProd;
    - Changes some words, e.g., GProd, GApp ... to <GProd>, <GApp> ..., to protect them from sub-tokenization.
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
                sexp[1][1].is_list() and len(sexp[1][1]) == 0:
            # then
            return SexpList([
                SexpList([
                    SexpString(TreeTransformationConsts.REMOVE),
                    self.transform(sexp[0][1]),
                ]),
                SexpString(TreeTransformationConsts.REMOVE),
            ])
        # end if

        # ( <A> <X> )  where <A> in [Id, ConstRef, Name, GVar]
        #   R   +
        if sexp.is_list() and len(sexp) == 2 and \
                sexp[0].is_string() and sexp[0].content in ["Id", "ConstRef", "Name", "GVar"]:
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

        # ( <A> <X> () )  where <A> in [GRef]
        #   R   +   R
        if sexp.is_list() and len(sexp) == 3 and \
                sexp[0].is_string() and sexp[0].content in ["GRef"] and \
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

        # ( GProd    . <X> ... )
        #   U<GProd> + R   +
        if sexp.is_list() and len(sexp) >= 3 and \
                sexp[0].is_string() and sexp[0].content == "GProd":
            # then
            return SexpList([
                SexpString(TreeTransformationConsts.UPDATE + "<GProd>"),
                self.transform(sexp[1]),
                SexpString(TreeTransformationConsts.REMOVE),
            ] + [self.transform(x) for x in sexp[3:]])
        # end if

        # ( <A>      ... )  where A in [GEvar, GPatVar, GApp, GLambda, GLetIn, GCases, GLetTuple, GIf, GRec, GSort, GHole, GCast, GInt]
        #   U'<'A'>' +
        if sexp.is_list() and len(sexp) >= 1 and \
                sexp[0].is_string() and sexp[0].content in ["GEvar", "GPatVar", "GApp", "GLambda", "GLetIn", "GCases", "GLetTuple", "GIf", "GRec", "GSort", "GHole", "GCast", "GInt"]:
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


class Level2Transformer(TreeTransformer):
    """
    Takes the core part of the tree.
    - Removes <GProd>s in the beginning of tree.
    """

    def __init__(self):
        super().__init__()
        return

    def transform(self, sexp: SexpNode) -> SexpNode:
        # ( <GProd> <Xs> )
        #   R       R. +
        if sexp.is_list() and len(sexp) >= 2 and \
                sexp[0].is_string() and sexp[0].content == "<GProd>":
            # then
            return SexpList([SexpString(TreeTransformationConsts.REMOVE)] * (len(sexp)-1) + [self.transform(sexp[-1])])
        # end if

        # Default (after seeing the first non-GProd
        return SexpString(TreeTransformationConsts.KEEP)


class Level1XTransformer(TreeTransformer):
    """
    Aggressive cutting to remove all unhelpful stuff from the tree.
    - Removes loc constructs;
    - Removes all name components indicators;
    - Removes all except the last name component for each qualified name;
    - Removes "Explicit" in GProd;
    - Changes some words, e.g., GProd, GApp ... to <GProd>, <GApp> ..., to protect them from sub-tokenization.
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
                sexp[1][1].is_list() and len(sexp[1][1]) == 0:
            # then
            return SexpList([
                SexpList([
                    SexpString(TreeTransformationConsts.REMOVE),
                    self.transform(sexp[0][1]),
                ]),
                SexpString(TreeTransformationConsts.REMOVE),
            ])
        # end if

        # ( <A> <X> )  where <A> in [Id, <del>ConstRef,</del> Name, GVar]
        #   R   +
        if sexp.is_list() and len(sexp) == 2 and \
                sexp[0].is_string() and sexp[0].content in ["Id", "Name", "GVar"]:
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

        # ( <A> <X> () )  where <A> in [GRef]
        #   R   +   R
        if sexp.is_list() and len(sexp) == 3 and \
                sexp[0].is_string() and sexp[0].content in ["GRef"] and \
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

        # ( GProd    . <X> ... )
        #   U<GProd> + R   +
        if sexp.is_list() and len(sexp) >= 3 and \
                sexp[0].is_string() and sexp[0].content == "GProd":
            # then
            return SexpList([
                SexpString(TreeTransformationConsts.UPDATE + "<GProd>"),
                self.transform(sexp[1]),
                SexpString(TreeTransformationConsts.REMOVE),
            ] + [self.transform(x) for x in sexp[3:]])
        # end if

        # ( <A>      ... )  where A in [GEvar, GPatVar, GApp, GLambda, GLetIn, GCases, GLetTuple, GIf, GRec, GSort, GHole, GCast, GInt]
        #   U'<'A'>' +
        if sexp.is_list() and len(sexp) >= 1 and \
                sexp[0].is_string() and sexp[0].content in ["GEvar", "GPatVar", "GApp", "GLambda", "GLetIn", "GCases", "GLetTuple", "GIf", "GRec", "GSort", "GHole", "GCast", "GInt"]:
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


class BackendSexpRandomTransformer(RandomTransformer):

    TARGET_SIZE_FRACTION = 0.085

    def __init__(self):
        super().__init__(target_size_fraction=self.TARGET_SIZE_FRACTION)
        return


class LemmaBackendSexpTransformers:

    LEVEL_1 = "l1"
    LEVEL_2 = "l2"
    DEPTH_10 = "d10"
    RANDOM = "rnd"
    LEVEL_1x = "l1x"

    LEVELS = [
        LEVEL_1,
        LEVEL_2,
    ]

    SPECIALS = [
        DEPTH_10,
        RANDOM,
        LEVEL_1x,
    ]

    TWO_STEPS = [
        LEVEL_1,
        LEVEL_2,
        LEVEL_1x,
    ]

    TRANSFORMERS = {
        LEVEL_1: Level1Transformer,
        LEVEL_2: Level2Transformer,
        DEPTH_10: Depth10Transformer,
        RANDOM: BackendSexpRandomTransformer,
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
