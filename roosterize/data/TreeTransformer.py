from typing import *

import abc
import random

from roosterize.data.TreeTransformation import TreeTransformationConsts
from roosterize.sexp import *


class TreeTransformer:

    def __init__(self):
        return

    @abc.abstractmethod
    def transform(self, sexp: SexpNode) -> SexpNode:
        raise NotImplementedError


class KeepAllTreeTransformer(TreeTransformer):

    def __init__(self):
        super().__init__()
        return

    def transform(self, sexp: SexpNode) -> SexpNode:
        return SexpString(TreeTransformationConsts.KEEP)


class Depth10Transformer(TreeTransformer):
    """
    Removes all nodes after depth 10.
    """

    def __init__(self):
        super().__init__()
        return

    def transform(self, sexp: SexpNode, depth: int = 1) -> SexpNode:
        if depth > 10:
            return None
        else:
            if sexp.is_list():
                transformed_children = [self.transform(c, depth+1) for c in sexp.get_children()]
                transformed_children = [c for c in transformed_children if c is not None]
                return SexpList(transformed_children)
            else:
                return sexp
            # end if
        # end if


class RandomTransformer(TreeTransformer):
    """
    Removes all nodes randomly, until reducing the nodes size to the target fraction.
    """

    def __init__(self, target_size_fraction: float):
        super().__init__()
        self.target_size_fraction = target_size_fraction
        return

    def transform(self, sexp: SexpNode, is_root: bool = True) -> SexpNode:
        # Each node, except root, gets a chance to be removed
        # If a list is removed, its children are connected to the parent list
        keep = is_root or random.random() < self.target_size_fraction

        if sexp.is_list():
            transformed_children = list()
            for c in sexp.get_children():
                transformed_c = self.transform(c, False)
                if transformed_c is None:
                    continue
                elif isinstance(transformed_c, list):
                    transformed_children.extend(transformed_c)
                else:
                    transformed_children.append(transformed_c)
                # end if
            # end for

            if keep:
                return SexpList(transformed_children)
            else:
                return transformed_children
            # end if
        else:  # is_string
            if keep:
                return sexp
            else:
                return None
            # end if
        # end if
