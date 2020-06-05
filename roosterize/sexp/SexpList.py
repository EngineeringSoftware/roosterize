from typing import *

import copy
import numpy as np

from roosterize.sexp.SexpNode import SexpNode


class SexpList(SexpNode):

    def __init__(self, children: List[SexpNode] = None):
        self.children = children if children is not None else list()
        return

    def __deepcopy__(self, memodict={}):
        return SexpList([copy.deepcopy(c) for c in self.children])

    def to_python_ds(self) -> list:
        return [child.to_python_ds() for child in self.children]

    def is_list(self):
        return True

    def get_children(self):
        return self.children

    def apply_recur(self, func: Callable[["SexpNode"], SexpNode.RecurAction]) -> NoReturn:
        recur_action = func(self)

        if recur_action == SexpNode.RecurAction.ContinueRecursion:
            for child in self.children:
                child.apply_recur(func)
            # end for
        # end if
        return

    def modify_recur(self,
            pre_children_modify: Callable[["SexpNode"], Tuple[Optional["SexpNode"], SexpNode.RecurAction]] = lambda x: (x, SexpNode.RecurAction.ContinueRecursion),
            post_children_modify: Callable[["SexpNode"], Optional["SexpNode"]] = lambda x: x,
    ) -> Optional["SexpNode"]:
        sexp, recur_action = pre_children_modify(self)

        if sexp is None:  return None

        if sexp.is_list() and recur_action == SexpNode.RecurAction.ContinueRecursion:
            child_i = 0
            while child_i < len(sexp.get_children()):
                new_child = sexp.get_children()[child_i].modify_recur(pre_children_modify, post_children_modify)
                if new_child is None:
                    del sexp.get_children()[child_i]
                else:
                    sexp.get_children()[child_i] = new_child
                    child_i += 1
                # end if
            # end for
        # end if

        sexp = post_children_modify(sexp)
        return sexp

    def height(self) -> int:
        return max([c.height() for c in self.children] + [0]) + 1

    def num_nodes(self) -> int:
        return sum([c.num_nodes() for c in self.children]) + 1

    def num_leaves(self) -> int:
        return sum([c.num_leaves() for c in self.children])

    def contains_str(self, s: str) -> bool:
        for c in self.children:
            if c.contains_str(s):
                return True
            # end if
        # end for
        return False

    def forward_depth_first_sequence(self,
            children_filtering_func: Callable[[Iterable["SexpNode"]], Iterable["SexpNode"]] = lambda x: x,
            use_parathesis: bool = False,
    ) -> List[str]:
        core = [t for c in children_filtering_func(self.children) for t in c.forward_depth_first_sequence(children_filtering_func, use_parathesis)]
        if use_parathesis:
            return ["("] + core + [")"]
        else:
            return core
        # end if

    def backward_depth_first_sequence(self,
            children_filtering_func: Callable[[Iterable["SexpNode"]], Iterable["SexpNode"]] = lambda x: x,
            use_parathesis: bool = False,
    ) -> List[str]:
        core = [t for c in children_filtering_func(reversed(self.children)) for t in c.backward_depth_first_sequence(children_filtering_func, use_parathesis)]
        if use_parathesis:
            return ["("] + core + [")"]
        else:
            return core
        # end if

    pprint_newline = "\n"
    pprint_tab = "  "

    def __str__(self) -> str:
        s = "("
        last_is_str = False
        for c in self.children:
            # Put space only between SexpString
            if c.is_string():
                if last_is_str:
                    s += " "
                # end if
                last_is_str = True
            # end if

            s += c.__str__()
        # end for
        s += ")"
        return s

    def pretty_format(self, max_depth: int = np.PINF) -> str:
        return self.pretty_format_recur(self, max_depth, 0).strip()

    @classmethod
    def pretty_format_recur(cls, sexp: SexpNode, max_depth: int, depth: int):
        if sexp.is_string():
            return sexp.pretty_format()
        # end if

        sexp: SexpList
        if len(sexp.children) == 0:
            return "()"
        else:
            if max_depth == 0:
                return " ... "
            else:
                return cls.pprint_newline + depth * cls.pprint_tab + \
                       "(" + " ".join([cls.pretty_format_recur(c, max_depth-1, depth+1) for c in sexp.children]) + ")"
            # end if
        # end if
