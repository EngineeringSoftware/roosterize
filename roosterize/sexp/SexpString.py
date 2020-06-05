from typing import *

import numpy as np

from roosterize.sexp.SexpNode import SexpNode


class SexpString(SexpNode):

    def __init__(self, content: str = None):
        self._content = content if content is not None else ""
        return

    def __deepcopy__(self, memodict={}):
        return SexpString(self.content)

    def to_python_ds(self) -> str:
        return self._content

    def is_string(self):
        return True

    def get_content(self):
        return self._content

    def apply_recur(self, func: Callable[["SexpNode"], SexpNode.RecurAction]) -> NoReturn:
        func(self)
        return

    def modify_recur(self,
            pre_children_modify: Callable[["SexpNode"], Tuple[Optional["SexpNode"], SexpNode.RecurAction]] = lambda x: (x, SexpNode.RecurAction.ContinueRecursion),
            post_children_modify: Callable[["SexpNode"], Optional["SexpNode"]] = lambda x: x,
    ) -> Optional["SexpNode"]:
        sexp, recur_action = pre_children_modify(self)
        if sexp is None:  return None
        sexp = post_children_modify(sexp)
        return sexp

    def height(self) -> int:
        return 0

    def num_nodes(self) -> int:
        return 1

    def num_leaves(self) -> int:
        return 1

    def contains_str(self, s: str) -> bool:
        return self._content == s

    def forward_depth_first_sequence(self,
            children_filtering_func: Callable[[List["SexpNode"]], List["SexpNode"]] = lambda x: x,
            use_parathesis: bool = False,
    ) -> List[str]:
        return [self.content]

    def backward_depth_first_sequence(self,
            children_filtering_func: Callable[[List["SexpNode"]], List["SexpNode"]] = lambda x: x,
            use_parathesis: bool = False,
    ) -> List[str]:
        return [self.content]

    def __str__(self) -> str:
        content = self.content
        content = content.replace("\\", "\\\\")  # Escape back slash (\)
        content = content.replace('"', '\\"')  # Escape quote (")
        if " " in content:
            content = '"' + content + '"'
        # end if
        return content

    def pretty_format(self, max_depth: int = np.PINF) -> str:
        return self._content
