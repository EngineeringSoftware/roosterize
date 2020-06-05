from typing import *

import abc
from collections import deque
from enum import Enum
import numpy as np
import sys

from roosterize.sexp.IllegalSexpOperationException import IllegalSexpOperationException


class SexpNode:
    """
    Abstract class of a node in sexp
    """
    @abc.abstractmethod
    def to_python_ds(self) -> Union[str, list]:
        """
        Converts this s-expression to python lists and strings.
        """

    def is_list(self) -> bool:
        """
        Checks if this node is a list.
        :return: True if this node is a list.
        """
        return False

    def is_string(self) -> bool:
        """
        Checks if this node is a string.
        :return: True if this node is a string.
        """
        return False

    def get_content(self) -> Optional[str]:
        """
        Gets the content of this (string) node.
        :return: self's content if this is a string node, otherwise None.
        """
        return None

    @property
    def content_no_quote(self):
        content = self.content
        if content.startswith('"'):  content = content[1:-1]
        return content

    def get_children(self) -> Optional[List["SexpNode"]]:
        """
        Gets the children of this (list) node.
        :return: self's children if this is a list node, otherwise None.
        """
        return None

    @property
    def content(self):
        """
        Gets the content of the SexpString, or throw exception.
        :return: the content, if it is an SexpString.
        :raises IllegalSexpOperationException: if it is an SexpList.
        """
        content = self.get_content()
        if content is None:
            raise IllegalSexpOperationException("Cannot get the content of an s-exp list.")
        else:
            return content
        # end if

    def __len__(self):
        """
        Gets the length of this SexpNode, which is always 0 when it is an SexpString.
        :return: the length of the list when it is an SexpList, or 0 if it is an SexpString.
        """
        if self.is_list():
            return len(self.get_children())
        else:
            return 0

    def __getitem__(self, index):
        """
        Gets the index-th child node if it is an SexpList, or throw exception.
        :param index: the index of the child node to get.
        :return: the index-th child, if it is an SexpList and index is valid.
        :raises IllegalSexpOperationException: if it is an SexpString or does not have enough children, or index < 0.
        """
        children = self.get_children()
        if children is None:
            raise IllegalSexpOperationException("Cannot get the children of an s-exp string.")
        elif isinstance(index, int):
            if index < -len(children) or index >= len(children):
                raise IllegalSexpOperationException(f"Cannot get child ({index}), this list only have {len(children)} children.")
            # end if
        # end if

        return children[index]

    @abc.abstractmethod
    def __deepcopy__(self, memodict={}):
        raise NotImplementedError

    class RecurAction(Enum):
        ContinueRecursion = 0
        StopRecursion = 1

    @abc.abstractmethod
    def apply_recur(self, func: Callable[["SexpNode"], RecurAction]) -> NoReturn:
        """
        Recursively visits (in depth first search order) each node in the sexp and applying func.
        :param func: the function to apply, takes in an SexpNode and returns RecurAction to specify if continue explore on this branch or not.
        """
        return NotImplemented

    @abc.abstractmethod
    def modify_recur(self,
            pre_children_modify: Callable[["SexpNode"], Tuple[Optional["SexpNode"], "SexpNode.RecurAction"]] = lambda x: (x, SexpNode.RecurAction.ContinueRecursion),
            post_children_modify: Callable[["SexpNode"], Optional["SexpNode"]] = lambda x: x,
    ) -> Optional["SexpNode"]:
        """
        Recursively visits (in depth first search order) each node in the sexp, and modify the sexp.
        :param pre_children_modify: the function that should be applied prior to applying the modification on children.
        :param post_children_modify: the function that should be applied after applying the modification on children.
        :return: the modified sexp to replace this sexp node, or None if deleting this node from parent list.
        """
        return NotImplemented

    @abc.abstractmethod
    def height(self) -> int:
        return NotImplemented

    @abc.abstractmethod
    def num_nodes(self) -> int:
        return NotImplemented

    @abc.abstractmethod
    def num_leaves(self) -> int:
        return NotImplemented

    @abc.abstractmethod
    def contains_str(self, s: str) -> bool:
        return NotImplemented

    @abc.abstractmethod
    def __str__(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def pretty_format(self, max_depth: int = np.PINF) -> str:
        """
        Formats this s-expression into an human-readable string.
        :return: a pretty human-readable string for this s-expression.
        """
        return NotImplemented

    def dot(self) -> str:
        """
        Returns the visualization in dot format.
        Generate pdf with: `dot -Tpdf $file -o $pdfFile`.
        """
        out = ""
        out += "digraph x {"
        toVisit: Deque[SexpNode] = deque()
        toVisit.append(self)
        while len(toVisit) > 0:
            currentSexp: SexpNode = toVisit.popleft()
            if currentSexp.is_string():
                label = currentSexp.content.replace('"', '\'')
                out += f"n{hash(currentSexp)% ((sys.maxsize + 1) * 2)} [label=\"{label}\" shape=none];\n"
            else:
                out += f"n{hash(currentSexp)% ((sys.maxsize + 1) * 2)} [shape=point];\n"
                for child in currentSexp.get_children():
                    toVisit.append(child)
                    out += f"n{hash(currentSexp)% ((sys.maxsize + 1) * 2)} -> n{hash(child)% ((sys.maxsize + 1) * 2)};\n"
                # end for
            # end if
        # end while
        out += "}\n"

        return out

    @abc.abstractmethod
    def forward_depth_first_sequence(self,
            children_filtering_func: Callable[[Iterable["SexpNode"]], Iterable["SexpNode"]] = lambda x: x,
            use_parathesis: bool = False,
    ) -> List[str]:
        return NotImplemented

    @abc.abstractmethod
    def backward_depth_first_sequence(self,
            children_filtering_func: Callable[[Iterable["SexpNode"]], Iterable["SexpNode"]] = lambda x: x,
            use_parathesis: bool = False,
    ) -> List[str]:
        return NotImplemented

    def jsonfy(self):
        return self.__str__()

    @classmethod
    def dejsonfy(cls, data):
        from roosterize.sexp.SexpParser import SexpParser
        return SexpParser.parse(data)
