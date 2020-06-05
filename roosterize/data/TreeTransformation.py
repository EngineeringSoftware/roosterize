from typing import *

from recordclass import RecordClass

from roosterize.sexp import *


class TreeTransformationConsts:
    KEEP = "K"
    REMOVE = "R"
    UPDATE = "U"


class TreeTransformation(RecordClass):
    transform_tree: SexpNode = None

    def transform_sexp(self, sexp: SexpNode) -> Optional[SexpNode]:
        """
        Transforms the sexp according to this transformation.
        :param sexp: the sexp to be changed. The original nodes in this sexp might be changed during the transformation.
        :return: the transformed sexp, or None if the transformation reduce the tree to nothing.
        """
        return self.transform_sexp_recur(sexp, self.transform_tree)

    @classmethod
    def transform_sexp_recur(cls, sexp: SexpNode, transform_tree: SexpNode) -> Optional[SexpNode]:
        if sexp.is_string() and transform_tree.is_string():
            if transform_tree.content == TreeTransformationConsts.KEEP:
                return sexp
            elif transform_tree.content[0] == TreeTransformationConsts.UPDATE:
                return SexpString(transform_tree.content[1:])
            else:  # REMOVE
                return None
            # end if
        elif sexp.is_list() and transform_tree.is_string():
            if transform_tree.content == TreeTransformationConsts.KEEP:
                return sexp
            elif transform_tree.content == TreeTransformationConsts.REMOVE:
                return None
            # end if
            # UPDATE is invalid here
        elif sexp.is_list() and transform_tree.is_list() and len(sexp) == len(transform_tree):
            new_children: List[SexpNode] = list()
            for child_i in range(len(sexp)):
                transformed_child = cls.transform_sexp_recur(sexp[child_i], transform_tree[child_i])
                if transformed_child is not None:  new_children.append(transformed_child)
            # end for

            # Abandon empty list
            if len(new_children) == 0:  return None

            # Squeeze singleton
            if len(new_children) == 1:  return new_children[0]

            return SexpList(new_children)
        # end if

        raise Exception(f"The sexp and keep_tree doesn't match!\n"
                        f"at sexp: {sexp.pretty_format(3)}\n"
                        f"at transform_tree: {transform_tree.pretty_format(3)}\n")
