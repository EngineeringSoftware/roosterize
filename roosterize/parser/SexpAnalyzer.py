from typing import *

import collections
import re
from recordclass import RecordClass

from seutil import LoggingUtils

from roosterize.data.Token import TokenConsts
from roosterize.parser.ParserUtils import ParserUtils
from roosterize.sexp import *


class SexpInfo:
    """
    Defines varies structs that may be results of SexpAnalyzer.
    """
    class Vernac(RecordClass):
        vernac_type: str = ""
        extend_type: str = ""
        vernac_sexp: SexpNode = None
        loc: "SexpInfo.Loc" = None

    class VernacConsts:
        type_abort = "VernacAbort"
        type_add_option = "VernacAddOption"
        type_arguments = "VernacArguments"
        type_assumption = "VernacAssumption"
        type_begin_section = "VernacBeginSection"
        type_bullet = "VernacBullet"
        type_bind_scope = "VernacBindScope"
        type_canonical = "VernacCanonical"
        type_chdir = "VernacChdir"
        type_check_may_eval = "VernacCheckMayEval"
        type_coercion = "VernacCoercion"
        type_combined_scheme = "VernacCombinedScheme"
        type_context = "VernacContext"
        type_declare_module_type = "VernacDeclareModuleType"
        type_define_module = "VernacDefineModule"
        type_definition = "VernacDefinition"
        type_delimiters = "VernacDelimiters"
        type_end_proof = "VernacEndProof"
        type_end_segment = "VernacEndSegment"
        type_end_subproof = "VernacEndSubproof"
        type_exact_proof = "VernacExactProof"
        type_existing_instance = "VernacExistingInstance"
        type_extend = "VernacExtend"
        type_fail = "VernacFail"
        type_fixpoint = "VernacFixpoint"
        type_hints = "VernacHints"
        type_identify_coercion = "VernacIdentityCoercion"
        type_import = "VernacImport"
        type_include = "VernacInclude"
        type_inductive = "VernacInductive"
        type_infix = "VernacInfix"
        type_instance = "VernacInstance"
        type_notation = "VernacNotation"
        type_open_close_scope = "VernacOpenCloseScope"
        type_print = "VernacPrint"
        type_proof = "VernacProof"
        type_remove_hints = "VernacRemoveHints"
        type_require = "VernacRequire"
        type_reserve = "VernacReserve"
        type_scheme = "VernacScheme"
        type_set_opacity = "VernacSetOpacity"
        type_set_option = "VernacSetOption"
        type_start_theorem_proof = "VernacStartTheoremProof"
        type_subproof = "VernacSubproof"
        type_syntactic_definition = "VernacSyntacticDefinition"
        type_syntax_extension = "VernacSyntaxExtension"

        extend_type_obligations = "Obligations"

    class Loc(RecordClass):
        filename: str
        lineno: int
        beg_charno: int
        end_charno: int

        def __lt__(self, other):
            if isinstance(other, type(self)):
                return self.end_charno <= other.beg_charno
            elif isinstance(other, (int, float)):
                return self.end_charno <= other
            else: raise ValueError

        def __gt__(self, other):
            if isinstance(other, type(self)):
                return self.beg_charno >= other.end_charno
            elif isinstance(other, (int, float)):
                return self.beg_charno >= other
            else: raise ValueError

        def __eq__(self, other):
            if isinstance(other, type(self)):
                return (self.beg_charno, self.end_charno, self.filename, self.lineno) == (other.beg_charno, other.end_charno, other.filename, other.lineno)
            else: raise ValueError

        def __hash__(self):
            return hash((self.filename, self.lineno, self.beg_charno, self.end_charno))

        def contains_charno_range(self, beg_charno: int, end_charno: int):
            if self.beg_charno <= beg_charno and self.end_charno >= end_charno:
                return True
            else:
                return False
            # end if

    class SertokSentence:
        tokens: List["SexpInfo.SertokToken"] = None

    class SertokToken:
        kind: str = ""
        content: str = ""
        loc: "SexpInfo.Loc" = None

    class ConstrExprR(RecordClass):
        expr_type: str = ""
        expr_sexp: SexpNode = None
        claimed_loc: "SexpInfo.Loc" = None
        loc: "SexpInfo.Loc" = None

        def __hash__(self):
            return hash((self.expr_sexp, self.loc))

    class ConstrExprRConsts:
        type_c_ref = "CRef"
        type_c_fix = "CFix"
        type_c_co_fix = "CCoFix"
        type_c_prod_n = "CProdN"
        type_c_lambda_n = "CLambdaN"
        type_c_let_in = "CLetIn"
        type_c_app_expl = "CAppExpl"
        type_c_app = "CApp"
        type_c_record = "CRecord"
        type_c_cases = "CCases"
        type_c_let_tuple = "CLetTuple"
        type_c_if = "CIf"
        type_c_hole = "CHole"
        type_c_pat_var = "CPatVar"
        type_c_evar = "CEvar"
        type_c_sort = "CSort"
        type_c_cast = "CCast"
        type_c_notation = "CNotation"
        type_c_generalization = "CGeneralization"
        type_c_prim = "CPrim"
        type_c_delimiters = "CDelimiters"

        types = [
            type_c_ref,
            type_c_fix,
            type_c_co_fix,
            type_c_prod_n,
            type_c_lambda_n,
            type_c_let_in,
            type_c_app_expl,
            type_c_app,
            type_c_record,
            type_c_cases,
            type_c_let_tuple,
            type_c_if,
            type_c_hole,
            type_c_pat_var,
            type_c_evar,
            type_c_sort,
            type_c_cast,
            type_c_notation,
            type_c_generalization,
            type_c_prim,
            type_c_delimiters,
        ]

    class CNotation(RecordClass):
        notation_shape: str = ""
        expr_sexp: SexpNode = None
        loc: "SexpInfo.Loc" = None
        args: List["SexpInfo.ConstrExprR"] = None
        notation_symbols: List[str] = None

    class CLocalAssum(RecordClass):
        sexp: SexpNode = None
        loc: "SexpInfo.Loc" = None
        loc_part_1: "SexpInfo.Loc" = None
        constr_expr_r: "SexpInfo.ConstrExprR" = None
        is_one_token: bool = False


class SexpAnalyzingException(Exception):

    def __init__(self, sexp: SexpNode = None, message: str = "", *args, **kwargs):
        self.sexp = sexp
        self.message = message

    def __str__(self):
        return f"{self.message}\nin sexp: {self.sexp.pretty_format()}"


class SexpAnalyzer:
    """
    Analyzing the parsed sexp of a Coq document, and retrieve varies information.
    """
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG)

    @classmethod
    def analyze_vernac(cls, sexp: SexpNode) -> SexpInfo.Vernac:
        """
        Analyzes an s-expression and parses it as a Vernac expression, gets type of the expression and source code location of the expression.

        Accepts s-expression:
        <sexp_vernac> = ( ( v (VernacExpr (...) ( <TYPE>  ... )) ) <sexp_loc> )
                              ^----------vernac_sexp-----------^
        :raises: SexpAnalyzingException if the sexp cannot be parsed that way.
        """
        try:
            if len(sexp) != 2: raise SexpAnalyzingException(sexp)

            v_child = sexp[0]
            loc_child = sexp[1]

            loc = cls.analyze_loc(loc_child)

            extend_type = ""

            if v_child[0].content == "v" and v_child[1][0].content == "VernacExpr":
                # ( v (VernacExpr()  (   <TYPE>  ... )) )
                #   0 1  1,0     1,1 1,2 1,2,0
                # ( v (VernacExpr()  <TYPE> ) )
                #                    1,2
                if v_child[1][2].is_list():
                    vernac_type = v_child[1][2][0].content
                    if vernac_type == SexpInfo.VernacConsts.type_extend:
                        # ( v (VernacExpr() ( VernacExtend  ( <EXTEND_TYPE> ... ) ...
                        extend_type = v_child[1][2][1][0].content
                    # end if
                else:
                    vernac_type = v_child[1][2].content
                # end if
            elif v_child[0].content == "v" and v_child[1][0].content == "VernacFail":
                # v_child
                # ( v (VernacFail ( ( v (VernacExpr () ( ...
                #   0 1  1,0
                vernac_type = "VernacFail"
            else:
                raise SexpAnalyzingException(sexp)
            # end if

            return SexpInfo.Vernac(vernac_type=vernac_type, extend_type=extend_type, vernac_sexp=v_child[1], loc=loc)
        except IllegalSexpOperationException:
            raise SexpAnalyzingException(sexp)

    @classmethod
    def analyze_constr_expr_r(cls, sexp: SexpNode, unicode_offsets: Optional[List[int]] = None) -> SexpInfo.ConstrExprR:
        """
        Analyzes a ConstrExprR sexp.
        <ConstrExprR> = ( ( v  (  CXxx ... ) ) (  loc ... )
                               ^-expr_sexp-^   ^-expr_loc-^
                          0 00 01 010          1  10
        """
        if sexp[0][0].content == "v" and sexp[0][1][0].content in SexpInfo.ConstrExprRConsts.types:
            loc_child = sexp[1]
            claimed_loc = cls.analyze_loc(loc_child, unicode_offsets)
            expr_type = sexp[0][1][0].content
            expr_sexp = sexp

            if expr_type == SexpInfo.ConstrExprRConsts.type_c_cast:
                # Find all nested loc
                locs: List[SexpInfo.Loc] = list()
                def find_all_loc(sexp_part: SexpNode) -> SexpNode.RecurAction:
                    nonlocal locs
                    try:
                        locs.append(cls.analyze_loc(sexp_part, unicode_offsets))
                        return SexpNode.RecurAction.ContinueRecursion
                    except (IllegalSexpOperationException, SexpAnalyzingException):
                        return SexpNode.RecurAction.ContinueRecursion
                    # end try
                # end def

                sexp.apply_recur(find_all_loc)

                loc = SexpInfo.Loc(
                    filename=claimed_loc.filename,
                    lineno=claimed_loc.lineno,
                    beg_charno=min([l.beg_charno for l in locs]),
                    end_charno=max([l.end_charno for l in locs]),
                )
            else:
                loc = claimed_loc
            # end if

            return SexpInfo.ConstrExprR(expr_type=expr_type, expr_sexp=expr_sexp, claimed_loc=claimed_loc, loc=loc)
        else:
            raise SexpAnalyzingException(sexp)

    @classmethod
    def analyze_c_local_assume(cls, sexp: SexpNode, unicode_offsets: Optional[List[int]] = None) -> SexpInfo.CLocalAssum:
        """
        Analyzes a CLocalAssume sexp.
        <CLocalAssume> = ( CLocalAssum ((   (v ...) (loc ...) )) ... ( (v (CXxx ...) ... ) (loc ...) ) )
                                                    ^-<loc_part_1>   ^---<ConstrExprR>---------------^
                           0            10 100     101           1   2
        """
        if sexp[0].content == "CLocalAssum" and sexp[1][0][0][0].content == "v":
            loc_part_1 = cls.analyze_loc(sexp[1][0][1], unicode_offsets)
            constr_expr_r: SexpInfo.ConstrExprR = cls.analyze_constr_expr_r(sexp[2], unicode_offsets)
            is_one_token = (loc_part_1 == constr_expr_r.loc)
            loc = SexpInfo.Loc(
                filename=loc_part_1.filename,
                lineno=loc_part_1.lineno,
                beg_charno=min([loc_part_1.beg_charno, constr_expr_r.loc.beg_charno]),
                end_charno=max([loc_part_1.end_charno, constr_expr_r.loc.end_charno]),
            )

            c_local_assume = SexpInfo.CLocalAssum(
                sexp=sexp,
                loc=loc,
                loc_part_1=loc_part_1,
                constr_expr_r=constr_expr_r,
                is_one_token=is_one_token
            )
            return c_local_assume
        else:
            raise SexpAnalyzingException(sexp)

    @classmethod
    def find_gallina_parts(cls, sexp: SexpNode, unicode_offsets: Optional[List[int]] = None) -> List[Union[SexpInfo.ConstrExprR, SexpInfo.CLocalAssum]]:
        """
        Analyzes a sexp (e.g., a vernacular sentence or L_tac sentence), finds all Gallina parts (ConstExprR or CLocalAssum).
        """
        try:
            cs_parts: List[Union[SexpInfo.ConstrExprR, SexpInfo.CLocalAssum]] = list()

            def find_cs_parts(sexp_part: SexpNode) -> SexpNode.RecurAction:
                nonlocal cs_parts
                try:
                    if sexp_part[0][0].content == "v" and sexp_part[0][1][0].content in SexpInfo.ConstrExprRConsts.types:
                        # ( ( v  (  CXxx ... ) )  ....
                        cs_parts.append(cls.analyze_constr_expr_r(sexp_part, unicode_offsets))
                        return SexpNode.RecurAction.StopRecursion
                    elif sexp_part[0].content == "CLocalAssum" and sexp_part[1][0][0][0].content == "v":
                        # ( CLocalAssum (( (v ...) ....
                        cs_parts.append(cls.analyze_c_local_assume(sexp_part, unicode_offsets))
                        return SexpNode.RecurAction.StopRecursion
                    else:
                        return SexpNode.RecurAction.ContinueRecursion
                    # end if
                except (IllegalSexpOperationException, SexpAnalyzingException):
                    return SexpNode.RecurAction.ContinueRecursion
                # end try
            # end def

            sexp.apply_recur(find_cs_parts)
            cs_parts.sort(key=lambda e: e.loc)

            return cs_parts
        except IllegalSexpOperationException:
            raise SexpAnalyzingException(sexp)

    RE_NOTATION_SHAPE = re.compile(r"(^|(?<= ))_(?=$| )")
    RE_NOTATION_REC = re.compile(r"(^|(?<= ))_ (?P<rec>(\S* )?)\.\. (?P=rec)_(?=$| )")

    @classmethod
    def find_c_notations(cls, sexp: SexpNode) -> List[SexpInfo.CNotation]:
        """
        Analyzes an Gallina-only sexp, finds all CNotations.
        """
        try:
            c_notations: List[SexpInfo.CNotation] = list()
            seen_locs: Set[SexpInfo.Loc] = set()

            def find_c_notation_parts(sexp_part: SexpNode) -> SexpNode.RecurAction:
                nonlocal c_notations
                try:
                    # ( ( v  ( CNotation ... ) ) (  loc ... )
                    #        ^---expr_sexp---^   ^-expr_loc-^
                    #   0 00 01 010              1  10
                    if sexp_part[0][0].content == "v" and sexp_part[0][1][0].content == SexpInfo.ConstrExprRConsts.type_c_notation:
                        loc_child = sexp_part[1]
                        loc = cls.analyze_loc(loc_child)

                        # Only parse unseen CNotations
                        if loc in seen_locs:
                            return SexpNode.RecurAction.StopRecursion
                        # end if
                        seen_locs.add(loc)

                        expr_sexp = sexp_part

                        # ... (  CNotation (   XLevel Shape ) (   ( A1 A2 .. ) ( .. ) ( .. ) ( .. ) )
                        #     01 010       011 0110   0111      012 0120         0121   0122   0123
                        notation_shape = expr_sexp[0][1][1][1].content
                        notation_symbols = list()
                        notation_recur_idx = -1
                        notation_recur_symbol = None
                        if notation_shape[0] == '"' and notation_shape[-1] == '"':
                            # Notation with arguments
                            notation_shape = notation_shape[1:-1]  # [1:-1] is to remove preceding and trailing "
                            rec_match = cls.RE_NOTATION_REC.search(notation_shape)
                            if rec_match is None:
                                # Notation without recursive pattern
                                notation_symbols.extend(cls.RE_NOTATION_SHAPE.split(notation_shape)[::2])  # [::2] is to remove the split points and only keep symbols
                            else:
                                # Notation with recursive pattern: ".." and the separators are removed
                                notation_recur_symbol = rec_match.group("rec")
                                notation_symbols.extend(cls.RE_NOTATION_SHAPE.split(notation_shape[:rec_match.start()])[::2])  # [::2] is to remove the split points and only keep symbols
                                notation_recur_idx = len(notation_symbols)
                                notation_symbols.extend(cls.RE_NOTATION_SHAPE.split(notation_shape[rec_match.end():])[::2])  # [::2] is to remove the split points and only keep symbols
                            # end if
                            num_args = len(notation_symbols) - 1
                        else:
                            # Notation without argument
                            notation_symbols = [notation_shape]
                            num_args = 0
                        # end if

                        args_sexps: List[SexpNode] = list()
                        for i in range(4):
                            args_sexps.extend(expr_sexp[0][1][2][i].get_children())
                        # end for
                        if notation_recur_idx == -1:
                            # No recursive pattern: try to match num_args with len(args_sexps)
                            if num_args != len(args_sexps):
                                cls.logger.warning(f"Notation: num of args doesnot match: {num_args} (in {notation_symbols}) != {len(args_sexps)} (in sexp {sexp_part.pretty_format()})")
                                raise SexpAnalyzingException(sexp, f"num of args doesnot match: {num_args} (in {notation_symbols}) != {len(args_sexps)} (in sexp)")
                            # end if
                        else:
                            # Recursive pattern: use len(arg_sexps) to imply num_args
                            for i in range(len(args_sexps) - num_args): notation_symbols.insert(notation_recur_idx, notation_recur_symbol)
                            num_args = len(args_sexps)
                        # end if

                        args: List[SexpInfo.ConstrExprR] = [cls.analyze_constr_expr_r(arg_sexp) for arg_sexp in args_sexps]
                        args.sort(key=lambda a: a.loc)

                        c_notations.append(SexpInfo.CNotation(expr_sexp=expr_sexp, loc=loc, notation_shape=notation_shape, args=args, notation_symbols=notation_symbols))
                        return SexpNode.RecurAction.StopRecursion
                    else:
                        return SexpNode.RecurAction.ContinueRecursion
                    # end if
                except (IllegalSexpOperationException, SexpAnalyzingException):
                    return SexpNode.RecurAction.ContinueRecursion
                # end try
            # end def

            sexp.apply_recur(find_c_notation_parts)

            return c_notations
        except IllegalSexpOperationException:
            raise SexpAnalyzingException(sexp)

    @classmethod
    def analyze_loc(cls, sexp: SexpNode, unicode_offsets: Optional[List[int]] = None) -> SexpInfo.Loc:
        """
        Analyzes an loc s-expression and gets source code location information.

        Accepts s-expression:
        <sexp_loc> = ( loc (( (fname(InFile <FILENAME>)) (line_nb X) (bol_pos X) (line_nb_last <LINENO>) (bol_pos_last X) (bp <BEG_CHARNO>) (ep <END_CHARNO>) )) )

        :raises: SexpAnalyzingException if the sexp cannot be parsed that way.
        """
        try:
            if len(sexp) != 2:
                raise SexpAnalyzingException(sexp)
            # end if

            if sexp[0].content != "loc":
                raise SexpAnalyzingException(sexp)
            # end if

            data_child = sexp[1][0]
            if len(data_child) != 7:
                raise SexpAnalyzingException(sexp)
            # end if

            filename_child = data_child[0]
            if filename_child[0].content != "fname":
                raise SexpAnalyzingException(sexp)
            # end if
            filename = filename_child[1][1].content

            lineno_last_child = data_child[3]
            if lineno_last_child[0].content != "line_nb_last":
                raise SexpAnalyzingException(sexp)
            # end if
            lineno = int(lineno_last_child[1].content)

            bp_child = data_child[5]
            if bp_child[0].content != "bp":
                raise SexpAnalyzingException(sexp)
            # end if
            beg_charno = int(bp_child[1].content)

            ep_child = data_child[6]
            if ep_child[0].content != "ep":
                raise SexpAnalyzingException(sexp)
            # end if
            end_charno = int(ep_child[1].content)

            if unicode_offsets is not None:
                beg_charno = ParserUtils.coq_charno_to_actual_charno(beg_charno, unicode_offsets)
                end_charno = ParserUtils.coq_charno_to_actual_charno(end_charno, unicode_offsets)
            # end if

            return SexpInfo.Loc(filename=filename, lineno=lineno, beg_charno=beg_charno, end_charno=end_charno)
        except IllegalSexpOperationException:
            raise SexpAnalyzingException(sexp)

    @classmethod
    def analyze_sertok_sentences(cls, tok_sexp_list: List[SexpNode], unicode_offsets: Optional[List[int]] = None) -> List[SexpInfo.SertokSentence]:
        sentences: List[SexpInfo.SertokSentence] = list()

        for sertok_sentence_sexp in tok_sexp_list:
            # ( Sentence  ( tok ... ) )
            #   0         1
            try:
                if sertok_sentence_sexp[0].content != "Sentence":  raise SexpAnalyzingException(sertok_sentence_sexp, "Not a valid SertokSentence sexp")

                sentence = SexpInfo.SertokSentence()
                sentence.tokens = list()
                for sertok_token_sexp in sertok_sentence_sexp[1].get_children():
                    sentence.tokens.append(cls.analyze_sertok_token(sertok_token_sexp, unicode_offsets))
                # end for

                sentences.append(sentence)
            except IllegalSexpOperationException:
                raise SexpAnalyzingException(sertok_sentence_sexp)
            # end try
        # end for

        return sentences

    SERTOK_TOKEN_KIND_MAPPING = {
        "BULLET": TokenConsts.KIND_SYMBOL,
        "IDENT": TokenConsts.KIND_ID,
        "KEYWORD": TokenConsts.KIND_KEYWORD,
        "LEFTQMARK": TokenConsts.KIND_SYMBOL,
        "NUMERAL": TokenConsts.KIND_NUMBER,
        "STRING": TokenConsts.KIND_STR,
    }

    @classmethod
    def analyze_sertok_token(cls, sexp: SexpNode, unicode_offsets: Optional[List[int]] = None) -> SexpInfo.SertokToken:
        # ( ( v  (  <KIND> <CONTENT> )) ( loc ... ) ) )
        #   0 00 01 010    011          1
        try:
            if sexp[0][0].content != "v":  raise SexpAnalyzingException(sexp)

            sertok_token = SexpInfo.SertokToken()

            # Kind and content
            if sexp[0][1].is_list():
                sertok_token.kind = sexp[0][1][0].content
                if sertok_token.kind == "NUMERAL":
                    # ( ( int ? ) (frac ? ) (exp ? ) )
                    #   0         1         2
                    if len(sexp[0][1][1][0][1].content) > 0:
                        sertok_token.content = sexp[0][1][1][0][1].content
                    elif len(sexp[0][1][1][1][1].content) > 0:
                        sertok_token.content = sexp[0][1][1][1][1].content
                    elif len(sexp[0][1][1][2][1].content) > 0:
                        sertok_token.content = sexp[0][1][1][2][1].content
                    else:
                        raise SexpAnalyzingException(sexp[0][1][1], message="Unknown numeral")
                    # end if
                else:
                    sertok_token.content = sexp[0][1][1].content
                # end if
            else:
                sertok_token.kind = sexp[0][1].content
                if sertok_token.kind == "LEFTQMARK":
                    sertok_token.content = "?"
                else:
                    raise SexpAnalyzingException(sexp, message="Unknown special token")
                # end if
            # end if

            # Normalize token kind
            if sertok_token.kind in cls.SERTOK_TOKEN_KIND_MAPPING:
                sertok_token.kind = cls.SERTOK_TOKEN_KIND_MAPPING[sertok_token.kind]
            # end if

            # Loc
            sertok_token.loc = cls.analyze_loc(sexp[1], unicode_offsets)

            # It can never be empty string; if it is, it is '""'
            if len(sertok_token.content) == 0:  sertok_token.content = '""'

            # Escape the " in coq style
            if sertok_token.content[0] == '"' and sertok_token.content[-1] == '"' and '"' in sertok_token.content[1:-1]:
                sertok_token.content = '"' + sertok_token.content[1:-1].replace('"', '""') + '"'
            # end if

            # Adjust content to remove quotes, if necessary
            if sertok_token.content[0] == '"' and sertok_token.content[-1] == '"' and len(sertok_token.content) != sertok_token.loc.end_charno - sertok_token.loc.beg_charno:
                sertok_token.content = sertok_token.content[1:-1]
            # end if

            # Adjust content to add quotes, if necessary
            if sertok_token.kind == TokenConsts.KIND_STR and len(sertok_token.content) == sertok_token.loc.end_charno - sertok_token.loc.beg_charno - 2:
                sertok_token.content = '"' + sertok_token.content + '"'
            # end if

            # Fix for charno mismatch  TODO: this should be eventually fixed in Coq
            if sertok_token.loc.end_charno - sertok_token.loc.beg_charno < len(sertok_token.content):
                sertok_token.loc.end_charno = len(sertok_token.content) + sertok_token.loc.beg_charno
            # end if

            return sertok_token
        except IllegalSexpOperationException:
            raise SexpAnalyzingException(sexp)

    @classmethod
    def find_i_pat_ids(cls, sexp: SexpNode) -> Counter[str]:
        try:
            i_pat_ids: Counter[str] = collections.Counter()

            def match_i_pad_id_recur(sexp_part: SexpNode) -> SexpNode.RecurAction:
                nonlocal i_pat_ids
                try:
                    # ( IPatId ( Id <pat_id> ))
                    #   0      1 10 11
                    if sexp_part[0].content == "IPatId" and sexp_part[1][0].content == "Id":
                        i_pat_ids[sexp_part[1][1].content] += 1
                        return SexpNode.RecurAction.StopRecursion
                    else:
                        return SexpNode.RecurAction.ContinueRecursion
                    # end if
                except (IllegalSexpOperationException, SexpAnalyzingException):
                    return SexpNode.RecurAction.ContinueRecursion
                # end try
            # end def

            sexp.apply_recur(match_i_pad_id_recur)

            return i_pat_ids
        except IllegalSexpOperationException:
            raise SexpAnalyzingException(sexp)

    @classmethod
    def cut_lemma_backend_sexp(cls, sexp: SexpNode) -> SexpNode:
        def pre_children_modify(current: SexpNode) -> Tuple[Optional[SexpNode], SexpNode.RecurAction]:
            while True:
                no_change = True

                # TODO: Different Coq.Init names, experiment removing them at later phases.

                # ( ( v <X> ) ( loc () ) ) -> <X>
                if current.is_list() and len(current) == 2 and\
                    current[0].is_list() and len(current[0]) == 2 and\
                    current[0][0].is_string() and current[0][0].content == "v" and\
                    current[1].is_list() and len(current[1]) == 2 and\
                    current[1][0].is_string() and current[1][0].content == "loc" and\
                    current[1][1].is_list() and len(current[1][1]) == 0:
                    # then
                    current = current[0][1]
                    no_change = False
                # end if

                # ( <A> <X> ) -> <X>, where <A> in [Id, ConstRef, Name, GVar]
                if current.is_list() and len(current) == 2 and\
                    current[0].is_string() and current[0].content in ["Id", "ConstRef", "Name", "GVar"]:
                    # then
                    current = current[1]
                    no_change = False
                # end if

                # # ( <A> ( <Xs> ) <Ys> ) -> ( <Ys> ), where <A> in [MPfile]
                # if current.is_list() and len(current) >= 2 and \
                #         current[0].is_string() and current[0].content in ["MPfile"] and \
                #         current[1].is_list():
                #     # then
                #     del current.get_children()[0:2]
                #     no_change = False
                # # end if
                #
                # # ( <A> ( <Xs> ) <Ys> ) -> ( <Xs> <Ys> ), where <A> in [MPdot, DirPath, Constant, MulInd]
                # if current.is_list() and len(current) >= 2 and\
                #     current[0].is_string() and current[0].content in ["MPdot", "DirPath", "Constant", "MutInd"] and\
                #     current[1].is_list():
                #     # then
                #     current[1].get_children().extend(current[2:])
                #     current = current[1]
                #     no_change = False
                # # end if

                # ( <A> ( <Xs> ) <Ys> ) -> <Ys[-1]>, or ( <A> ( <Xs> ) ) -> <Xs[-1]>, where <A> in [MPdot, DirPath, Constant, MulInd, MPfile]
                if current.is_list() and len(current) >= 2 and\
                    current[0].is_string() and current[0].content in ["MPdot", "DirPath", "Constant", "MutInd", "MPfile"] and\
                    current[1].is_list() and len(current[1]) > 0:
                    # then
                    if len(current) == 2:
                        current = current[1][-1]
                    else:
                        current = current[-1]
                    # end if
                    no_change = False
                # end if

                # ( <A> <X> ()) -> <X>, where <A> in [GRef]
                if current.is_list() and len(current) == 3 and\
                    current[0].is_string() and current[0].content in ["GRef"] and\
                    current[2].is_list() and len(current[2]) == 0:
                    # then
                    current = current[1]
                    no_change = False
                # end if

                # # ( <A> <X> ( <Ys> ) ) -> ( <X> <Ys> ), where <A> in [GApp]
                # if current.is_list() and len(current) == 3 and\
                #     current[0].is_string() and current[0].content in ["GApp"] and\
                #     current[2].is_list():
                #     # then
                #     current = SexpList([current[1]] + current[2].get_children())
                #     no_change = False
                # # end if

                # ( IndRef ( ( <Xs> ) <n> ) ) -> <Xs[-1]>
                if current.is_list() and len(current) == 2 and\
                    current[0].is_string() and current[0].content == "IndRef" and\
                    current[1].is_list() and len(current[1]) == 2 and\
                    current[1][0].is_list() and len(current[1][0]) > 0:
                    # then
                    current = current[1][0][-1]
                    no_change = False
                # end if

                # ( ConstructRef ( ( ( <Xs> ) <n> ) <m> ) ) -> <Xs[-1]>
                if current.is_list() and len(current) == 2 and\
                    current[0].is_string() and current[0].content == "ConstructRef" and\
                    current[1].is_list() and len(current[1]) == 2 and\
                    current[1][0].is_list() and len(current[1][0]) == 2 and\
                    current[1][0][0].is_list() and len(current[1][0][0]) > 0:
                    # then
                    current = current[1][0][0][-1]
                    no_change = False
                # end if

                if no_change:  break
            # end while

            # ( GProd . <X> ... ) -> ( GProd . ... )
            if current.is_list() and len(current) >= 3 and \
                    current[0].is_string() and current[0].content == "GProd":
                # then
                del current.get_children()[2]
            # end if

            return current, SexpNode.RecurAction.ContinueRecursion
        # end def

        sexp = sexp.modify_recur(pre_children_modify)

        return sexp

    @classmethod
    def split_lemma_backend_sexp(cls, sexp: SexpNode) -> Tuple[Optional[SexpNode], SexpNode]:
        last_gprod_node: Optional[SexpNode] = None
        first_non_gprod_node: SexpNode = None
        def find_first_non_gprod(sexp: SexpNode):
            nonlocal last_gprod_node, first_non_gprod_node
            if sexp.is_list() and len(sexp) >= 2 and\
                sexp[0].is_string() and sexp[0].content == "GProd":
                # then
                last_gprod_node = sexp
                find_first_non_gprod(sexp[-1])
            else:
                first_non_gprod_node = sexp
                return
            # end if
        # end def

        find_first_non_gprod(sexp)

        # TODO: return a list of the GProds instead of tree, currently set to None
        return None, first_non_gprod_node
