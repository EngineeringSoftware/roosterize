from typing import *

from seutil import LoggingUtils

from roosterize.data.CoqDocument import CoqDocument, VernacularSentence
from roosterize.data.LanguageId import LanguageId
from roosterize.data.Token import Token, TokenConsts
from roosterize.parser.ParserUtils import ParserUtils
from roosterize.parser.SexpAnalyzer import SexpAnalyzer, SexpInfo
from roosterize.sexp import *


class CoqParserConsts:

    VERNAC_TYPES_GALLINA_OR_VERNAC_CONTROL = [
        # pure-VernacControl like
        SexpInfo.VernacConsts.type_abort,
        SexpInfo.VernacConsts.type_add_option,
        SexpInfo.VernacConsts.type_arguments,
        SexpInfo.VernacConsts.type_begin_section,
        SexpInfo.VernacConsts.type_bind_scope,
        SexpInfo.VernacConsts.type_canonical,
        SexpInfo.VernacConsts.type_chdir,
        SexpInfo.VernacConsts.type_check_may_eval,
        SexpInfo.VernacConsts.type_coercion,
        SexpInfo.VernacConsts.type_combined_scheme,
        SexpInfo.VernacConsts.type_declare_module_type,
        SexpInfo.VernacConsts.type_define_module,
        SexpInfo.VernacConsts.type_delimiters,
        SexpInfo.VernacConsts.type_end_proof,
        SexpInfo.VernacConsts.type_end_segment,
        SexpInfo.VernacConsts.type_existing_instance,
        SexpInfo.VernacConsts.type_fail,
        SexpInfo.VernacConsts.type_hints,
        SexpInfo.VernacConsts.type_identify_coercion,
        SexpInfo.VernacConsts.type_import,
        SexpInfo.VernacConsts.type_include,
        SexpInfo.VernacConsts.type_notation,
        SexpInfo.VernacConsts.type_open_close_scope,
        SexpInfo.VernacConsts.type_print,
        SexpInfo.VernacConsts.type_proof,
        SexpInfo.VernacConsts.type_remove_hints,
        SexpInfo.VernacConsts.type_require,
        SexpInfo.VernacConsts.type_scheme,
        SexpInfo.VernacConsts.type_set_opacity,
        SexpInfo.VernacConsts.type_set_option,
        SexpInfo.VernacConsts.type_syntactic_definition,
        SexpInfo.VernacConsts.type_syntax_extension,

        # pure-Gallina like
        SexpInfo.VernacConsts.type_assumption,
        SexpInfo.VernacConsts.type_context,
        SexpInfo.VernacConsts.type_definition,
        SexpInfo.VernacConsts.type_fixpoint,
        SexpInfo.VernacConsts.type_inductive,
        SexpInfo.VernacConsts.type_infix,
        SexpInfo.VernacConsts.type_instance,
        SexpInfo.VernacConsts.type_reserve,
        SexpInfo.VernacConsts.type_start_theorem_proof,
    ]

    VERNAC_TYPES_LTAC = [
        SexpInfo.VernacConsts.type_bullet,
        SexpInfo.VernacConsts.type_end_subproof,
        SexpInfo.VernacConsts.type_exact_proof,
        SexpInfo.VernacConsts.type_extend,
        SexpInfo.VernacConsts.type_subproof,
    ]

    # From https://coq.github.io/doc/v8.10/refman/coq-cmdindex.html
    COMMANDS_SEQUENCES: List[List[str]] = [
        ["Abort"],
        ["About"],
        ["Add"],
        ["Add", "Field"],
        ["Add", "LoadPath"],
        ["Add", "ML", "Path"],
        ["Add", "Morphism"],
        ["Add", "Parametric", "Morphism"],
        ["Add", "Parametric", "Relation"],
        ["Add", "Rec", "LoadPath"],
        ["Add", "Rec", "ML", "Path"],
        ["Add", "Relation"],
        ["Add", "Ring"],
        ["Add", "Setoid"],
        ["Admit", "Obligations"],
        ["Admitted"],
        ["Arguments"],
        ["Axiom"],
        ["Axioms"],
        ["Back"],
        ["BackTo"],
        ["Backtrack"],
        ["Bind", "Scope"],
        ["Canonical"],
        ["Cd"],
        ["Check"],
        ["Class"],
        ["Close", "Scope"],
        ["Coercion"],
        ["CoFixpoint"],
        ["CoInductive"],
        ["Collection"],
        ["Combined", "Scheme"],
        ["Compute"],
        ["Conjecture"],
        ["Conjectures"],
        ["Constraint"],
        ["Context"],
        ["Corollary"],
        ["Create", "HintDb"],
        ["Cumulative"],
        ["Declare", "Custom", "Entry"],
        ["Declare", "Instance"],
        ["Declare", "Left", "Step"],
        ["Declare", "ML", "Module"],
        ["Declare", "Module"],
        ["Declare", "Reduction"],
        ["Declare", "Right", "Step"],
        ["Declare", "Scope"],
        ["Defined"],
        ["Definition"],
        ["Delimit", "Scope"],
        ["Derive"],
        ["Derive", "Inversion"],
        ["Drop"],
        ["End"],
        ["Eval"],
        ["Example"],
        ["Existential"],
        ["Existing", "Instance"],
        ["Export"],
        ["Extract", "Constant"],
        ["Extract", "Inductive"],
        ["Extract", "Inlined", "Constant"],
        ["Extraction"],
        ["Extraction", "Blacklist"],
        ["Extraction", "Implicit"],
        ["Extraction", "Inline"],
        ["Extraction", "Language"],
        ["Extraction", "Library"],
        ["Extraction", "NoInline"],
        ["Extraction", "TestCompile"],
        ["Fact"],
        ["Fail"],
        ["Fixpoint"],
        ["Focus"],
        ["Function"],
        ["Functional", "Scheme"],
        ["Generalizable"],
        ["Generalizable", "All", "Variables"],
        ["Generalizable", "No", "Variables"],
        ["Global"],
        ["Global", "Close", "Scope"],
        ["Global", "Generalizable"],
        ["Global", "Instance"],
        ["Global", "Opaque"],
        ["Global", "Open", "Scope"],
        ["Global", "Transparent"],
        ["Goal"],
        ["Grab", "Existential", "Variables"],
        ["Guarded"],
        ["Hint"],
        ["Hint", "Constants"],
        ["Hint", "Constructors"],
        ["Hint", "Cut"],
        ["Hint", "Extern"],
        ["Hint", "Immediate"],
        ["Hint", "Mode"],
        ["Hint", "Opaque"],
        ["Hint", "Resolve"],
        ["Hint", "Rewrite"],
        ["Hint", "Transparent"],
        ["Hint", "Unfold"],
        ["Hint", "Variables"],
        ["Hint", "View", "for"],
        ["Hint", "View", "for", "apply"],
        ["Hint", "View", "for", "move"],
        ["Hypotheses"],
        ["Hypothesis"],
        ["Identity", "Coercion"],
        ["Implicit", "Types"],
        ["Import"],
        ["Include"],
        ["Inductive"],
        ["Infix"],
        ["Info"],
        ["Inline"],
        ["Inspect"],
        ["Instance"],
        ["Lemma"],
        ["Let"],
        ["Let", "CoFixpoint"],
        ["Let", "Fixpoint"],
        ["Load"],
        ["Local"],
        ["Local", "Close", "Scope"],
        ["Local", "Declare", "Custom", "Entry"],
        ["Local", "Definition"],
        ["Local", "Notation"],
        ["Local", "Open", "Scope"],
        ["Local", "Parameter"],
        ["Locate"],
        ["Locate", "File"],
        ["Locate", "Library"],
        ["Ltac"],
        ["Module"],
        ["Module", "Type"],
        ["Monomorphic"],
        ["Next", "Obligation"],
        ["NonCumulative"],
        ["Notation"],
        ["Numeral", "Notation"],
        ["Obligation"],
        ["Obligation", "Tactic"],
        ["Obligations"],
        ["Opaque"],
        ["Open", "Scope"],
        ["Optimize", "Heap"],
        ["Optimize", "Proof"],
        ["Parameter"],
        ["Parameters"],
        ["Polymorphic"],
        ["Polymorphic", "Constraint"],
        ["Polymorphic", "Universe"],
        ["Prenex", "Implicits"],
        ["Preterm"],
        ["Primitive"],
        ["Print"],
        ["Print", "All"],
        ["Print", "All", "Dependencies"],
        ["Print", "Assumptions"],
        ["Print", "Canonical", "Projections"],
        ["Print", "Classes"],
        ["Print", "Coercion", "Paths"],
        ["Print", "Coercions"],
        ["Print", "Custom", "Grammar"],
        ["Print", "Extraction", "Blacklist"],
        ["Print", "Extraction", "Inline"],
        ["Print", "Firstorder", "Solver"],
        ["Print", "Grammar", "constr"],
        ["Print", "Grammar", "pattern"],
        ["Print", "Grammar", "tactic"],
        ["Print", "Graph"],
        ["Print", "Hint"],
        ["Print", "HintDb"],
        ["Print", "Implicit"],
        ["Print", "Instances"],
        ["Print", "Libraries"],
        ["Print", "LoadPath"],
        ["Print", "Ltac"],
        ["Print", "Ltac", "Signatures"],
        ["Print", "ML", "Modules"],
        ["Print", "ML", "Path"],
        ["Print", "Module"],
        ["Print", "Module", "Type"],
        ["Print", "Opaque", "Dependencies"],
        ["Print", "Options"],
        ["Print", "Rewrite", "HintDb"],
        ["Print", "Scope"],
        ["Print", "Scopes"],
        ["Print", "Strategy"],
        ["Print", "Table"],
        ["Print", "Tables"],
        ["Print", "Term"],
        ["Print", "Transparent", "Dependencies"],
        ["Print", "Universes"],
        ["Print", "Universes", "Subgraph"],
        ["Print", "Visibility"],
        ["Program", "Definition"],
        ["Program", "Fixpoint"],
        ["Program", "Instance"],
        ["Program", "Lemma"],
        ["Proof"],
        ["Proof", "using"],
        ["Proof", "with"],
        ["Proposition"],
        ["Pwd"],
        ["Qed"],
        ["Quit"],
        ["Record"],
        ["Recursive", "Extraction"],
        ["Recursive", "Extraction", "Library"],
        ["Redirect"],
        ["Register"],
        ["Register", "Inline"],
        ["Remark"],
        ["Remove"],
        ["Remove", "Hints"],
        ["Remove", "LoadPath"],
        ["Require"],
        ["Require", "Export"],
        ["Require", "Import"],
        ["Reserved", "Notation"],
        ["Reset"],
        ["Reset", "Extraction", "Blacklist"],
        ["Reset", "Extraction", "Inline"],
        ["Reset", "Ltac", "Profile"],
        ["Restart"],
        ["Save"],
        ["Scheme"],
        ["Scheme", "Equality"],
        ["Search"],
        ["SearchAbout"],
        ["SearchHead"],
        ["SearchPattern"],
        ["SearchRewrite"],
        ["Section"],
        ["Separate", "Extraction"],
        ["Set"],
        ["Show"],
        ["Show", "Conjectures"],
        ["Show", "Existentials"],
        ["Show", "Intro"],
        ["Show", "Intros"],
        ["Show", "Ltac", "Profile"],
        ["Show", "Obligation", "Tactic"],
        ["Show", "Proof"],
        ["Show", "Script"],
        ["Show", "Universes"],
        ["Solve", "All", "Obligations"],
        ["Solve", "Obligations"],
        ["Strategy"],
        ["String", "Notation"],
        ["Structure"],
        ["SubClass"],
        ["Tactic", "Notation"],
        ["Test"],
        ["Theorem"],
        ["Time"],
        ["Timeout"],
        ["Transparent"],
        ["Typeclasses", "eauto"],
        ["Typeclasses", "Opaque"],
        ["Typeclasses", "Transparent"],
        ["Undelimit", "Scope"],
        ["Undo"],
        ["Unfocus"],
        ["Unfocused"],
        ["Universe"],
        ["Unset"],
        ["Unshelve"],
        ["Variable"],
        ["Variables"],
        ["Variant"],
    ]

    MAX_COMMANDS_SEQUENCE_LENGTH = max([len(cs) for cs in COMMANDS_SEQUENCES])

    SYMBOLS = ["!", "%", "&", "&&", "(", "()", ")", "*", "+", "++", ",", "-", "->", ".", ".(", "..", "/", "/\\", ":", "::", ":<", ":=", ":>", ";", "<", "<-", "<->", "<:", "<=", "<>", "=", "=>", "=_D", ">", ">->", ">=", "?", "?=", "@", "[", "\\/", "]", "^", "{", "|", "|-", "||", "}", "~", "#[", "'"]

    CONSTR_EXPR_TYPES_ONE_TOKEN = [
        SexpInfo.ConstrExprRConsts.type_c_ref,
        SexpInfo.ConstrExprRConsts.type_c_hole,
        SexpInfo.ConstrExprRConsts.type_c_sort,
        SexpInfo.ConstrExprRConsts.type_c_prim,
    ]


class CoqParser:
    """
    Parses (different parts of) Coq code.
    """
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)
    from ..Debug import Debug
    if Debug.is_debug: logger.setLevel(LoggingUtils.DEBUG)

    @classmethod
    def parse_sertok_sentences(cls, sertok_sentences: List[SexpInfo.SertokSentence], source_code: str) -> List[VernacularSentence]:
        cur_lineno = 1
        cur_charno = 0
        vernac_sentences: List[VernacularSentence] = list()
        command_tokens: List[str]
        command_next_tokens: Optional[Set[str]]

        for sertok_sentence in sertok_sentences:
            vernac_sentence = VernacularSentence()
            vernac_sentence.tokens = list()

            command_tokens = list()
            command_next_tokens = set([cs[0] for cs in CoqParserConsts.COMMANDS_SEQUENCES])

            for sertok_token in sertok_sentence.tokens:
                # Add comment token if necessary (multiple comment will be grouped together as one)
                # The comment tokens have to be added before calculating the spacing offsets, as they are calculated based on the charno/lineno diff between two consequent tokens
                # They will be removed eventually, after calculating spacing info
                if sertok_token.loc.beg_charno > cur_charno:
                    com_beg_end = ParserUtils.find_comment(source_code[cur_charno:sertok_token.loc.beg_charno])
                    if com_beg_end is not None:
                        com_end_lineno = cur_lineno + source_code[cur_charno:cur_charno+com_beg_end[1]].count("\n")
                        vernac_sentence.tokens.append(Token(
                            kind=TokenConsts.KIND_COMMENT, content = source_code[cur_charno+com_beg_end[0]:cur_charno+com_beg_end[1]],
                            beg_charno=cur_charno+com_beg_end[0],
                            end_charno=cur_charno+com_beg_end[1],
                            lineno=com_end_lineno
                        ))
                    # end if
                # end if

                # Special treatment for FIELD type:
                if sertok_token.kind == "FIELD":
                    vernac_sentence.tokens.append(Token(
                        kind=TokenConsts.KIND_SYMBOL, content=".",
                        beg_charno=sertok_token.loc.beg_charno,
                        end_charno=sertok_token.loc.beg_charno+1,
                        lineno=sertok_token.loc.lineno,
                    ))
                    sertok_token.kind = TokenConsts.KIND_ID
                    sertok_token.loc.beg_charno += 1
                # end if

                # Adjustment token kind: sertok put KEYWORD and SYMBOL in the same category
                if sertok_token.kind == TokenConsts.KIND_KEYWORD and sertok_token.content in CoqParserConsts.SYMBOLS:
                    sertok_token.kind = TokenConsts.KIND_SYMBOL
                # end if

                # Adjustment token kind: for vernac commands in the beginning of the sentence
                if command_next_tokens is not None and len(command_tokens) < CoqParserConsts.MAX_COMMANDS_SEQUENCE_LENGTH:
                    if sertok_token.content in command_next_tokens:
                        sertok_token.kind = TokenConsts.KIND_KEYWORD
                        command_tokens.append(sertok_token.content)
                        command_i = len(command_tokens)
                        command_next_tokens = set([cs[command_i] for cs in CoqParserConsts.COMMANDS_SEQUENCES if cs[:command_i] == command_tokens and len(cs) > command_i])
                    else:
                        command_next_tokens = None
                    # end if
                # end if

                # Add token into sentence
                vernac_sentence.tokens.append(Token(
                    kind=sertok_token.kind, content=sertok_token.content,
                    beg_charno=sertok_token.loc.beg_charno,
                    end_charno=sertok_token.loc.end_charno,
                    lineno=sertok_token.loc.lineno,
                ))

                cur_lineno = sertok_token.loc.lineno
                cur_charno = sertok_token.loc.end_charno
            # end for
            vernac_sentences.append(vernac_sentence)
        # end for

        # Set coffset, loffset, indentation
        cur_lineno = 1
        cur_charno = 0
        for vernac_sentence in vernac_sentences:
            for token in vernac_sentence.tokens:
                start_lineno = token.lineno - token.content.count("\n")
                if start_lineno > cur_lineno:  # Has new line
                    latest_newline = source_code[cur_charno:token.beg_charno].rfind("\n") + cur_charno + 1
                    token.loffset = start_lineno - cur_lineno
                    token.indentation = token.beg_charno - latest_newline
                    token.coffset = TokenConsts.OFFSET_INVALID
                else:
                    token.indentation = TokenConsts.OFFSET_INVALID
                    token.coffset = token.beg_charno - cur_charno
                    token.loffset = TokenConsts.OFFSET_INVALID
                # end if

                cur_lineno = token.lineno
                cur_charno = token.end_charno
            # end for
        # end for

        # Remove all comments
        for vernac_i, vernac_sentence in enumerate(vernac_sentences):
            for token_i in reversed(range(len(vernac_sentence.tokens))):
                if vernac_sentence.tokens[token_i].kind == TokenConsts.KIND_COMMENT:
                    # cls.logger.debug(f"Removing comment at vernac#{vernac_i}, token#{token_i}; loffset {vernac_sentence.tokens[0].loffset}, indentation {vernac_sentence.tokens[0].indentation}, coffset {vernac_sentence.tokens[0].coffset}; content {vernac_sentence.tokens[0].content}")
                    if token_i == 0 and vernac_sentence.tokens[0].loffset != TokenConsts.OFFSET_INVALID:
                        # Push the spacing info onto the next token, if it is a comment-block (there must be one non-comment token in the sentence).
                        vernac_sentence.tokens[1].loffset = vernac_sentence.tokens[0].loffset
                        vernac_sentence.tokens[1].indentation = vernac_sentence.tokens[0].indentation
                        vernac_sentence.tokens[1].coffset = vernac_sentence.tokens[0].coffset
                    # end if
                    vernac_sentence.tokens.pop(token_i)
                # end if
            # end for
        # end for

        return vernac_sentences

    @classmethod
    def should_ignore_gallina_part(cls, gallina_part: Union[SexpInfo.ConstrExprR, SexpInfo.CLocalAssum]) -> bool:
        if isinstance(gallina_part, SexpInfo.ConstrExprR):
            return gallina_part.expr_type in CoqParserConsts.CONSTR_EXPR_TYPES_ONE_TOKEN
        else:
            return gallina_part.is_one_token
        # end if

    @classmethod
    def parse_document(cls,
            source_code: str,
            ast_sexp_list: List[SexpNode],
            tok_sexp_list: List[SexpNode],
            unicode_offsets: List[int],
    ) -> CoqDocument:
        # Parse tok sexp to vernacular setences
        sertok_sentences: List[SexpInfo.SertokSentence] = SexpAnalyzer.analyze_sertok_sentences(tok_sexp_list, unicode_offsets)
        vernac_sentences: List[VernacularSentence] = cls.parse_sertok_sentences(sertok_sentences, source_code)

        # Parse ast sexp
        vernac_asts: List[Optional[SexpInfo.Vernac]] = [SexpAnalyzer.analyze_vernac(ast_sexp) for ast_sexp in ast_sexp_list]

        # Prepare for warning of unseen vernacular types and token kinds
        unseen_vernacular_types = set()
        unseen_token_kinds = set()

        # Assign language identifications
        is_in_ltac_part: bool = False
        is_next_maybe_ltac_part: bool = False
        for vernac_i, vernac_sentence in enumerate(vernac_sentences):
            vernac_ast = vernac_asts[vernac_i]

            # Detect sentence lid from the vernacular type
            if vernac_ast.vernac_type in CoqParserConsts.VERNAC_TYPES_GALLINA_OR_VERNAC_CONTROL:
                sentence_lid = LanguageId.Vernac
            elif vernac_ast.vernac_type in CoqParserConsts.VERNAC_TYPES_LTAC:
                sentence_lid = LanguageId.Ltac
            else:
                if vernac_ast.vernac_type not in unseen_vernacular_types:
                    cls.logger.warning(f"Unknown vernacular type {vernac_ast.vernac_type} at line {vernac_ast.loc.lineno} of {vernac_ast.loc.filename}")
                    unseen_vernacular_types.add(vernac_ast.vernac_type)
                # end if
                sentence_lid = LanguageId.Vernac
            # end if

            # Check if this sentence really starts Ltac part
            if is_next_maybe_ltac_part:
                if sentence_lid == LanguageId.Ltac:  is_in_ltac_part = True
                is_next_maybe_ltac_part = False
            # end if

            # Check sentence_lid against is_in_ltac_part
            if not is_in_ltac_part and sentence_lid == CoqParserConsts.VERNAC_TYPES_LTAC:
                cls.logger.warning(f"Ltac sentence in non-Ltac part, likely to be a bug, at {vernac_ast.loc}")
            # end if

            # Find Gallina parts, and do special treatment for one-token Gallina parts
            gallina_parts: List[Union[SexpInfo.ConstrExprR, SexpInfo.CLocalAssum]] = SexpAnalyzer.find_gallina_parts(vernac_ast.vernac_sexp, unicode_offsets)
            one_token_gallina_parts: List[Union[SexpInfo.ConstrExprR, SexpInfo.CLocalAssum]] = list()

            for gp_idx in reversed(range(len(gallina_parts))):
                if cls.should_ignore_gallina_part(gallina_parts[gp_idx]):
                    one_token_gallina_parts.insert(0, gallina_parts[gp_idx])
                    gallina_parts.pop(gp_idx)
                # end if
            # end for

            # Check if a proof should start/end
            if vernac_ast.vernac_type in [SexpInfo.VernacConsts.type_start_theorem_proof, SexpInfo.VernacConsts.type_definition, SexpInfo.VernacConsts.type_instance]:
                # Next sentence may be Ltac; WARNING: we don't handle nested proof for now
                is_next_maybe_ltac_part = True
            elif (vernac_ast.vernac_type == SexpInfo.VernacConsts.type_proof) or \
                (vernac_ast.vernac_type == SexpInfo.VernacConsts.type_extend and vernac_ast.extend_type == SexpInfo.VernacConsts.extend_type_obligations):
                # Next sentence must be Ltac; WARNING: we don't handle nested proof for now
                is_in_ltac_part = True
            elif vernac_ast.vernac_type in [SexpInfo.VernacConsts.type_end_proof, SexpInfo.VernacConsts.type_abort, SexpInfo.VernacConsts.type_exact_proof]:
                # Next sentence should be out of Ltac
                # TODO the following check is not applied for now, because "Admitted." gives false alarm
                # if not is_in_ltac_part and not is_next_maybe_ltac_part and vernac_ast.vernac_type not in SexpInfo.VernacConsts.type_abort:  cls.logger.warning(f"EndProof appeared in non-Ltac part, likely to be a bug, at {vernac_ast.loc}.")
                is_in_ltac_part = False
            # end if

            for token in vernac_sentence.tokens:
                # Check for unseen kind
                if token.kind not in [TokenConsts.KIND_ID, TokenConsts.KIND_SYMBOL, TokenConsts.KIND_KEYWORD, TokenConsts.KIND_NUMBER, TokenConsts.KIND_STR]:
                    if token.kind not in unseen_token_kinds:
                        cls.logger.warning(f"Unseen token kind {token.kind} at token {token}")
                        unseen_token_kinds.add(token.kind)
                    # end if
                # end if

                # Default to sentence lid
                token.lang_id = sentence_lid

                # Mark gallina parts
                for gp in gallina_parts:
                    if gp.loc.contains_charno_range(token.beg_charno, token.end_charno):
                        token.lang_id = LanguageId.Gallina
                        break
                    # end if
                # end for

                # Mark one-token gallina parts, only if it's not already marked by a longer gallina part
                if token.lang_id == LanguageId.Gallina:  continue
                for gp in one_token_gallina_parts:
                    if gp.loc.contains_charno_range(token.beg_charno, token.end_charno):
                        token.lang_id = LanguageId.Gallina
                        token.is_one_token_gallina = True
                        break
                    # end if
                # end for
            # end for

        # end for

        # Create coq document
        coq_document = CoqDocument()
        coq_document.sentences = vernac_sentences

        return coq_document
