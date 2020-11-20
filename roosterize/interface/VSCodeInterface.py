from typing import List, Tuple

import numpy as np
from pygls.features import WINDOW_SHOW_MESSAGE_REQUEST
from pygls.server import LanguageServer
from pygls.types import Diagnostic, DiagnosticSeverity, MessageActionItem, MessageType, Position, Range, \
    ShowMessageRequestParams

from roosterize.data.Lemma import Lemma
from roosterize.interface.CommandLineInterface import CommandLineInterface, ProcessedFile


class VSCodeInterface(CommandLineInterface):

    def __init__(self):
        super().__init__()
        self.ls: LanguageServer = None

    def set_language_server(self, ls: LanguageServer):
        self.ls = ls

    def ask_for_confirmation(self, text: str) -> bool:
        yes = MessageActionItem("yes")
        no = MessageActionItem("no")
        future = self.ls.lsp.send_request(
            WINDOW_SHOW_MESSAGE_REQUEST,
            params=ShowMessageRequestParams(MessageType.Warning, text, [yes, no]),
        )

        selected = future.result()
        if selected.title == "yes":
            return True
        else:
            return False

    def show_message(self, text: str):
        self.ls.show_message(text)

    def report_predictions(self, data: ProcessedFile, candidates_logprobs: List[List[Tuple[str, float]]]):
        # First, figure out what to suggest
        good_names: List[Lemma] = []
        bad_names_and_suggestions: List[Tuple[Lemma, str, float]] = []
        bad_names_no_suggestion: List[Tuple[Lemma, str, float]] = []

        for lemma, pred in zip(data.lemmas, candidates_logprobs):
            acceptable_names = [n for n, s in pred[:self.no_suggestion_if_in_top_k]]
            if lemma.name in acceptable_names:
                good_names.append(lemma)
            else:
                top_suggestion, logprob = pred[0]
                score = np.exp(logprob)
                if score < self.min_suggestion_likelihood:
                    bad_names_no_suggestion.append((lemma, top_suggestion, score))
                else:
                    bad_names_and_suggestions.append((lemma, top_suggestion, score))

        total = len(good_names) + len(bad_names_and_suggestions) + len(bad_names_no_suggestion)
        self.show_message(f"{data.path}: Analyzed {total} lemma names, "
                          f"{len(good_names)} ({len(good_names)/total:.1%}) look good. "
                          f"Roosterize made {len(bad_names_and_suggestions)} suggestions.")

        # Publish suggestions as diagnostics
        uri = data.path.as_uri()
        diagnostics = []

        if len(bad_names_and_suggestions) > 0:
            for lemma, suggestion, score in sorted(bad_names_and_suggestions, key=lambda x: x[2], reverse=True):
                beg_line, beg_col, end_line, end_col = self.get_lemma_name_position(lemma)
                diagnostics.append(Diagnostic(
                    range=Range(Position(beg_line, beg_col), Position(end_line, end_col)),
                    message=f"Suggestion: {suggestion} (likelihood: {score:.2f})",
                    source="Roosterize",
                    severity=DiagnosticSeverity.Warning,
                ))
        if len(bad_names_no_suggestion) > 0:
            for lemma, suggestion, score in sorted(bad_names_no_suggestion, key=lambda x: x[2], reverse=True):
                beg_line, beg_col, end_line, end_col = self.get_lemma_name_position(lemma)
                diagnostics.append(Diagnostic(
                    range=Range(Position(beg_line, beg_col), Position(end_line, end_col)),
                    message=f"Suggestion: {suggestion} (likelihood: {score:.2f})",
                    source="Roosterize",
                    severity=DiagnosticSeverity.Information,
                ))
        self.ls.publish_diagnostics(uri, diagnostics)

    @classmethod
    def get_lemma_name_position(cls, lemma: Lemma) -> Tuple[int, int, int, int]:
        """
        Returns the position of the lemma's name: beg_line, beg_col, end_line, end_col.
        Assuming the lemma starts on a new line.
        """
        cur_col = 0
        for tok in lemma.vernac_command:
            if tok.loffset > 0:
                cur_col = 0

            if tok.indentation >= 0:
                cur_col += tok.indentation
            elif tok.coffset >= 0:
                cur_col += tok.coffset

            cur_col += len(tok.content)

        lineno = tok.lineno-1
        return lineno, cur_col+1, lineno, cur_col+1+len(lemma.name)
