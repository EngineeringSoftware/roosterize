import warnings
import re
import shutil
import tempfile
import urllib
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from roosterize.Macros import Macros
from seutil import BashUtils, IOUtils

from roosterize.data.CoqDocument import CoqDocument
from roosterize.data.DataMiner import DataMiner
from roosterize.data.Definition import Definition
from roosterize.data.Lemma import Lemma
from roosterize.data.ModelSpec import ModelSpec
from roosterize.ml.MLModels import MLModels
from roosterize.ml.naming.NamingModelBase import NamingModelBase
from roosterize.parser.CoqParser import CoqParser, SexpNode, SexpParser
from roosterize.parser.ParserUtils import ParserUtils
from roosterize.interface.RoosterizeDirUtils import RoosterizeDirUtils


class ProcessedFile(NamedTuple):
    path: Path
    source_code: str
    doc: CoqDocument
    ast_sexp_list: List[SexpNode]
    tok_sexp_list: List[SexpNode]
    unicode_offsets: List[int]
    lemmas: List[Lemma]
    definitions: List[Definition]


class CommandLineInterface:

    SHARED_CONFIGS = [
        "beam_search_size",
        "k",
        "min_suggestion_likelihood",
        "no_suggestion_if_in_top_k",
    ]
    GLOBAL_CONFIGS = SHARED_CONFIGS + ["model_url"]
    LOCAL_CONFIGS = SHARED_CONFIGS + [
        "serapi_options",
        "exclude_files",
        "exclude_pattern",
        "compile_cmd",
    ]

    def __init__(self):
        self.model: NamingModelBase = None

        # Configs (default values)
        self.beam_search_size = 5
        self.k = 5
        self.min_suggestion_likelihood = 0.2
        self.no_suggestion_if_in_top_k = 3
        self.exclude_files = None
        self.exclude_pattern = None
        self.serapi_options = None
        self.model_url = "https://github.com/EngineeringSoftware/roosterize/releases/download/v1.1.0+8.10.2-beta/roosterize-model-t1.tgz"
        self.compile_cmd = None
        self.loaded_config_prj: Path = None

        self.load_configs()

        # Filter out the torchtext warning. TODO: remove this filter after removing dependency to OpenNMT
        warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is recommended to use")
        return

    def load_configs(self, prj_root: Optional[Path] = None, force_reload: bool = False):
        """
        Load configs (first project-local, then global) to this user interface.
        """
        # If the configs of the current project is already loaded, skip
        if not force_reload and prj_root is not None and prj_root == self.loaded_config_prj:
            return

        # Reset the project-local config indicator
        self.loaded_config_prj = None

        # First, load global config
        global_config_file = RoosterizeDirUtils.get_global_config_file()
        if global_config_file.exists():
            global_config = IOUtils.load(global_config_file, IOUtils.Format.yaml)
            self.set_configs_from_dict(global_config, self.GLOBAL_CONFIGS)

        # Then, load local config
        if prj_root is not None:
            local_config_file = RoosterizeDirUtils.get_local_config_file(prj_root)
            if local_config_file.exists():
                local_config = IOUtils.load(local_config_file, IOUtils.Format.yaml)
                self.set_configs_from_dict(local_config, self.LOCAL_CONFIGS)

            self.loaded_config_prj = prj_root

    def set_configs_from_dict(self, d: dict, fields: List[str]):
        for f in fields:
            if f in d:
                setattr(self, f, d[f])

    def download_global_model(self, force_yes: bool = False):
        """
        Downloads a global Roosterize model.
        """
        global_model_dir = RoosterizeDirUtils.get_global_model_dir()
        if global_model_dir.exists():
            ans = self.ask_for_confirmation(
                f"A Roosterize model already exists at {global_model_dir}. "
                f"Do you want to delete it and download again?"
            )
            if force_yes:
                ans = True
            if ans != True:
                return
            IOUtils.rm_dir(global_model_dir)

        self.show_message("Downloading Roosterize model...")

        # Download and unpack
        temp_model_dir = Path(tempfile.mkdtemp(prefix="roosterize"))

        urllib.request.urlretrieve(self.model_url, str(temp_model_dir / "model.tgz"))
        with IOUtils.cd(temp_model_dir):
            BashUtils.run("tar xzf model.tgz", expected_return_code=0)

            # Move the stuff to global model place
            shutil.move(str(Path.cwd() / "model"), global_model_dir)

        # Delete temp dir
        IOUtils.rm_dir(temp_model_dir)

        self.show_message("Finish downloading Roosterize model.")

    RE_SERAPI_OPTIONS = re.compile(r"-R (?P<src>\S+) (?P<tgt>\S+)")

    def infer_serapi_options(self, prj_root: Path) -> str:
        # Try to use the one loaded from config
        self.load_configs(prj_root)
        if self.serapi_options is not None:
            return self.serapi_options

        # Try to infer from _CoqProject
        coq_project_file = prj_root / "_CoqProject"
        possible_serapi_options = []
        if coq_project_file.exists():
            coq_project = IOUtils.load(coq_project_file, IOUtils.Format.txt)
            for l in coq_project.splitlines():
                match = self.RE_SERAPI_OPTIONS.fullmatch(l.strip())
                if match is not None:
                    possible_serapi_options.append(f"-R {match.group('src')},{match.group('tgt')}")

        if len(possible_serapi_options) > 0:
            serapi_options = " ".join(possible_serapi_options)
            return serapi_options
        else:
            return ""

    def suggest_naming(self, file_path: Path, prj_root: Optional[Path] = None):
        """
        Processes a file to get its lemmas and runs the model to get predictions.
        """
        # Figure out which project we're at, and then load configs
        if prj_root is None:
            prj_root = RoosterizeDirUtils.auto_infer_project_root(file_path)
        self.load_configs(prj_root)

        # Infer SerAPI options
        serapi_options = self.infer_serapi_options(prj_root)

        # If user provided compile_cmd, first compile the project
        if self.compile_cmd is not None:
            with IOUtils.cd(prj_root):
                BashUtils.run(self.compile_cmd, expected_return_code=0)

        # Parse file
        data = self.parse_file(file_path, prj_root, serapi_options)

        # Load model
        self.load_local_model(prj_root)
        model = self.get_model()

        # Use the model to make predictions
        # Temp dirs for processed data and results
        temp_data_dir = Path(tempfile.mkdtemp(prefix="roosterize"))

        # Dump lemmas & definitions
        temp_raw_data_dir = temp_data_dir / "raw"
        temp_raw_data_dir.mkdir()
        IOUtils.dump(
            temp_raw_data_dir / "lemmas.json",
            IOUtils.jsonfy(data.lemmas),
            IOUtils.Format.json,
        )
        IOUtils.dump(
            temp_raw_data_dir / "definitions.json",
            IOUtils.jsonfy(data.definitions),
            IOUtils.Format.json,
        )

        # Model-specific process
        temp_processed_data_dir = temp_data_dir / "processed"
        temp_processed_data_dir.mkdir()
        model.process_data_impl(temp_raw_data_dir, temp_processed_data_dir)

        # Invoke eval
        candidates_logprobs = model.eval_impl(
            temp_processed_data_dir,
            beam_search_size=self.beam_search_size,
            k=self.k,
        )

        # Save predictions
        IOUtils.rm_dir(temp_data_dir)

        # Report predictions
        self.report_predictions(data, candidates_logprobs)
        return

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

        # Print suggestions
        total = len(good_names) + len(bad_names_and_suggestions) + len(bad_names_no_suggestion)
        print(f"== Analyzed {total} lemma names, "
              f"{len(good_names)} ({len(good_names)/total:.1%}) conform to the learned naming conventions.")
        if len(bad_names_and_suggestions) > 0:
            print(f"==========")
            print(f"== {len(bad_names_and_suggestions)} can be improved and here are Roosterize's suggestions:")
            for lemma, suggestion, score in sorted(bad_names_and_suggestions, key=lambda x: x[2], reverse=True):
                print(f"Line {lemma.vernac_command[0].lineno}: {lemma.name} => {suggestion} (likelihood: {score:.2f})")
        if len(bad_names_no_suggestion) > 0:
            print(f"==========")
            print(f"== {len(bad_names_no_suggestion)} can be improved but Roosterize cannot provide good suggestion:")
            for lemma, suggestion, score in sorted(bad_names_no_suggestion, key=lambda x: x[2], reverse=True):
                print(f"Line {lemma.vernac_command[0].lineno}: {lemma.name} (best guess: {suggestion}; likelihood: {score:.2f})")

    def load_local_model(self, prj_root: Path) -> None:
        """
        Try to load the local model, if it exists; otherwise do nothing.
        """
        if self.model is None:
            local_model_dir = RoosterizeDirUtils.get_local_model_dir(prj_root)
            if local_model_dir.is_dir():
                model_spec = IOUtils.dejsonfy(
                    IOUtils.load(local_model_dir / "spec.json", IOUtils.Format.json),
                    ModelSpec,
                )
                self.model = MLModels.get_model(local_model_dir, model_spec, is_eval=True)

    def get_model(self) -> NamingModelBase:
        """
        Try to get the currently loaded model; if no model is loaded, gets the global model.
        The local model can be loaded by invoking load_local_model (before invoking this method).
        """
        if self.model is None:
            # Load global model
            global_model_dir = RoosterizeDirUtils.get_global_model_dir()
            model_spec = IOUtils.dejsonfy(
                IOUtils.load(global_model_dir / "spec.json", IOUtils.Format.json),
                ModelSpec,
            )
            self.model = MLModels.get_model(global_model_dir, model_spec, is_eval=True)
        return self.model

    def parse_file(self, file_path: Path, prj_root: Path, serapi_options: str):
        source_code = IOUtils.load(file_path, IOUtils.Format.txt)
        unicode_offsets = ParserUtils.get_unicode_offsets(source_code)

        with IOUtils.cd(prj_root):
            rel_path = file_path.relative_to(prj_root)
            ast_sexp_str = BashUtils.run(f"sercomp {serapi_options} --mode=sexp -- {rel_path}", expected_return_code=0).stdout
            tok_sexp_str = BashUtils.run(f"sertok {serapi_options} -- {rel_path}", expected_return_code=0).stdout

            ast_sexp_list: List[SexpNode] = SexpParser.parse_list(ast_sexp_str)
            tok_sexp_list: List[SexpNode] = SexpParser.parse_list(tok_sexp_str)

            doc = CoqParser.parse_document(
                source_code,
                ast_sexp_list,
                tok_sexp_list,
                unicode_offsets=unicode_offsets,
            )
            doc.file_name = str(rel_path)

            # Collect lemmas & definitions
            lemmas: List[Lemma] = DataMiner.collect_lemmas_doc(doc, ast_sexp_list, serapi_options)
            definitions: List[Definition] = DataMiner.collect_definitions_doc(doc, ast_sexp_list)

        return ProcessedFile(file_path, source_code, doc, ast_sexp_list, tok_sexp_list, unicode_offsets, lemmas, definitions)

    def improve_project_model(self, prj_root: Optional[Path]):
        if prj_root is None:
            prj_root = RoosterizeDirUtils.auto_infer_project_root()

        # Deactivate loaded model
        self.model = None

        # Delete existing local model
        local_model_dir = RoosterizeDirUtils.get_local_model_dir(prj_root)
        if local_model_dir.exists():
            ans = self.ask_for_confirmation(
                f"A Roosterize model already exists at {local_model_dir}"
                f"Do you want to delete it and train again?"
            )
            if not ans:
                return
            else:
                IOUtils.rm_dir(local_model_dir)

        # Copy global model to local model, but remove "training complete" marker
        global_model_dir = RoosterizeDirUtils.get_global_model_dir()
        if not global_model_dir.exists():
            raise Exception("Global Roosterize model not found! Please download model first.")
        shutil.copytree(global_model_dir, local_model_dir)

        # Load local model
        self.load_local_model(prj_root)
        model = self.get_model()

        # Collect all lemmas in this project
        temp_data_dir = Path(tempfile.mkdtemp(prefix="roosterize"))

        DataMiner.extract_data_project(
            prj_root,
            files=None,
            exclude_files=self.exclude_files,
            exclude_pattern=self.exclude_pattern,
            serapi_options=self.infer_serapi_options(prj_root),
            output_path=temp_data_dir
        )

        # TODO: Split data into train/val set, then process each data (no pre-processing / rebuilding vocab!)

        # TODO: Train model

        # Delete temp file
        IOUtils.rm_dir(temp_data_dir)

    def ask_for_confirmation(self, text: str) -> bool:
        ans = input(text + "\n[yes/no] > ")
        parsed_ans = self.parse_yes_no_answer(ans)
        if parsed_ans is None:
            return False
        return parsed_ans

    @classmethod
    def parse_yes_no_answer(cls, ans: str) -> Optional[bool]:
        if str.lower(ans) in ["y", "yes"]:
            return True
        elif str.lower(ans) in ["n", "no"]:
            return False
        else:
            return None

    def show_message(self, text: str):
        print(text)
