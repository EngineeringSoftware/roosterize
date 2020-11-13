from typing import *

import collections
import copy
import hashlib
import math
import numpy as np
from pathlib import Path
import random
import re
from tqdm import tqdm
import traceback
import sys

from seutil import LoggingUtils, IOUtils, BashUtils
from seutil.project import Project

from roosterize.data.CoqDocument import CoqDocument
from roosterize.FilesManager import FilesManager
from roosterize.data.Definition import Definition
from roosterize.data.Lemma import Lemma
from roosterize.data.LemmaBackendSexpTransformers import LemmaBackendSexpTransformers
from roosterize.data.LemmaForeendSexpTransformers import LemmaForeendSexpTransformers
from roosterize.Environment import Environment
from roosterize.Macros import Macros
from roosterize.parser.CoqParser import CoqParser
from roosterize.parser.ParserUtils import ParserUtils
from roosterize.parser.SexpAnalyzer import SexpAnalyzer, SexpInfo
from roosterize.sexp import *
from roosterize.Utils import Utils


class DataMiner:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG)
    from roosterize.Debug import Debug
    if Debug.is_debug: logger.setLevel(LoggingUtils.DEBUG)

    Project.set_downloads_dir(Macros.downloads_dir)

    TASK_COQ_DOCUMENTS = FilesManager.COQ_DOCUMENTS  # "coq-documents"
    TASK_DATA_INDEXES = FilesManager.DATA_INDEXES  # "data-indexes"
    TASK_DEFINITIONS = FilesManager.DEFINITIONS  # "definitions"
    TASK_INSTALL_COQ_PROJECTS = "install-coq-projects"
    TASK_LEMMA = FilesManager.LEMMAS  # "lemmas"
    TASK_LEMMA_BACKEND_SEXP_TRANSFORMATIONS = FilesManager.LEMMAS_BACKEND_SEXP_TRANSFORMATIONS  # "lemmas-bsexp-transformations"
    TASK_LEMMA_FILTERED = FilesManager.LEMMAS_FILTERED  # "lemmas-filtered"
    TASK_LEMMA_FOREEND_SEXP_TRANSFORMATIONS = FilesManager.LEMMAS_FOREEND_SEXP_TRANSFORMATIONS  # "lemmas-fsexp-transformations"

    dataset_dir = Macros.project_dir.parent / "math-comp-corpus"

    @classmethod
    def collect_data(cls, **options) -> NoReturn:
        data_mgr = FilesManager(cls.dataset_dir)

        task = options["task"]

        projects_path = Path(options.get("corpus", cls.dataset_dir / "projects-standalone-8.10.yml"))
        projects: List[Project] = IOUtils.dejsonfy(IOUtils.load(projects_path, "json"), Project)

        if task == cls.TASK_COQ_DOCUMENTS:
            files = Utils.get_option_as_list(options, "files", None)
            is_verifying_tokenizer = Utils.get_option_as_boolean(options, "verify-tokenizer")
            cls.collect_coq_documents_projects(data_mgr, projects, files, is_verifying_tokenizer)
        elif task == cls.TASK_DATA_INDEXES:
            cls.collect_data_indexes(data_mgr, projects)
        elif task == cls.TASK_DEFINITIONS:
            cls.collect_definitions(data_mgr)
        elif task == cls.TASK_INSTALL_COQ_PROJECTS:
            cls.install_coq_projects(projects)
        elif task == cls.TASK_LEMMA:
            files = Utils.get_option_as_list(options, "files", None)
            cls.collect_lemmas(data_mgr, projects, files)
        elif task == cls.TASK_LEMMA_BACKEND_SEXP_TRANSFORMATIONS:
            cls.collect_lemmas_backend_sexp_transformations(data_mgr)
        elif task == cls.TASK_LEMMA_FILTERED:
            cls.filter_lemmas(data_mgr)
        elif task == cls.TASK_LEMMA_FOREEND_SEXP_TRANSFORMATIONS:
            cls.collect_lemmas_foreend_sexp_transformations(data_mgr)
        else:
            LoggingUtils.log_and_raise(cls.logger, f"Unknown task {task}", ValueError)
        # end if
        return

    @classmethod
    def collect_coq_documents_projects(cls,
            data_mgr: FilesManager,
            projects: List[Project],
            files: List[str] = None,
            is_verifying_tokenizer: bool = False,
    ) -> NoReturn:
        # Prepare the used directories (coq-documents, raw-files, original-files)
        for rel_path in [
            [FilesManager.COQ_DOCUMENTS],
            [FilesManager.RAW_FILES],
            [FilesManager.ORIGINAL_FILES],
        ]:
            data_mgr.clean_path(rel_path)
            data_mgr.resolve(rel_path).mkdir(parents=True)
        # end for

        coq_documents: List[CoqDocument] = list()

        names_projects = {p.full_name: p for p in projects}

        for i, project in enumerate(projects):
            try:
                cls.logger.info(f"Project {i + 1}/{len(projects)}: {project.full_name}")
                coq_documents_project = cls.collect_coq_documents_project(data_mgr, project, names_projects=names_projects, files=files, is_verifying_tokenizer=is_verifying_tokenizer)
            except KeyboardInterrupt:
                raise
            except:
                cls.logger.warning(f"Error while processing project {project.full_name}: {traceback.format_exc()}")
                continue
            else:
                coq_documents.extend(coq_documents_project)
            # end try
        # end for

        # Save datasets
        data_mgr.dump_data([FilesManager.COQ_DOCUMENTS, FilesManager.COQ_DOCUMENTS], coq_documents, IOUtils.Format.json, is_batched=True)
        return

    @classmethod
    def load_coq_documents(cls, data_mgr: FilesManager) -> List[CoqDocument]:
        return data_mgr.load_data([FilesManager.COQ_DOCUMENTS, FilesManager.COQ_DOCUMENTS], IOUtils.Format.json, is_batched=True, clz=CoqDocument)

    @classmethod
    def collect_coq_documents_project(cls,
            data_mgr: FilesManager,
            project: Project,
            names_projects: Dict[str, Project],
            files: List[str] = None,
            is_verifying_tokenizer: bool = False,
    ) -> List[CoqDocument]:
        coq_documents: List[CoqDocument] = list()

        # Clone and checkout repo
        project.clone()
        project.checkout(project.data["sha"], is_forced=True)

        # Build the project
        cls.install_coq_project(project, names_projects)

        # For each file, parse code to tokens
        with IOUtils.cd(project.checkout_dir):
            coq_files: List[str] = BashUtils.run(f"find -name '*.v' -type f").stdout.split("\n")[:-1]
            if files is not None:
                coq_files = [f for f in coq_files if f[2:] in files]  # [2:] is to remove the ./
            # end if
            re_ignore_path = re.compile(project.data["ignore_path_regex"]) if "ignore_path_regex" in project.data else None
            for i, coq_file in enumerate(coq_files):
                try:
                    coq_file = coq_file[2:]
                    cls.logger.debug(f"File {i + 1}/{len(coq_files)}: {coq_file}")

                    # Check if file is ignored
                    if re_ignore_path is not None and re_ignore_path.fullmatch(coq_file):
                        cls.logger.info(f"Ignoring file {coq_file}")
                        continue
                    # end if

                    # Read file
                    with open(coq_file, "r", newline="") as f:
                        source_code = f.read()
                    # end with

                    # Get unicode offsets
                    unicode_offsets = ParserUtils.get_unicode_offsets(source_code)

                    # Save original file to original_files
                    data_mgr.dump_data([FilesManager.ORIGINAL_FILES,project.full_name, coq_file], source_code, IOUtils.Format.txt)

                    # Call SerAPI
                    serapi_options = project.data.get("serapi_options", "")
                    ast_sexp_str: str = BashUtils.run(f"sercomp {serapi_options} --mode=sexp -- {coq_file}", expected_return_code=0).stdout
                    tok_sexp_str: str = BashUtils.run(f"sertok {serapi_options} -- {coq_file}", expected_return_code=0).stdout

                    # Save ast sexp to dataset (.ast.sexp)
                    data_mgr.dump_data([FilesManager.RAW_FILES,project.full_name, coq_file[:-2] + ".ast.sexp"], ast_sexp_str, IOUtils.Format.txt)

                    # Save tok sexp to dataset (.tok.sexp)
                    data_mgr.dump_data([FilesManager.RAW_FILES, project.full_name, coq_file[:-2] + ".tok.sexp"], tok_sexp_str, IOUtils.Format.txt)

                    # Parse ast sexp
                    ast_sexp_list: List[SexpNode] = SexpParser.parse_list(ast_sexp_str)
                    tok_sexp_list: List[SexpNode] = SexpParser.parse_list(tok_sexp_str)

                    # Verify the tokenizer if requested
                    if is_verifying_tokenizer:
                        if not cls.verify_tokenizer(tok_sexp_list, source_code, unicode_offsets):
                            LoggingUtils.log_and_raise(cls.logger, "Tokenized content doesn't match original file!", Exception)
                        # end if
                    # end if

                    # Parse the document
                    coq_document = CoqParser.parse_document(source_code, ast_sexp_list, tok_sexp_list, unicode_offsets=unicode_offsets)

                    # Save the parsed document (printed format) to raw_files
                    data_mgr.dump_data([FilesManager.RAW_FILES, project.full_name, coq_file], coq_document.str_with_space(), IOUtils.Format.txt)

                    # Set meta data
                    coq_document.file_name = coq_file
                    coq_document.project_name = project.full_name
                    coq_document.revision = project.revision

                    coq_documents.append(coq_document)
                except KeyboardInterrupt:
                    cls.logger.warning("Keyboard interrupt!")
                    raise
                except:
                    cls.logger.warning(f"File {coq_file} failed! Exception was: {traceback.format_exc()}")
                    continue
                # end try
            # end for
        # end with

        return coq_documents

    @classmethod
    def verify_tokenizer(cls, tok_sexp_list: List[SexpNode], source_code: str, unicode_offsets: List[int]) -> bool:
        sertok_sentences = SexpAnalyzer.analyze_sertok_sentences(tok_sexp_list, unicode_offsets)
        vernac_sentences = CoqParser.parse_sertok_sentences(sertok_sentences, source_code)

        code_i = 0
        has_error: bool = False

        for sent_i, sentence in enumerate(vernac_sentences):
            for token_i, token in enumerate(sentence.tokens):
                # Check space/comment
                if token.beg_charno != code_i:
                    if not ParserUtils.is_ws_or_comment(source_code[code_i:token.beg_charno]):
                        cls.logger.error(f"Unresolved characters at charno {code_i} to {token.beg_charno}; next expect token {token.content} beginning at charno {token.beg_charno} (lineno {token.lineno}); file content {source_code[code_i:token.beg_charno]};")
                        cls.logger.error(f"assotiated sexp: \n{tok_sexp_list[sent_i][1][token_i].pretty_format()}")
                        has_error = True
                    # end if
                # end if

                # Check token
                code_i = token.beg_charno

                if token.content != source_code[code_i:token.end_charno]:
                    cls.logger.error(f"Mismatch token at charno {code_i} to {token.end_charno}; expect token {token.content} beginning at charno {token.beg_charno} (lineno {token.lineno}); file content {source_code[code_i:token.end_charno]};")
                    cls.logger.error(f"assotiated sexp: \n{tok_sexp_list[sent_i][1][token_i].pretty_format()}")
                    has_error = True
                # end if

                code_i = token.end_charno
        # end for, for

        # Check space/comment at end of file
        if code_i != len(source_code):
            if not ParserUtils.is_ws_or_comment(source_code[code_i:len(source_code)]):
                cls.logger.error(f"Unresolved characters at charno {code_i} to {len(source_code)} (end of file); file content {source_code[code_i:len(source_code)]}")
                has_error = True
            # end if
        # end if

        return not has_error

    @classmethod
    def install_coq_projects(cls, projects: List[Project]) -> None:
        names_projects = {p.full_name: p for p in projects}
        for i, p in enumerate(projects):
            cls.logger.info(f"Installing {p.full_name} ({i}/{len(projects)})")
            cls.install_coq_project(p, names_projects)
        # end for
        return

    @classmethod
    def install_coq_project(cls, project: Project, names_projects: Dict[str, Project]) -> None:
        """
        :requires: the project is cloned and checked-out to the desired version.
        """
        if not project.is_cloned:
            project.clone()
            project.checkout(project.data["sha"], is_forced=True)
        # end if

        # Check if the project is already compiled
        confirmation_file = "lpc-installed.txt"
        confirmation_content = project.revision + " " + BashUtils.run("opam list coq -s", expected_return_code=0).stdout.strip()
        if (project.checkout_dir/confirmation_file).is_file() and IOUtils.load(project.checkout_dir/confirmation_file, "txt") == confirmation_content:
            cls.logger.debug(f"Project {project.full_name} already installed")
            return
        # end if

        project.clean()

        # Install dependencies
        for dependency in project.data.get("dependencies", []):
            dependency_project = names_projects.get(dependency)
            if dependency_project is None:  raise Exception(f"Cannot find dependency {dependency}")
            cls.logger.info(f"For Project {project.full_name}, installing dependency {dependency}")
            cls.install_coq_project(dependency_project, names_projects)
        # end for

        if "build_cmd" not in project.data:  raise Exception(f"Project {project.full_name} does not have build_cmd")
        if "install_cmd" not in project.data:  raise Exception(f"Project {project.full_name} does not have install_cmd")

        with IOUtils.cd(project.checkout_dir):
            # Build
            cls.logger.info(f"Project {project.full_name}: Building with {project.data['build_cmd']}")
            r = BashUtils.run(project.data["build_cmd"])
            if r.return_code != 0:
                raise Exception(f"Compilation failed! Return code is {r.return_code}! stdout:\n{r.stdout}\n; stderr:\n{r.stderr}")
            else:
                cls.logger.debug(f"Compilation finished. Return code is {r.return_code}. stdout:\n{r.stdout}\n; stderr:\n{r.stderr}")
            # end if

            # Install
            cls.logger.info(f"Project {project.full_name}: Installing with {project.data['install_cmd']}")
            r = BashUtils.run(project.data["install_cmd"])
            if r.return_code != 0:
                raise Exception(f"Installation failed! Return code is {r.return_code}! stdout:\n{r.stdout}\n; stderr:\n{r.stderr}")
            else:
                cls.logger.debug(f"Installation finished. Return code is {r.return_code}. stdout:\n{r.stdout}\n; stderr:\n{r.stderr}")
            # end if

            IOUtils.dump(project.checkout_dir / confirmation_file, confirmation_content, "txt")
        # end with
        return

    @classmethod
    def collect_data_indexes(cls, data_mgr: FilesManager, projects: List[Project]) -> NoReturn:
        """
        Split the dataset and record the data indexes for {t1, t2, t3, lo, ta, allgroup} * {train, val, test, all} dataset parts.
        """
        data_mgr.clean_path([FilesManager.DATA_INDEXES])
        data_mgr.resolve([FilesManager.DATA_INDEXES]).mkdir(parents=True)

        # (Random) Split by train/val/test
        cls.logger.info(f"Splitting regular dataset info train/val/test sets with ratio of {Macros.DS_TRAIN_RATIO}/{Macros.DS_VAL_RATIO}/{Macros.DS_TEST_RATIO}")
        cls.logger.info(f"Splitting leave-out dataset info train/val/test sets with ratio of {Macros.DS_LO_TRAIN_RATIO}/{Macros.DS_LO_VAL_RATIO}/{Macros.DS_LO_TEST_RATIO}")

        # Load and sort coq-documents data
        coq_documents: List[CoqDocument] = cls.load_coq_documents(data_mgr)
        coq_documents.sort(key=lambda d: d.get_data_index())

        cls.logger.info(f"Total dataset #doc = {len(coq_documents)}")
        if len(coq_documents) < 10:
            cls.logger.warning(f"Dataset is probably too small: {len(coq_documents)}")
        # end if

        trainevals_data_indexes: Dict[str, Set[str]] = collections.defaultdict(set)

        # Split data for each project, using the same random seed salted with the project name
        for project in projects:
            documents_this_project: List[CoqDocument] = sorted([d for d in coq_documents if d.project_name == project.full_name])

            hasher = hashlib.sha256()
            hasher.update(str.encode(project.full_name))
            hasher.update(str.encode(str(Environment.random_seed)))
            salted_seed = int.from_bytes(hasher.digest(), "big")
            random.seed(salted_seed)
            random.shuffle(documents_this_project)

            if project.data["group"] in [Macros.DS_GROUP_T1, Macros.DS_GROUP_T2, Macros.DS_GROUP_T3]:
                train_ratio, val_ratio, test_ratio = Macros.DS_TRAIN_RATIO, Macros.DS_VAL_RATIO, Macros.DS_TEST_RATIO
            elif project.data["group"] in [Macros.DS_GROUP_LO]:
                train_ratio, val_ratio, test_ratio = Macros.DS_LO_TRAIN_RATIO, Macros.DS_LO_VAL_RATIO, Macros.DS_LO_TEST_RATIO
            else:
                LoggingUtils.log_and_raise(cls.logger, f"Invalid group name {project.data['group']} for {project.full_name}", Exception)
            # end if

            train_val_split_point = int(math.ceil(train_ratio * len(documents_this_project)))
            val_test_split_point = int(math.ceil((train_ratio + val_ratio) * len(documents_this_project)))

            trainevals_data_indexes[Macros.DS_TRAIN].update(set([d.get_data_index() for d in documents_this_project[:train_val_split_point]]))
            trainevals_data_indexes[Macros.DS_VAL].update(set([d.get_data_index() for d in documents_this_project[train_val_split_point:val_test_split_point]]))
            trainevals_data_indexes[Macros.DS_TEST].update(set([d.get_data_index() for d in documents_this_project[val_test_split_point:]]))
        # end for

        trainevals_data_indexes[Macros.DS_TRAINEVAL_ALL] = set.union(*trainevals_data_indexes.values())

        cls.logger.info(f"Train/eval split #doc:\n" + ";\n".join([
            f"{traineval}: {len(data_indexes)}"
            for traineval, data_indexes in trainevals_data_indexes.items()
        ]))

        # Split by groups
        groups_project_names: Dict[str, List[str]] = {group: [p.full_name for p in projects if p.data["group"] == group] for group in Macros.DS_GROUPS}
        groups_data_indexes: Dict[str, Set[str]] = dict()

        for group, project_names in groups_project_names.items():
            documents_this_group: List[CoqDocument] = [d for d in coq_documents if d.project_name in project_names]
            groups_data_indexes[group] = set([d.get_data_index() for d in documents_this_group])
        # end for

        groups_data_indexes[Macros.DS_GROUP_TA] = set.union(groups_data_indexes[Macros.DS_GROUP_T1], groups_data_indexes[Macros.DS_GROUP_T2], groups_data_indexes[Macros.DS_GROUP_T3])
        groups_data_indexes[Macros.DS_GROUP_ALL] = set.union(groups_data_indexes[Macros.DS_GROUP_T1], groups_data_indexes[Macros.DS_GROUP_T2], groups_data_indexes[Macros.DS_GROUP_T3], groups_project_names[Macros.DS_GROUP_LO])

        cls.logger.info(f"Groups split #doc:\n" + ";\n".join([
            f"{group}: {len(data_indexes)}"
            for group, data_indexes in groups_data_indexes.items()
        ]))

        # The final data indexes is cross product of the two splits
        for traineval in Macros.DS_TRAINEVALS + [Macros.DS_TRAINEVAL_ALL]:
            for group in Macros.DS_GROUPS + [Macros.DS_GROUP_TA, Macros.DS_GROUP_ALL]:
                data_indexes = list(set.intersection(groups_data_indexes[group], trainevals_data_indexes[traineval]))
                cls.logger.info(f"{group}-{traineval} #doc = {len(data_indexes)}")

                data_mgr.dump_data([FilesManager.DATA_INDEXES, f"{group}-{traineval}.json"], data_indexes, IOUtils.Format.jsonPretty)
            # end for
        # end for
        return

    RE_PATH_TO_QUALIFIED_PREFIX = re.compile(r"-[QR] (?P<path>[^,]+),(?P<qprefix>\S+)")

    @classmethod
    def collect_lemmas(cls, data_mgr: FilesManager, projects: List[Project], files: List[str] = None):
        data_mgr.clean_path([FilesManager.LEMMAS])
        data_mgr.resolve([FilesManager.LEMMAS]).mkdir(parents=True)

        # Increase recursion limit because the backend sexps are CRAZZZZY deep
        sys.setrecursionlimit(10000)

        # Load coq-documents
        coq_documents: List[CoqDocument] = cls.load_coq_documents(data_mgr)
        if files is not None:  coq_documents = [d for d in coq_documents if d.file_name in files]

        lemmas: List[Lemma] = list()

        # Prepare serapi_options
        project_2_serapi_options: Dict[str, str] = {p.full_name: p.data["serapi_options"] for p in projects}

        errors: List[Tuple[str, str]] = list()

        for doc_i, doc in enumerate(tqdm(coq_documents)):
            try:
                cls.logger.info(f"Collecting from file {doc.get_data_index()} ({doc_i}/{len(coq_documents)}). Collected: {len(lemmas)}")

                # Load AST sexp
                ast_sexp_list: List[SexpNode] = SexpParser.parse_list(data_mgr.load_data([FilesManager.RAW_FILES, doc.get_data_index()[:-2] + ".ast.sexp"], IOUtils.Format.txt))

                # Collect lemmas from this doc
                lemmas_doc: List[Lemma] = cls.collect_lemmas_doc(doc, ast_sexp_list, project_2_serapi_options[doc.project_name])
                lemmas.extend(lemmas_doc)
            except KeyboardInterrupt:
                cls.logger.warning(f"Keyboard Interrupt!")
                raise
            except:
                cls.logger.warning(f"Error while parsing {doc.get_data_index()}: {traceback.format_exc()}")
                cls.logger.warning(f"The script will continue on other files before it returns with failure. Use Ctrl+C to cut it early.")
                errors.append((doc.get_data_index(), traceback.format_exc()))
                continue
            # end try
        # end for

        if len(errors) > 0:
            LoggingUtils.log_and_raise(cls.logger, f"There were {len(errors)} errors during collection.", Exception)
            data_mgr.dump_data([FilesManager.LEMMAS, "errors.txt"], errors, IOUtils.Format.jsonPretty)
        # end if

        # Assign uids
        for lemma_i, lemma in enumerate(lemmas):  lemma.uid = lemma_i

        data_mgr.dump_data([FilesManager.LEMMAS], lemmas, IOUtils.Format.json, is_batched=True, per_batch=5000)
        return

    @classmethod
    def filter_lemmas(cls, data_mgr: FilesManager):
        # Increase recursion limit because the backend sexps are CRAZZZZY deep
        sys.setrecursionlimit(10000)

        data_mgr.clean_path([FilesManager.LEMMAS_FILTERED])
        data_mgr.resolve([FilesManager.LEMMAS_FILTERED]).mkdir(parents=True)

        # Load lemmas
        lemmas: List[Lemma] = data_mgr.load_data([FilesManager.LEMMAS], IOUtils.Format.json, is_batched=True, clz=Lemma)
        heights: List[int] = [l.backend_sexp.height() for l in lemmas]

        depth_cutoff_point = sorted(heights)[int(np.ceil(Macros.LEMMAS_DEPTH_CUTOFF * len(lemmas)))]
        data_indexes_names: List[Tuple[str, str]] = [(l.data_index, l.name) for l in lemmas if l.backend_sexp.height() <= depth_cutoff_point]
        cls.logger.info(f"Cutoff depth is {depth_cutoff_point}, and {len(data_indexes_names)} data are included")

        lemmas_filtered: List[Lemma] = [l for l in lemmas if (l.data_index, l.name) in data_indexes_names]

        # Assign uids
        for lemma_i, lemma in enumerate(lemmas_filtered):  lemma.uid = lemma_i

        data_mgr.dump_data([FilesManager.LEMMAS_FILTERED], lemmas_filtered, IOUtils.Format.json, is_batched=True, per_batch=5000)
        return

    @classmethod
    def collect_definitions(cls, data_mgr: FilesManager):
        data_mgr.clean_path([FilesManager.DEFINITIONS])
        data_mgr.resolve([FilesManager.DEFINITIONS]).mkdir(parents=True)

        # Load coq-documents
        coq_documents: List[CoqDocument] = cls.load_coq_documents(data_mgr)

        definitions: List[Definition] = list()

        errors: List[Tuple[str, str]] = list()

        for doc_i, doc in enumerate(tqdm(coq_documents)):
            try:
                # Load AST sexp
                ast_sexp_list: List[SexpNode] = SexpParser.parse_list(data_mgr.load_data([FilesManager.RAW_FILES, doc.get_data_index()[:-2] + ".ast.sexp"], IOUtils.Format.txt))
                definitions_doc: List[Definition] = cls.collect_definitions_doc(doc, ast_sexp_list)

                definitions.extend(definitions_doc)
            except KeyboardInterrupt:
                cls.logger.warning(f"Keyboard Interrupt!")
                raise
            except:
                cls.logger.warning(f"Error while parsing {doc.get_data_index()}: {traceback.format_exc()}")
                cls.logger.warning(f"The script will continue on other files before it returns with failure. Use Ctrl+C to cut it early.")
                errors.append((doc.get_data_index(), traceback.format_exc()))
                continue
            # end try
        # end for

        if len(errors) > 0:
            LoggingUtils.log_and_raise(cls.logger, f"There were {len(errors)} errors during collection.", Exception)
            data_mgr.dump_data([FilesManager.DEFINITIONS, "errors.txt"], errors, IOUtils.Format.jsonPretty)
        # end if

        data_mgr.dump_data([FilesManager.DEFINITIONS, "definitions.json"], definitions, IOUtils.Format.json)
        return

    @classmethod
    def collect_lemmas_backend_sexp_transformations(cls, data_mgr: FilesManager):
        data_mgr.clean_path([cls.TASK_LEMMA_BACKEND_SEXP_TRANSFORMATIONS])
        data_mgr.resolve([cls.TASK_LEMMA_BACKEND_SEXP_TRANSFORMATIONS]).mkdir(parents=True)

        # Increase recursion limit because the backend sexps are CRAZZZZY deep
        sys.setrecursionlimit(10000)

        lemmas_filtered: List[Lemma] = data_mgr.load_data([FilesManager.LEMMAS_FILTERED], IOUtils.Format.json, is_batched=True, clz=Lemma)

        # Main stream transformations, applied one after another
        levels_lemmas_bsexp_transformed: Dict[str, List[SexpNode]] = dict()

        last_level: Optional[str] = None  # None means original
        for level in LemmaBackendSexpTransformers.LEVELS:
            cls.logger.info(f"Doing {last_level if last_level is not None else 'orig'} -> {level} transformation")
            levels_lemmas_bsexp_transformed[level] = list()

            for lemma_i, lemma in enumerate(tqdm(lemmas_filtered)):
                orig_sexp = lemma.backend_sexp if last_level is None else levels_lemmas_bsexp_transformed[last_level][lemma_i]
                bsexp_transformed = LemmaBackendSexpTransformers.transform(level, copy.deepcopy(orig_sexp))
                levels_lemmas_bsexp_transformed[level].append(bsexp_transformed)
            # end for

            last_level = level

            data_mgr.dump_data([cls.TASK_LEMMA_BACKEND_SEXP_TRANSFORMATIONS, level, "transformed"], levels_lemmas_bsexp_transformed[level], IOUtils.Format.json, is_batched=True, per_batch=5000)
        # end for

        # Other special transformation, directly applied on original trees
        for tr_name in LemmaBackendSexpTransformers.SPECIALS:
            cls.logger.info(f"Doing orig -> {tr_name} transformation")
            bsexp_transformed_list = list()
            for lemma_i, lemma in enumerate(tqdm(lemmas_filtered)):
                orig_sexp = lemma.backend_sexp
                bsexp_transformed = LemmaBackendSexpTransformers.transform(tr_name, copy.deepcopy(orig_sexp))

                bsexp_transformed_list.append(bsexp_transformed)
            # end for

            data_mgr.dump_data([cls.TASK_LEMMA_BACKEND_SEXP_TRANSFORMATIONS, tr_name, "transformed"], bsexp_transformed_list, IOUtils.Format.json, is_batched=True, per_batch=5000)
        # end for
        return

    @classmethod
    def collect_lemmas_foreend_sexp_transformations(cls, data_mgr: FilesManager):
        data_mgr.clean_path([cls.TASK_LEMMA_FOREEND_SEXP_TRANSFORMATIONS])
        data_mgr.resolve([cls.TASK_LEMMA_FOREEND_SEXP_TRANSFORMATIONS]).mkdir(parents=True)

        # Increase recursion limit because the backend sexps are CRAZZZZY deep
        sys.setrecursionlimit(10000)

        lemmas_filtered: List[Lemma] = data_mgr.load_data([FilesManager.LEMMAS_FILTERED], IOUtils.Format.json, is_batched=True, clz=Lemma)

        # Main stream transformations, applied one after another
        levels_lemmas_fsexp_transformed: Dict[str, List[SexpNode]] = dict()

        last_level: Optional[str] = None  # None means original
        for level in LemmaForeendSexpTransformers.LEVELS:
            cls.logger.info(f"Doing {last_level if last_level is not None else 'orig'} -> {level} transformation")
            levels_lemmas_fsexp_transformed[level] = list()

            for lemma_i, lemma in enumerate(tqdm(lemmas_filtered)):
                orig_sexp = lemma.ast_sexp if last_level is None else levels_lemmas_fsexp_transformed[last_level][lemma_i]
                fsexp_transformed = LemmaForeendSexpTransformers.transform(level, copy.deepcopy(orig_sexp))

                levels_lemmas_fsexp_transformed[level].append(fsexp_transformed)
            # end for

            last_level = level

            data_mgr.dump_data([cls.TASK_LEMMA_FOREEND_SEXP_TRANSFORMATIONS, level, "transformed"], levels_lemmas_fsexp_transformed[level], IOUtils.Format.json, is_batched=True, per_batch=5000)
        # end for

        # Other special transformation, directly applied on level 0 trees
        for tr_name in LemmaForeendSexpTransformers.SPECIALS:
            cls.logger.info(f"Doing {LemmaForeendSexpTransformers.LEVEL_0} -> {tr_name} transformation")
            fsexp_transformed_list = list()
            for lemma_i, lemma in enumerate(tqdm(lemmas_filtered)):
                orig_sexp = levels_lemmas_fsexp_transformed[LemmaForeendSexpTransformers.LEVEL_0][lemma_i]
                fsexp_transformed = LemmaForeendSexpTransformers.transform(tr_name, copy.deepcopy(orig_sexp))

                fsexp_transformed_list.append(fsexp_transformed)
            # end for

            data_mgr.dump_data([cls.TASK_LEMMA_FOREEND_SEXP_TRANSFORMATIONS, tr_name, "transformed"], fsexp_transformed_list, IOUtils.Format.json, is_batched=True, per_batch=5000)
        # end for
        return

    VTYPES_LEMMA = [SexpInfo.VernacConsts.type_start_theorem_proof]
    VTYPES_MODULE_BEG = [SexpInfo.VernacConsts.type_define_module]
    VTYPES_MODULE_END = [SexpInfo.VernacConsts.type_end_segment]
    VTYPES_DEFINITIONS = [SexpInfo.VernacConsts.type_definition]

    @classmethod
    def collect_lemmas_doc(cls,
            doc: CoqDocument,
            ast_sexp_list: List[SexpNode],
            serapi_options: str,
    ) -> List[Lemma]:
        lemmas_doc: List[Lemma] = list()
        data_index = doc.get_data_index()

        # Maintain a stack of module
        modules: List[str] = list()

        # Prepare qualified name prefix
        qprefix_this_doc = "./" + doc.file_name[:-2]  # Remove .v
        for m in cls.RE_PATH_TO_QUALIFIED_PREFIX.finditer(serapi_options):
            path = m.group("path")
            if path != ".":  path = "./" + path
            qprefix = m.group("qprefix")

            if qprefix_this_doc.startswith(path):
                qprefix_this_doc = qprefix + qprefix_this_doc[len(path):]
                break
            # end if
        # end for
        if qprefix_this_doc.startswith("./"):  qprefix_this_doc = qprefix_this_doc[len("./"):]
        qprefix_this_doc = qprefix_this_doc.replace("/", ".")

        for sent_i, sent in enumerate(doc.sentences):
            ast_sexp = ast_sexp_list[sent_i]
            vernac = SexpAnalyzer.analyze_vernac(ast_sexp)

            if vernac.vernac_type in cls.VTYPES_MODULE_BEG:
                # (VernacExpr()(VernacDefineModule()  (  (   v   ( Id <module name>)) ...
                #  0         1 2 20               21  22 220  2201    22011
                module_name = vernac.vernac_sexp[2][2][0][1][1].content_no_quote
                modules.append(module_name)
            elif vernac.vernac_type in cls.VTYPES_MODULE_END:
                # (VernacExpr()(VernacEndSegment  (  (   v   ( Id <module name>)) ...
                #  0         1 2 20               21 210  2101    21011
                try:
                    module_name = vernac.vernac_sexp[2][1][0][1][1].content_no_quote
                except:
                    print(vernac.vernac_sexp.pretty_format())
                    raise
                # end try
                if len(modules) > 0 and module_name == modules[-1]:  modules.pop()  # EndModule and EndSection share the same vernac type
            elif vernac.vernac_type in cls.VTYPES_LEMMA:
                # (VernacExpr()(VernacStartTheoremProof Lemma ( ( ( ( ( v (       Id <lemma name>))
                #  0         1 2 20                     21   22   2200000 2200001    22000011
                lemma = Lemma()
                lemma.data_index = data_index

                lemma.name = vernac.vernac_sexp[2][2][0][0][0][0][1][1].content_no_quote
                lemma.qname = qprefix_this_doc + "." + ".".join(modules + [lemma.name])

                # Find lemma content, after the first token matching the lemma name
                tok_i = 0
                for tok in sent.tokens:
                    if tok.content == lemma.name:  break
                    tok_i += 1
                # end for
                if tok_i == len(sent.tokens):  LoggingUtils.log_and_raise(cls.logger, f"Lemma name {lemma.name} didn't appear in the source code {sent.str_with_space()}", Exception)

                lemma.vernac_command = sent.tokens[:tok_i]
                lemma.statement = sent.tokens[tok_i + 1:]
                lemma.ast_sexp = vernac.vernac_sexp

                lemmas_doc.append(lemma)
            # end if
        # end for

        # Use sername to get the backend representations
        lemma_qnames: str = "".join([l.qname + "\n" for l in lemmas_doc])
        lemma_qnames_file = BashUtils.get_temp_file()
        IOUtils.dump(lemma_qnames_file, lemma_qnames, IOUtils.Format.txt)

        lemma_qnames_backend_sexps_str: str = BashUtils.run(f"sername --require-lib={qprefix_this_doc} {lemma_qnames_file}", expected_return_code=0).stdout
        IOUtils.rm(lemma_qnames_file)
        for qname_backend_sexp_str in lemma_qnames_backend_sexps_str.splitlines():
            qname, backend_sexp_str = qname_backend_sexp_str.split(":", 1)
            backend_sexp = SexpParser.parse(backend_sexp_str)

            for lemma in lemmas_doc:
                if lemma.qname == qname:
                    lemma.backend_sexp = backend_sexp
                    break
                # end if
            # end for
        # end for

        lemmas_doc = [l for l in lemmas_doc if l.backend_sexp is not None]
        return lemmas_doc

    @classmethod
    def collect_definitions_doc(cls,
            doc: CoqDocument,
            ast_sexp_list: List[SexpNode],
    ) -> List[Definition]:
        definitions_doc: List[Definition] = list()
        data_index = doc.get_data_index()
        for sent_i, sent in enumerate(doc.sentences):
            ast_sexp = ast_sexp_list[sent_i]
            vernac = SexpAnalyzer.analyze_vernac(ast_sexp)

            if vernac.vernac_type in cls.VTYPES_DEFINITIONS:
                # (VernacExpr()( VernacDefinition (  NoDischarge Definition) (  (   (    v     (     Name   (      Id      codom   ))) ...
                #  0         1 2 20               21 210         211         22 220 2200 22000 22001 220010 220011 2200110 2200111
                try:
                    if vernac.vernac_sexp[2][1][0].content == "NoDischarge" and vernac.vernac_sexp[2][1][1].content == "Definition":
                        definition = Definition()
                        definition.data_index = data_index

                        definition.name = vernac.vernac_sexp[2][2][0][0][1][1][1].content_no_quote

                        definitions_doc.append(definition)
                    # end if
                except IllegalSexpOperationException:
                    continue
                # end try
            # end if
        # end for
        return definitions_doc

    @classmethod
    def extract_data_project(cls,
            project_path: Path,
            files: Optional[List[str]],
            exclude_files: Optional[List[str]],
            exclude_pattern: Optional[str],
            serapi_options: str,
            output_path: Path,
    ):
        # 1. Prepare output path
        if output_path.is_dir():
            cls.logger.warning(f"{output_path} already exists, will overwrite the files.")
        elif output_path.is_file():
            LoggingUtils.log_and_raise(cls.logger, f"{output_path} already exists as a file. Aborting.", Exception)
        else:
            IOUtils.mk_dir(output_path)
        # end if

        # 2. Extract documents, tok.sexp and ast.sexp
        coq_documents: Dict[str, CoqDocument] = collections.OrderedDict()
        ast_sexp_lists: Dict[str, List[SexpNode]] = dict()
        tok_sexp_lists: Dict[str, List[SexpNode]] = dict()

        with IOUtils.cd(project_path):
            coq_files: List[str] = BashUtils.run(f"find -name '*.v' -type f").stdout.split("\n")[:-1]
            coq_files = [coq_file[2:] for coq_file in coq_files]

            if files is not None:
                coq_files = [f for f in coq_files if f in files]
            # end if

            if exclude_files is not None:
                coq_files = [f for f in coq_files if f not in exclude_files]
            # end if

            if exclude_pattern is not None:
                re_exclude_pattern = re.compile(exclude_pattern)
                coq_files = [f for f in coq_files if not re_exclude_pattern.fullmatch(f)]
            # end if

            for i, coq_file in enumerate(tqdm(coq_files)):
                try:
                    # Read file
                    with open(coq_file, "r", newline="") as f:
                        source_code = f.read()
                    # end with

                    # Get unicode offsets
                    unicode_offsets = ParserUtils.get_unicode_offsets(source_code)

                    # Call SerAPI
                    ast_sexp_str: str = BashUtils.run(f"sercomp {serapi_options} --mode=sexp -- {coq_file}", expected_return_code=0).stdout
                    tok_sexp_str: str = BashUtils.run(f"sertok {serapi_options} -- {coq_file}", expected_return_code=0).stdout

                    # Parse ast sexp
                    ast_sexp_list: List[SexpNode] = SexpParser.parse_list(ast_sexp_str)
                    tok_sexp_list: List[SexpNode] = SexpParser.parse_list(tok_sexp_str)

                    # Parse the document
                    coq_document = CoqParser.parse_document(source_code, ast_sexp_list, tok_sexp_list, unicode_offsets=unicode_offsets)

                    # Set meta data
                    coq_document.file_name = coq_file
                    coq_document.project_name = project_path.name

                    coq_documents[coq_file] = coq_document
                    ast_sexp_lists[coq_file] = ast_sexp_list
                    tok_sexp_lists[coq_file] = tok_sexp_list
                except KeyboardInterrupt:
                    cls.logger.warning("Keyboard interrupt!")
                    raise
                except:
                    cls.logger.warning(f"File {coq_file} failed! Exception was: {traceback.format_exc()}")
                    continue
                # end try
            # end for

            # 3. Extract and save lemmas and definitions
            lemmas: List[Lemma] = list()
            definitions: List[Definition] = list()

            # Increase recursion limit because the backend sexps are CRAZZZZY deep
            sys.setrecursionlimit(10000)

            for file_path, doc in tqdm(coq_documents.items()):
                ast_sexp_list = ast_sexp_lists[file_path]
                lemmas_doc = cls.collect_lemmas_doc(doc, ast_sexp_list, serapi_options)
                lemmas.extend(lemmas_doc)
                definitions_doc = cls.collect_definitions_doc(doc, ast_sexp_list)
                definitions.extend(definitions_doc)
            # end for

            IOUtils.dump(output_path/"lemmas.json", IOUtils.jsonfy(lemmas), IOUtils.Format.json)
            IOUtils.dump(output_path/"definitions.json", IOUtils.jsonfy(definitions), IOUtils.Format.json)
        # end with
        return

    @classmethod
    def extract_data_from_corpus(cls,
            corpus_path: Path,
            trainevals: List[str],
            groups: List[str],
            output_path: Path,
    ):
        # 1. Prepare output path
        if output_path.is_dir():
            cls.logger.warning(f"{output_path} already exists, will overwrite the files.")
        elif output_path.is_file():
            LoggingUtils.log_and_raise(cls.logger, f"{output_path} already exists as a file. Aborting.", Exception)
        else:
            IOUtils.mk_dir(output_path)
        # end if

        assert all([traineval in Macros.DS_TRAINEVALS for traineval in trainevals])
        assert all([group in Macros.DS_GROUPS+[Macros.DS_GROUP_TA] for group in groups])

        data_mgr = FilesManager(corpus_path)

        # 2. Load lemmas and definitions
        lemmas_filtered: List[Lemma] = data_mgr.load_data([FilesManager.LEMMAS_FILTERED], IOUtils.Format.json, is_batched=True, clz=Lemma)
        definitions: List[Definition] = data_mgr.load_data([FilesManager.DEFINITIONS, "definitions.json"], IOUtils.Format.json, clz=Definition)

        # 3. Output to output_path for each combination of traineval and group
        for traineval in trainevals:
            for group in groups:
                IOUtils.mk_dir(output_path/f"{group}-{traineval}")
                data_indexes = IOUtils.load(Macros.project_dir/"training"/f"{group}-{traineval}.json", IOUtils.Format.json)
                IOUtils.dump(output_path/f"{group}-{traineval}/lemmas.json", IOUtils.jsonfy([l for l in lemmas_filtered if l.data_index in data_indexes]), IOUtils.Format.json)
                IOUtils.dump(output_path/f"{group}-{traineval}/definitions.json", IOUtils.jsonfy([d for d in definitions if d.data_index in data_indexes]), IOUtils.Format.json)
            # end for
        # end for
        return
