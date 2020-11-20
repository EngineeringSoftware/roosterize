import math
import traceback
from pathlib import Path
from typing import Any, Callable, Iterator, List, NoReturn, Optional, Union

from seutil import IOUtils, LoggingUtils
from tqdm import tqdm

logger = LoggingUtils.get_logger(__name__)


class FilesManager:
    """
    Handles the loading/dumping of files in a dataset.
    """

    ALL_LEMMAS_BACKEND_SEXP_TRANSFORMATIONS = "all-lemmas-bsexp-transformations"
    ALL_LEMMAS_FOREEND_SEXP_TRANSFORMATIONS = "all-lemmas-fsexp-transformations"
    COQ_DOCUMENTS = "coq-documents"
    LEMMAS = "lemmas"
    LEMMAS_BACKEND_SEXP_TRANSFORMATIONS = "lemmas-bsexp-transformations"
    LEMMAS_FILTERED = "lemmas-filtered"
    LEMMAS_FOREEND_SEXP_TRANSFORMATIONS = "lemmas-fsexp-transformations"
    DATA_INDEXES = "data-indexes"
    RAW_FILES = "raw-files"
    ORIGINAL_FILES = "original-files"
    DEFINITIONS = "definitions"

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return

    def clean_path(self, rel_path: Union[str, List[str]]):
        abs_path = self.data_dir / self.assemble_rel_path(rel_path)
        if abs_path.exists():
            logger.info(f"Removing existing things at {abs_path}")
            IOUtils.rm(abs_path)
        # end if
        return

    @classmethod
    def is_json_format(cls, fmt: IOUtils.Format) -> bool:
        return fmt in [IOUtils.Format.json, IOUtils.Format.jsonPretty, IOUtils.Format.jsonNoSort]

    def dump_data(
            self,
            rel_path: Union[str, List[str]],
            data: Any,
            fmt: IOUtils.Format,
            is_batched: bool = False,
            per_batch: int = 100,
            exist_ok: bool = False,
    ):
        abs_path = self.data_dir / self.assemble_rel_path(rel_path)
        if abs_path.exists() and not exist_ok:
            raise IOError(f"Cannot rewrite existing data at {abs_path}")

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        if not is_batched:
            if self.is_json_format(fmt):
                data = IOUtils.jsonfy(data)
            IOUtils.dump(abs_path, data, fmt)
        else:
            # In batched mode, the data need to be slice-able and sizable
            IOUtils.rm(abs_path)
            abs_path.mkdir(parents=True)

            for batch_i in tqdm(range(math.ceil(len(data) / per_batch))):
                data_batch = data[per_batch * batch_i: per_batch * (batch_i + 1)]
                if self.is_json_format(fmt):
                    data_batch = IOUtils.jsonfy(data_batch)
                IOUtils.dump(abs_path / f"batch-{batch_i}.{fmt.get_extension()}", data_batch, fmt)
        return

    def load_data(
            self,
            rel_path: Union[str, List[str]],
            fmt: IOUtils.Format,
            is_batched: bool = False,
            clz=None,
    ) -> Any:
        if self.is_json_format(fmt) and clz is None:
            logger.warning(f"Load data from {rel_path} with json format, but did not specify clz (at {traceback.format_stack()})")

        abs_path = self.data_dir / self.assemble_rel_path(rel_path)
        if not abs_path.exists():
            raise IOError(f"Cannot find data at {abs_path}")

        if not is_batched:
            data = IOUtils.load(abs_path, fmt)
            if self.is_json_format(fmt) and clz is not None:
                data = IOUtils.dejsonfy(data, clz)
            return data
        else:
            data = list()
            batch_numbers = sorted([int(str(f.stem).split("-")[1]) for f in abs_path.iterdir()])
            for batch_number in tqdm(batch_numbers):
                batch_file = abs_path / f"batch-{batch_number}.{fmt.get_extension()}"
                data_batch = IOUtils.load(batch_file, fmt)
                if self.is_json_format(fmt) and clz is not None:
                    data_batch = IOUtils.dejsonfy(data_batch, clz)
                data.extend(data_batch)
            return data

    def iter_batched_data(
            self,
            rel_path: Union[str, List[str]],
            fmt: IOUtils.Format,
            clz=None,
    ) -> Iterator:
        if self.is_json_format(fmt) and clz is None:
            logger.warning(f"Load data from {rel_path} with json format, but did not specify clz")

        abs_path = self.data_dir / self.assemble_rel_path(rel_path)
        if not abs_path.exists():
            raise IOError(f"Cannot find data at {abs_path}")

        batch_numbers = sorted([int(str(f.stem).split("-")[1]) for f in abs_path.iterdir()])
        for batch_number in batch_numbers:
            batch_file = abs_path / f"batch-{batch_number}.{fmt.get_extension()}"
            for data_entry in IOUtils.load(batch_file, fmt):
                if self.is_json_format(fmt) and clz is not None:
                    data_entry = IOUtils.dejsonfy(data_entry, clz)
                # end if
                yield data_entry

    def dump_ckpt(
            self,
            rel_path: Union[str, List[str]],
            obj: Any,
            ckpt_id: int,
            dump_func: Callable[[Any, str], NoReturn],
            ckpt_keep_max: int = 5,
    ) -> NoReturn:
        abs_path = self.data_dir / self.assemble_rel_path(rel_path)
        abs_path.mkdir(parents=True, exist_ok=True)

        ckpt_file_name = str(abs_path / str(ckpt_id))
        dump_func(obj, ckpt_file_name)

        # Remove older checkpoints
        if ckpt_keep_max != -1:
            ckpt_ids = [int(str(f.name)) for f in abs_path.iterdir()]
            for ckpt_id in sorted(ckpt_ids)[:-ckpt_keep_max]:
                IOUtils.rm(abs_path / str(ckpt_id))
        return

    def load_ckpt(
            self,
            rel_path: Union[str, List[str]],
            load_func: Callable[[str], Any],
            ckpt_id: Optional[int] = None,
    ) -> Any:
        abs_path = self.data_dir / self.assemble_rel_path(rel_path)
        if not abs_path.exists():
            raise IOError(f"Cannot find data at {abs_path}")

        if ckpt_id is None:
            # Find the latest ckpt
            ckpt_ids = [int(str(f.name)) for f in abs_path.iterdir()]
            ckpt_id = max(ckpt_ids)
            logger.info(f"Loading the latest checkpoint {ckpt_id} at {abs_path}")

        return load_func(str(abs_path / str(ckpt_id)))

    def resolve(self, rel_path: Union[str, List[str]]) -> Path:
        return self.data_dir / self.assemble_rel_path(rel_path)

    @classmethod
    def assemble_rel_path(cls, rel_path: Union[str, List[str]]) -> str:
        if not isinstance(rel_path, str):
            rel_path = "/".join(rel_path)
        return rel_path
