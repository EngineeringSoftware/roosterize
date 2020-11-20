from typing import Optional
from pathlib import Path, PurePath

from seutil import IOUtils


class RoosterizeDirUtils:
    """
    Utility functions to manage .roosterize directories.
    """

    @classmethod
    def auto_infer_project_root(cls, optional_file: Optional[Path] = None) -> Path:
        """
        Automatically infers the appropriate .roosterize directory for the project, which
        should locate in the same directory as _CoqProject does. If a file is provided,
        try to look for the nearest directory that contains the file and has a _CoqProject;
        otherwise, start looking from the current directory instead.

        Returns:
            The inferred .roosterize directory path.
        """
        curp = optional_file
        if curp is None:
            curp = Path.cwd()

        # Find the latest _CoqProject directory
        while True:
            if len(curp.parts) <= 1:
                raise IOError("Cannot find _CoqProject")

            if not curp.is_dir():
                curp = curp.parent
                continue

            if (curp / "_CoqProject").is_file():
                break
            else:
                curp = curp.parent
                continue

        return curp

    # @classmethod
    # def get_local_cache_dir(cls, prj_root: Path):
    #     return prj_root / ".roosterize" / "files"

    @classmethod
    def get_local_model_dir(cls, prj_root: Path):
        return prj_root / ".roosterize" / "model"

    @classmethod
    def get_global_model_dir(cls):
        return Path.home() / ".roosterize" / "model"

    @classmethod
    def get_local_config_file(cls, prj_root: Path):
        return prj_root / ".roosterizerc"

    @classmethod
    def get_global_config_file(cls):
        return Path.home() / ".roosterizerc"

    # @classmethod
    # def get_cache_rel_path(cls, file_path: Path, prj_root: Path) -> PurePath:
    #     """
    #     Gets the relative path (relative to cache_dir) of the cache file.
    #     """
    #     rel_path = file_path.relative_to(prj_root)
    #     # Flatten multiple levels to one level
    #     flatten_str = ".".join([p for p in rel_path.parts]) + ".cache"
    #     return PurePath(flatten_str)
    #
    # @classmethod
    # def get_cache_cksum_rel_path(cls, file_path: Path, prj_root: Path) -> PurePath:
    #     """
    #     Gets the relative path (relative to cache_dir) of the cache checksum file.
    #     """
    #     rel_path = file_path.relative_to(prj_root)
    #     # Flatten multiple levels to one level
    #     flatten_str = ".".join([p for p in rel_path.parts]) + ".cksum"
    #     return PurePath(flatten_str)
