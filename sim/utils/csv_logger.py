import csv
import os
from typing import List, Literal


class CSVLogger:
    def __init__(
        self,
        filepath: str,
        columns: List[str],
        header_lines: List[str] | None = None,
        if_exists: Literal["error", "overwrite", "version"] = "version",
    ):
        """
        Parameters
        ----------
        filepath : str
            Target CSV file path.
        columns : List[str]
            Column headers.
        header_lines : List[str] | None
            Optional header lines (prefixed with '#').
        if_exists : str
            How to handle an existing file:
            - "error": raise FileExistsError
            - "overwrite": overwrite the file
            - "version": create versioned file name (default)
        """
        self.filepath = self._resolve_filepath(filepath, if_exists)
        self.columns = columns
        self._initialize_file(header_lines if header_lines is not None else [])

    def _resolve_filepath(self, filepath: str, if_exists: str) -> str:
        if not os.path.exists(filepath):
            return filepath

        if if_exists == "error":
            raise FileExistsError(f"File already exists: {filepath}")

        if if_exists == "overwrite":
            return filepath  # just overwrite later

        if if_exists == "version":
            return self._get_versioned_filepath(filepath)

        raise ValueError(f"Invalid if_exists mode: {if_exists}")

    def _get_versioned_filepath(self, filepath: str) -> str:
        base, ext = os.path.splitext(filepath)
        version = 1
        new_filepath = f"{base}_{version}{ext}"
        while os.path.exists(new_filepath):
            version += 1
            new_filepath = f"{base}_{version}{ext}"
        return new_filepath

    def _initialize_file(self, header_lines: List[str]):
        with open(self.filepath, "w", newline="") as f:
            for line in header_lines:
                f.write(f"# {line}\n")
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()

    def log(self, **kwargs):
        row = {col: kwargs.get(col, "") for col in self.columns}
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(row)
