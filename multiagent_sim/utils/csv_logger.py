import csv
import os


class CSVLogger:
    def __init__(
        self, filepath: str, columns: list[str], header_lines: list[str] = None
    ):
        self.filepath = self._get_versioned_filepath(filepath)
        self.columns = columns
        self.header_lines = header_lines if header_lines is not None else []
        # Write headers only once
        with open(self.filepath, "x", newline="") as f:
            for line in self.header_lines:
                f.write(f"# {line}\n")
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()

    def _get_versioned_filepath(self, filepath: str) -> str:
        base, ext = os.path.splitext(filepath)
        version = 0
        new_filepath = filepath
        while os.path.exists(new_filepath):
            version += 1
            new_filepath = f"{base}_{version}{ext}"
        return new_filepath

    def log(self, **kwargs):
        # Only write columns specified in self.columns
        row = {col: kwargs.get(col, "") for col in self.columns}
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(row)
