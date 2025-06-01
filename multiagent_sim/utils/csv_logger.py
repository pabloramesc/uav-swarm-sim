import csv

class CSVLogger:
    def __init__(self, filepath, columns, header_lines=None):
        self.filepath = filepath
        self.columns = columns
        self.header_lines = header_lines if header_lines is not None else []
        # Write headers only once
        with open(self.filepath, "w", newline="") as f:
            for line in self.header_lines:
                f.write(f"# {line}\n")
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()

    def log(self, **kwargs):
        # Only write columns specified in self.columns
        row = {col: kwargs.get(col, "") for col in self.columns}
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(row)