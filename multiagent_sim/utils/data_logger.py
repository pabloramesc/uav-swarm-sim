import numpy as np
import datetime
import os


def get_unique_log_path(log_folder: str, log_file: str) -> str:
    """
    Returns a unique log file path by appending _n if the file already exists.
    """
    base_name, ext = os.path.splitext(os.path.basename(log_file))
    log_path = os.path.join(log_folder, base_name + ext)
    counter = 1
    while os.path.exists(log_path):
        log_path = os.path.join(log_folder, f"{base_name}_{counter}{ext}")
        counter += 1
    return log_path


class DataLogger:
    def __init__(
        self,
        columns: list[str],
        log_file: str = None,
        log_folder: str = "log",
        dump_every: int = 100,
    ):
        if not os.path.exists(log_folder):
            os.makedirs(log_folder, exist_ok=True)

        if log_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"log_{timestamp}.npz"

        if not log_file.endswith(".npz"):
            raise ValueError("Log file must have a .npz extension")

        log_path = get_unique_log_path(log_folder, log_file)

        self.columns = columns
        self.log_file = log_path
        self.dump_every = dump_every
        self._data = []
        self._step = 0

    def append(self, row):
        self._data.append(row)
        self._step += 1
        if self._step % self.dump_every == 0:
            self.dump()

    def dump(self):
        arr = np.array(self._data)
        np.savez_compressed(self.log_file, columns=self.columns, data=arr)

    def clear(self):
        self._data = []
        self._step = 0


def load_log_file(log_file: str) -> dict[str, np.ndarray]:
    """
    Loads any .npz log file and returns a dict mapping column names to numpy arrays.
    """
    data = np.load(log_file, allow_pickle=True)
    columns = data["columns"]
    arr = data["data"]
    columns = [str(col) for col in columns]
    return {col: arr[:, i] for i, col in enumerate(columns)}