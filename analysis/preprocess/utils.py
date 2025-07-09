from pathlib import Path
import pickle
from typing import Any, List

from models.probes.memory_probe import *


def load_pickles(pickle_paths: List[str | Path]) -> List[Any]:
    records: List[Any] = []
    for p in pickle_paths:
        p = Path(p)
        with p.open("rb") as f:
            records.extend(pickle.load(f))
    return records

def load_pickles_from_folder(
    folder: str | Path,
    pattern: str = "mem_epoch*_rank*.pkl",
    verbose: bool = True,
) -> List[Any]:
    """
    Load **all** pickle files in `folder` that match `pattern`
    (e.g. mem_epoch000_rank0.pkl, …_rank1.pkl, …).

    Parameters
    ----------
    folder  : directory that contains the pickles.
    pattern : glob pattern; default matches the rank-split files.
    verbose : if True, prints how many files were found / loaded.

    Returns
    -------
    List[Any]: concatenated records from every pickle.
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if verbose:
        print(f"Found {len(files)} pickle files in {folder}")

    if not files:
        return []

    return load_pickles(files)