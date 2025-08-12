import numpy as np

def run_tsne(x: np.ndarray, **kwargs):
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, **kwargs).fit_transform(x)

from typing import Sequence, Set
from analysis.preprocess.mem_record import ProcessedMemRecord

def get_task_intersections_from_records(
    records: Sequence[ProcessedMemRecord],
    image_tasks: dict[int, set[int]]
) -> set[frozenset[int]]:
    """
    Finds all unique task intersections that are present in a list of records.
    """
    intersections = set()
    for record in records:
        if record.img_id in image_tasks:
            intersections.add(frozenset(image_tasks[record.img_id]))
    return intersections

def get_records_from_task_intersection(
    records: Sequence[ProcessedMemRecord],
    intersection: set[int],
    image_tasks: dict[int, set[int]]
) -> list[ProcessedMemRecord]:
    """
    Filters a list of records to only those whose image_id belongs to a specific task intersection.
    """
    filtered_records = []
    for record in records:
        if record.img_id in image_tasks and image_tasks[record.img_id] == intersection:
            filtered_records.append(record)
    return filtered_records
