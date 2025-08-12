# analysis/core/task_maps.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Iterable, DefaultDict

from upsetplot import UpSet, from_memberships
import pandas as pd, matplotlib.pyplot as plt, numpy as np
import seaborn as sns

def load_single_task(
    json_path: str | Path,
) -> Tuple[Set[int],                        # images in this file
           Dict[int, Set[int]],             # img_id → {label_id,…}
           Dict[int, Set[int]]]:            # label_id → {img_id,…}
    """
    Read one *singular* COCO JSON (e.g. ``test_task_2.json``) and extract
    image-label relationships.

    Returns
    -------
    image_set      : set of image IDs present
    image_labels   : {image_id -> set(category_id)}
    label_images   : {category_id -> set(image_id)}
    """
    json_path = Path(json_path)
    with json_path.open("r") as f:
        coco = json.load(f)

    image_set: Set[int] = {img["id"] for img in coco["images"]}
    image_labels: DefaultDict[int, Set[int]] = defaultdict(set)
    label_images: DefaultDict[int, Set[int]] = defaultdict(set)

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        lab_id = ann["category_id"]
        image_labels[img_id].add(lab_id)
        label_images[lab_id].add(img_id)

    return image_set, dict(image_labels), dict(label_images)

@dataclass
class TaskImages:
    task_id: int
    images: Set[int]

def build_global_task_images_maps(task_images_list: Iterable[TaskImages]
                                  ) -> Tuple[Dict[int, Set[int]], # task_id -> images
                                             Dict[int, Set[int]] # image_id -> tasks
                                             ]:
    task2imgs: Dict[int, Set[int]]   = defaultdict(set)
    img2tasks: Dict[int, Set[int]]   = defaultdict(set)

    for task_images_prop in task_images_list:
        task_id = task_images_prop.task_id
        imgs = task_images_prop.images

        task2imgs[task_id].update(imgs)
        for img in imgs:
            img2tasks[img].add(task_id)

    return dict(task2imgs), dict(img2tasks)

@dataclass
class LabelImagesProperty:
    img2lab: Dict[int, Set[int]]  # image_id -> {label_id,…}
    lab2img: Dict[int, Set[int]]  # label_id -> {image_id,…}

def build_global_label_images_maps(
    label_images_list: Iterable[LabelImagesProperty]
) -> Tuple[
    Dict[int, Set[int]],   # label_id -> images
    Dict[int, Set[int]],   # image_id -> labels
]:
    """
    Build global maps for label-image relationships.

    Returns
    -------
    label_images  : {label_id -> set(image_id)}
    image_labels  : {image_id -> set(label_id)}
    """
    label_images: Dict[int, Set[int]] = defaultdict(set)
    image_labels: Dict[int, Set[int]] = defaultdict(set)

    for prop in label_images_list:
        img2lab = prop.img2lab
        lab2img = prop.lab2img

        for img, labs in img2lab.items():
            image_labels.setdefault(img, set()).update(labs)
        for lab, imset in lab2img.items():
            label_images.setdefault(lab, set()).update(imset)
    return dict(label_images), dict(image_labels)


def create_properties_from_json_list(
    json_list: Iterable[str | Path],
    task_ids: Iterable[int]
) -> Tuple[
    List[TaskImages],          # list of TaskImages
    List[LabelImagesProperty]  # list of LabelImagesProperty
]:
    task_images_prop_list =[]
    label_images_prop_list = []

    for json_path, task_id in zip(json_list, task_ids):
        json_path = Path(json_path)
        imgs, img2lab, lab2img = load_single_task(json_path)

        task_images_prop_list.append(TaskImages(task_id=task_id, images=imgs))
        label_images_prop_list.append(LabelImagesProperty(img2lab=img2lab, lab2img=lab2img))
    return task_images_prop_list, label_images_prop_list


# json_list = [
#     "/home/kren04/shield/MD_DETR_runs/upload/mscoco/0/test_task_1.json",
#     "/home/kren04/shield/MD_DETR_runs/upload/mscoco/0/test_task_2.json",
#     "/home/kren04/shield/MD_DETR_runs/upload/mscoco/0/test_task_3.json",
#     "/home/kren04/shield/MD_DETR_runs/upload/mscoco/0/test_task_4.json"
# ]

# task_images_prop_list, label_images_prop_list = create_properties_from_json_list(
#     json_list=json_list,
#     task_ids=[1, 2, 3, 4]
# )

# t2i, i2t = build_global_task_images_maps(task_images_prop_list)
# l2i, i2l = build_global_label_images_maps(label_images_prop_list)


# def _membership_series(image_tasks: dict[int, set[int]]) -> pd.Series:
#     """Convert {image -> {tasks}} into UpsetPlot-compatible Series."""
#     memberships = [
#         tuple(sorted(str(t) for t in task_set)) for task_set in image_tasks.values()
#     ]
#     return from_memberships(memberships, data=[1]*len(memberships))

# def plot_upset(image_tasks: dict[int, set[int]], title: str|None=None):
#     s = _membership_series(image_tasks)
#     UpSet(s, 
#           subset_size="sum",
#           show_counts=True, sort_by='cardinality').plot()
#     if title: plt.suptitle(title)
#     plt.show()

# plot_upset(i2t, title="Image overlap across tasks")

def task_intersection_heatmap(task_images: dict[int, set[int]]):
    tasks = sorted(task_images)
    n = len(tasks)
    mat = np.zeros((n, n), int)
    for i, ti in enumerate(tasks):
        for j, tj in enumerate(tasks):
            mat[i, j] = len(task_images[ti] & task_images[tj])
    df = pd.DataFrame(mat, index=tasks, columns=tasks)
    plt.figure(figsize=(4+0.3*n, 4+0.3*n))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.title("Shared images between tasks")
    plt.xlabel("task"); plt.ylabel("task")
    plt.show()

# task_intersection_heatmap(t2i)







# TASK_RE = re.compile(r"test_task_(\d+)\.json$")

# def build_global_maps(
#     json_dir: str | Path,
# ) -> Tuple[
#     Dict[int, Set[int]],   # task_id -> images
#     Dict[int, Set[int]],   # image_id -> tasks
#     Dict[int, Set[int]],   # label_id -> images
#     Dict[int, Set[int]],   # image_id -> labels
# ]:
#     """
#     Iterate over every file named ``test_task_<digit>.json`` in *json_dir*
#     ( *not* the concatenated versions ) and build four dictionaries:

#       • task_images   : {task_id -> set(image_id)}
#       • image_tasks   : {image_id -> set(task_id)}
#       • label_images  : {label_id -> set(image_id)}
#       • image_labels  : {image_id -> set(label_id)}

#     Files whose <digit> part is not exactly **one** digit are ignored.
#     """
#     json_dir = Path(json_dir)
#     task_images: Dict[int, Set[int]]   = defaultdict(set)
#     image_tasks: Dict[int, Set[int]]   = defaultdict(set)
#     label_images: Dict[int, Set[int]]  = defaultdict(set)
#     image_labels: Dict[int, Set[int]]  = defaultdict(set)

#     for path in json_dir.glob("test_task_*.json"):
#         m = TASK_RE.fullmatch(path.name)
#         if not m:
#             continue                       # skip "test_task_123.json" etc.
#         task_id = int(m.group(1))          # single digit, e.g. 2

#         imgs, img2lab, lab2img = load_single_task(path)

#         # update global maps
#         task_images[task_id].update(imgs)
#         for img in imgs:
#             image_tasks[img].add(task_id)
#         for img, labs in img2lab.items():
#             image_labels.setdefault(img, set()).update(labs)
#         for lab, imset in lab2img.items():
#             label_images.setdefault(lab, set()).update(imset)

#     return (dict(task_images),
#             dict(image_tasks),
#             dict(label_images),
#             dict(image_labels))
