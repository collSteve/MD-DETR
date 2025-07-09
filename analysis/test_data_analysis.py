# analysis/task_image_maps.py
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple

TASK_RE = re.compile(r"test_task_([0-9]+)\.json")

def load_json(path: str | Path) -> Dict:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)

def _load_image_ids(path: Path) -> Set[int]:
    """Return the set of image IDs in one COCO-style JSON file."""
    with path.open("r") as f:
        data = json.load(f)
    return {img["id"] for img in data["images"]}

def build_task_image_maps(json_dir: str | Path
                          ) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """
    Scan *json_dir* for files named like ``test_task_1.json`` or
    ``test_task_123.json`` and build

        1. task_images : {task_id ➜ {image_id,…}}
        2. image_tasks : {image_id ➜ {task_id,…}}

    Returns
    -------
    (task_images, image_tasks)
    """
    json_dir = Path(json_dir)
    task_images: Dict[int, Set[int]] = defaultdict(set)
    image_tasks: Dict[int, Set[int]] = defaultdict(set)

    for path in json_dir.glob("test_task_*.json"):
        m = TASK_RE.fullmatch(path.name)
        if not m:
            continue                                      # skip weird names
        # A name like "test_task_123.json" means “tasks 1,2,3”
        task_chars = m.group(1)
        tasks = {int(ch) for ch in task_chars.lstrip("0")} or {0}

        image_ids = _load_image_ids(path)
        for t in tasks:
            task_images[t].update(image_ids)
        for img_id in image_ids:
            image_tasks[img_id].update(tasks)

    return task_images, image_tasks


# ---------------------------------------------------------------------------
# Example usage (run only when called directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint, argparse
    parser = argparse.ArgumentParser(description="Build task–image maps")
    parser.add_argument("json_dir", help="Directory containing test_task_*.json")
    args = parser.parse_args()

    t2i, i2t = build_task_image_maps(args.json_dir)
    print("Tasks found:", sorted(t2i))
    print("Images counted:", len(i2t))
    # pretty-print a tiny sample
    print("\nTask ➜ #images (first 5 tasks)")
    pprint.pprint({k: len(v) for k, v in list(t2i.items())[:5]})
    print("\nImage ➜ tasks (first 5 images)")
    pprint.pprint({k: sorted(v) for k, v in list(i2t.items())[:5]})
