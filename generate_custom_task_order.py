import argparse
import os
import sys
from pathlib import Path
import json

# Add the project root to the Python path to allow importing from 'datasets'
# This makes the script runnable from anywhere in the project
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from datasets.create_coco_instance import create_task_json, task_info_coco

def generate_reordered_annotations(order, base_train_ann, base_test_ann, output_dir_root):
    """
    Generates COCO annotation files for a custom continual learning task order.

    Args:
        order (list[int]): The desired task order, e.g., [1, 2, 4, 3].
        base_train_ann (str): Path to the main COCO training annotation JSON.
        base_test_ann (str): Path to the main COCO validation annotation JSON.
        output_dir_root (str): The root directory where the new task annotations will be saved.
    """
    if not all(i in order for i in [1, 2, 3, 4]) or len(order) != 4:
        raise ValueError("Order must be a permutation of [1, 2, 3, 4]")

    order_str = "_".join(map(str, order))
    output_dir = os.path.join(output_dir_root, f"order_{order_str}")
    
    print(f"Generating annotations for task order: {order_str}")
    print(f"Output directory: {output_dir}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get the canonical task definitions (class names, offsets, etc.)
    canonical_task_map, _ = task_info_coco()

    # Generate the single-task annotation files based on the new order
    for new_task_id, original_task_id in enumerate(order, 1):
        print(f"  - Generating files for new Task {new_task_id} (from original Task {original_task_id})")
        
        cat_names, offset, _ = canonical_task_map[original_task_id]
        
        # Create training set for the new task
        create_task_json(
            root_json=base_train_ann,
            cat_names=cat_names,
            offset=offset,
            set_type='train',
            output_dir=output_dir,
            task_id=str(new_task_id)
        )
        
        # Create test set for the new task
        create_task_json(
            root_json=base_test_ann,
            cat_names=cat_names,
            offset=offset,
            set_type='test',
            output_dir=output_dir,
            task_id=str(new_task_id)
        )

    # Generate the combined test sets for evaluation (e.g., test_task_12.json, test_task_123.json)
    print("\nGenerating combined test sets for evaluation...")
    for i in range(2, 5):
        combined_task_ids_str = "".join(map(str, range(1, i + 1)))
        print(f"  - Generating combined test set for tasks: {combined_task_ids_str}")

        all_cat_names = []
        for task_num in range(i):
            original_task_id = order[task_num]
            all_cat_names.extend(canonical_task_map[original_task_id][0])
        
        create_task_json(
            root_json=base_test_ann,
            cat_names=all_cat_names,
            offset=0,  # For combined evaluation, offset is always 0
            set_type='test',
            output_dir=output_dir,
            task_id=combined_task_ids_str
        )

    print("\nAnnotation generation complete.")
    print(f"You can now run an experiment with the argument: --task_ann_dir {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate COCO annotation files for a custom continual learning task order.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--order', 
        nargs='+', 
        type=int, 
        required=True,
        help="The desired task order. Example: --order 1 2 4 3"
    )
    parser.add_argument(
        '--train_ann', 
        type=str, 
        default="/home/kren04/shield/MD-DETR/data/mscoco/0/instances_train2017.json",
        help="Path to the main training annotation JSON file."
    )
    parser.add_argument(
        '--test_ann', 
        type=str, 
        default="/home/kren04/shield/MD-DETR/data/mscoco/0/instances_val2017.json",
        help="Path to the main validation annotation JSON file."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="/home/kren04/shield/MD-DETR/data/mscoco_reordered/",
        help="Root directory to save the generated annotation folders."
    )
    
    args = parser.parse_args()

    # It's safer to use absolute paths
    if not os.path.isabs(args.train_ann):
        args.train_ann = os.path.join(project_root, args.train_ann)
    if not os.path.isabs(args.test_ann):
        args.test_ann = os.path.join(project_root, args.test_ann)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    generate_reordered_annotations(
        order=args.order,
        base_train_ann=args.train_ann,
        base_test_ann=args.test_ann,
        output_dir_root=args.output_dir
    )
