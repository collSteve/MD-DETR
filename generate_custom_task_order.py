import argparse
import os
import sys
from pathlib import Path
import json

# Add the project root to the Python path to allow importing from 'datasets'
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from datasets.create_coco_instance import create_task_json, task_info_coco

def merge_coco_files(file_paths, output_path):
    """
    Merges multiple COCO annotation files into a single file, ensuring no duplicate
    images or categories while preserving all original category IDs.
    """
    merged_data = {
        'info': {}, 'licenses': [], 'images': [],
        'annotations': [], 'categories': []
    }
    seen_image_ids, seen_category_ids = set(), set()

    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as f:
            data = json.load(f)
        if i == 0:
            merged_data['info'] = data.get('info', {})
            merged_data['licenses'] = data.get('licenses', [])
        for image_info in data.get('images', []):
            if image_info['id'] not in seen_image_ids:
                merged_data['images'].append(image_info)
                seen_image_ids.add(image_info['id'])
        for category_info in data.get('categories', []):
            if category_info['id'] not in seen_category_ids:
                merged_data['categories'].append(category_info)
                seen_category_ids.add(category_info['id'])
        merged_data['annotations'].extend(data.get('annotations', []))

    with open(output_path, 'w') as f:
        json.dump(merged_data, f)

def generate_reordered_annotations(order, base_train_ann, base_test_ann, output_dir_root):
    """
    Generates COCO annotation files for a custom continual learning task order.
    This version correctly re-numbers class IDs to be contiguous and merges
    test files to preserve those new IDs.
    """
    if sorted(order) != [1, 2, 3, 4]:
        raise ValueError("Order must be a permutation of [1, 2, 3, 4]")

    order_str = "_".join(map(str, order))
    output_dir = os.path.join(output_dir_root, f"order_{order_str}")
    
    print(f"Generating annotations for task order: {order_str}")
    print(f"Output directory: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    canonical_task_map, _ = task_info_coco()
    
    # Part 1: Generate single-task files with CONTIGUOUS offsets (The First Fix)
    current_offset = 0
    single_task_test_files = []

    for new_task_id, original_task_id in enumerate(order, 1):
        class_names, _, class_count = canonical_task_map[original_task_id]
        print(f"  - Generating files for new Task {new_task_id} (from original Task {original_task_id}) with offset {current_offset}")
        
        create_task_json(
            root_json=base_train_ann, cat_names=class_names, offset=current_offset,
            set_type='train', output_dir=output_dir, task_id=str(new_task_id)
        )
        
        test_file_path = os.path.join(output_dir, f"test_task_{new_task_id}.json")
        create_task_json(
            root_json=base_test_ann, cat_names=class_names, offset=current_offset,
            set_type='test', output_dir=output_dir, task_id=str(new_task_id)
        )
        single_task_test_files.append(test_file_path)

        current_offset += class_count

    # Part 2: Generate combined test sets by merging (The Second Fix)
    print("\nGenerating combined test sets for evaluation by merging...")
    for i in range(2, 5):
        combined_task_ids_str = "".join(map(str, range(1, i + 1)))
        print(f"  - Generating combined test set for tasks: {combined_task_ids_str}")
        files_to_merge = single_task_test_files[:i]
        output_path = os.path.join(output_dir, f"test_task_{combined_task_ids_str}.json")
        merge_coco_files(files_to_merge, output_path)

    print("\nAnnotation generation complete.")
    print(f"You can now run an experiment with the argument: experiment.split_point={order_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate COCO annotation files for a custom continual learning task order.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--order', nargs='+', type=int, required=True,
        help="The desired task order. Example: --order 1 2 4 3"
    )
    parser.add_argument(
        '--train_ann', type=str, 
        default="/home/kren04/datasets/MSCOCO/2017/annotations/instances_train2017.json",
        help="Path to the main training annotation JSON file."
    )
    parser.add_argument(
        '--test_ann', type=str, 
        default="/home/kren04/datasets/MSCOCO/2017/annotations/instances_val2017.json",
        help="Path to the main validation annotation JSON file."
    )
    parser.add_argument(
        '--output_dir', type=str, 
        default="/home/kren04/shield/MD_DETR_runs/upload/mscoco_reordered/",
        help="Root directory to save the generated annotation folders."
    )
    args = parser.parse_args()
    generate_reordered_annotations(
        order=args.order,
        base_train_ann=args.train_ann,
        base_test_ann=args.test_ann,
        output_dir_root=args.output_dir
    )