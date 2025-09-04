
import pickle
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
import torch
from tqdm import tqdm
from models.probes.query_probe import QueryRecord

RECORDS_BY_IMAGE_GLOBAL = {}

def verify_constancy(exp_dir: str):
    """
    Loads all query data from a multi-task validation run and verifies that
    the queries for any given image_id are identical across different tasks.
    """
    global RECORDS_BY_IMAGE_GLOBAL

    print(f"[DEBUG] Starting verification for experiment: {exp_dir}")
    probe_data_root = os.path.join(exp_dir)
    if not os.path.exists(probe_data_root):
        print(f"ERROR: Experiment directory not found at {probe_data_root}")
        return

    # 1. Find and load all query data files
    all_records = []
    print(f"[DEBUG] Starting to search for .pkl files in {probe_data_root}...")
    
    # First, collect all file paths to create a progress bar
    pkl_files = []
    # Add a counter to show that os.walk is working
    walk_count = 0
    for root, dirs, files in os.walk(probe_data_root):
        walk_count += 1
        if walk_count % 100 == 0:
            print(f"[DEBUG] Scanned {walk_count} directories...", end='\r')
        for file in files:
            if file.startswith("query_data_") and file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    
    print(f"\n[DEBUG] Found {len(pkl_files)} files. Starting to load them.")

    # Now, iterate with a progress bar
    for file_path in tqdm(pkl_files, desc="Loading Data Files"):
        try:
            with open(file_path, 'rb') as f:
                records = pickle.load(f)
                all_records.extend(records)
        except Exception as e:
            print(f"  - Could not load or process file {file_path}. Error: {e}")
    
    if not all_records:
        print("\nERROR: No query records were successfully loaded.")
        return

    print(f"\n[DEBUG] Loaded a total of {len(all_records)} records. Grouping by image_id...")

    # 2. Group records by image_id
    records_by_image = defaultdict(list)
    for record in all_records:
        records_by_image[record.image_id].append(record)

    RECORDS_BY_IMAGE_GLOBAL = dict(records_by_image)
    print(f"[DEBUG] Grouped records into {len(records_by_image)} unique images.")

    # 3. Compare queries for images seen in multiple contexts
    differing_images = 0
    identical_images = 0
    images_checked = 0

    print(f"--- Comparing queries for {len(records_by_image)} unique images... ---")
    for image_id, records in tqdm(records_by_image.items(), desc="Verifying Image Queries"):
        if len(records) > 1:
            images_checked += 1
            first_query_tensor = records[0].object_queries
            is_constant = True
            for i in range(1, len(records)):
                if not torch.allclose(first_query_tensor, records[i].object_queries, atol=1e-4):
                    is_constant = False
                    break
            
            if is_constant:
                identical_images += 1
            else:
                differing_images += 1
                print(f"  - FAILED: Queries for image_id {image_id} are NOT constant across tasks.")


    # 4. Report final results
    print("\n--- Verification Complete ---")
    if images_checked == 0:
        print("RESULT: No images were found in more than one validation context.")
        print("Please ensure you ran validation on at least two tasks with overlapping datasets.")
        return

    print(f"Checked {images_checked} images that appeared in multiple validation runs.")
    print(f"  - Images with IDENTICAL queries: {identical_images}")
    print(f"  - Images with DIFFERING queries: {differing_images}")

    if differing_images == 0:
        print("\nHYPOTHESIS CONFIRMED: The object queries are deterministic for a given image.")
    else:
        print("\nHYPOTHESIS REJECTED: The object queries change based on the validation context.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Verify that object queries are constant for the same image across different task validation runs."
    )
    parser.add_argument(
        '--exp_dir', 
        type=str, 
        required=True,
        help="Path to the root directory of the experiment (e.g., 'constancy_check_queries_v3')."
    )
    
    args = parser.parse_args()
    verify_constancy(args.exp_dir)
