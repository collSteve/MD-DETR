
import pickle
import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path to allow importing from models.probes
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.probes.query_probe import QueryRecord

def inspect_data(file_path: str, num_records: int):
    """
    Loads a pickled data file containing QueryRecords and prints a summary
    of the first few records.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        print("Please ensure the experiment name and path are correct.")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"Successfully loaded {len(data)} records from {file_path}")
        print(f"--- Inspecting first {min(num_records, len(data))} records ---")

        for i, record in enumerate(data[:num_records]):
            print(f"\n--- Record {i+1} ---")
            print(f"  Record Type: {type(record)}")
            print(f"  Image ID: {record.image_id}")
            print(f"  Task ID (during collection): {record.task_id}")
            print(f"  GT Class IDs in Image: {record.gt_class_ids}")
            print(f"  Object Queries Tensor Shape: {record.object_queries.shape}")
            print(f"  Object Queries Tensor DType: {record.object_queries.dtype}")
            # Print a small slice of the tensor to see the values
            print(f"  Object Queries Tensor Sample (first query, first 5 values):\n{record.object_queries[0, :5]}")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the contents of a query data pickle file.")
    parser.add_argument(
        '--exp_dir',
        type=str,
        required=True,
        help="Path to the root directory of the experiment (e.g., 'constancy_check_queries_v3')."
    )
    parser.add_argument(
        '--task_id',
        type=int,
        default=2,
        help="The task ID of the probe file to inspect."
    )
    parser.add_argument(
        '--tag',
        type=str,
        default='cur',
        help="The validation tag ('cur', 'prev', 'all') of the probe file to inspect."
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=0,
        help="The epoch number of the probe file to inspect."
    )
    parser.add_argument(
        '--rank',
        type=int,
        default=0,
        help="The rank of the probe file to inspect."
    )
    parser.add_argument(
        '--num',
        type=int,
        default=2,
        help="Number of records to inspect."
    )

    args = parser.parse_args()

    # Construct the full path to the pickle file based on the project's conventions
    file_path = os.path.join(
        args.exp_dir,
        f"Task_{args.task_id}",
        "query_probe",
        f"query_traces_task{args.task_id}",
        f"query_data_epoch{args.epoch:03d}_tag_{args.tag}_rank{args.rank}.pkl"
    )
    
    inspect_data(file_path, args.num)
