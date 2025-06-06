import os
import pickle


def checkpoint_filename(epoch: int)-> str:
    return f"checkpoint{epoch:04d}.pth"

def get_epoch_from_filename(filename: str) -> int:

    epoch_str = filename[len("checkpoint"):-len(".pth")]
    try:
        return int(epoch_str)
    except ValueError:
        raise ValueError(f"Invalid epoch number in filename: {filename}")
    
def dir_path_for_task(base_dir_path, task_id: int) -> str:

    return os.path.join(base_dir_path, f"Task_{task_id:02d}")

def load_pickle(file_path: str):
    with open(file_path, 'rb') as file:
        return pickle.load(file)