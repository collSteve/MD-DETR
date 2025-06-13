import torch

def load_memory_parameters(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_dict = ckpt["model"]

    memory_params = {k: v for k, v in model_dict.items() if k.startswith("model.prompts.")}

    return memory_params


def extract_key_space_params(memory_params):
    key_space_params = {} # subcat -> subcat, layer, id (number), wieghts
    for k, v in memory_params.items():
        if "k_list" in k:
            key_space_params[k] = v
    return key_space_params