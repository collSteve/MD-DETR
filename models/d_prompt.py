from .prompt import Prompt, PromptParam

class DynamicMemory(Prompt):
    def __init__(self, emb_d, n_tasks, prompt_param: PromptParam, key_dim=768, args=None):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim, args)
        