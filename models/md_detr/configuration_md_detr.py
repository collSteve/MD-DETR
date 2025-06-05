from transformers.models.deformable_detr import DeformableDetrConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MDDetrConfig(DeformableDetrConfig):
    model_type = "md_detr" 

    def __init__(
        self,
        use_prompts: bool = False,
        n_tasks: int = 1,
        num_prompts: int = 4,
        prompt_len: int = 8,
        local_query: bool = False,
        PREV_INTRODUCED_CLS: int = 0,
        CUR_INTRODUCED_CLS: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.use_prompts = use_prompts
        self.n_tasks = n_tasks
        self.num_prompts = num_prompts
        self.prompt_len = prompt_len
        self.local_query = local_query
        self.PREV_INTRODUCED_CLS = PREV_INTRODUCED_CLS
        self.CUR_INTRODUCED_CLS = CUR_INTRODUCED_CLS

        # Basic validation
        if self.use_prompts and self.n_tasks < 1:
            raise ValueError("`n_tasks` must be ≥ 1 when `use_prompts=True`")

        logger.debug(
            "Created ContinualDetrConfig with %s prompts per task (prompt_len=%s) "
            "for %s tasks. Previous/Current class split = (%s, %s).",
            self.num_prompts,
            self.prompt_len,
            self.n_tasks,
            self.PREV_INTRODUCED_CLS,
            self.CUR_INTRODUCED_CLS,
        )


__all__ = ["ContinualDetrConfig"]