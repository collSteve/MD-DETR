"""Prototype classifier for post-hoc continual-learning diagnostics.

Accumulates running mean decoder features per class from matched proposals,
and scores new features via temperature-scaled cosine similarity.
"""

import os
import torch
import torch.nn.functional as F


class PrototypeStore:
    """Running mean features per class, temperature-scaled cosine scoring.

    Save/load format: torch.save dict (not pickled class instance), so future
    refactors of this class don't invalidate saved prototypes.
    """

    def __init__(self, n_classes: int, d_model: int = 256):
        self.n_classes = n_classes
        self.d_model = d_model
        self.sum_ = torch.zeros(n_classes, d_model)
        self.count = torch.zeros(n_classes, dtype=torch.int64)
        self.prototypes = None  # (n_classes, d_model), set by finalize()
        self.seen = None        # (n_classes,) bool, set by finalize()

    def add(self, features: torch.Tensor, class_ids: torch.Tensor):
        """Accumulate features keyed by class id.

        features: (M, d_model) float
        class_ids: (M,) int in [0, n_classes)
        """
        class_ids = class_ids.long()
        self.sum_.index_add_(0, class_ids, features.float())
        self.count.index_add_(0, class_ids, torch.ones_like(class_ids, dtype=torch.int64))

    def finalize(self):
        self.seen = self.count > 0
        safe_count = self.count.clamp(min=1).unsqueeze(-1).float()
        self.prototypes = self.sum_ / safe_count

    def save(self, path: str):
        if self.prototypes is None:
            self.finalize()
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'prototypes': self.prototypes,
            'seen': self.seen,
            'count': self.count,
            'n_classes': self.n_classes,
            'd_model': self.d_model,
        }, path)

    @classmethod
    def load(cls, path: str, device=None) -> 'PrototypeStore':
        data = torch.load(path, map_location='cpu', weights_only=False)
        store = cls(int(data['n_classes']), int(data['d_model']))
        store.prototypes = data['prototypes']
        store.seen = data['seen']
        store.count = data['count']
        if device is not None:
            store.prototypes = store.prototypes.to(device)
            store.seen = store.seen.to(device)
        return store

    def score(self, features: torch.Tensor, temperature: float = 10.0,
              unseen_value: float = -10e10) -> torch.Tensor:
        """Score features by temperature-scaled cosine similarity to prototypes.

        features: (..., d_model) — e.g. (B, 300, 256) decoder outputs.
        Returns: (..., n_classes) logits. Unseen classes get `unseen_value`.
        """
        if self.prototypes is None:
            raise RuntimeError('PrototypeStore.score() called before finalize()/load().')
        device = features.device
        f = F.normalize(features, dim=-1)
        p = F.normalize(self.prototypes.to(device), dim=-1)
        logits = torch.matmul(f, p.T) * temperature  # (..., n_classes)
        seen = self.seen.to(device)
        logits = logits.masked_fill(~seen, unseen_value)
        return logits
