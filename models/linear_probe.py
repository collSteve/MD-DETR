"""Linear probe upper-bound diagnostic for MD-DETR.

Two classes:
  - FeatureStore:   accumulates raw per-proposal features + labels during extraction.
                    Unlike PrototypeStore (which accumulates class-wise running means),
                    this keeps the raw (feature, label) pairs so a linear classifier
                    can be trained on them offline.
  - LinearProbeScorer: loads a trained linear probe and provides a .score(features)
                       API matching PrototypeStore.score(). Used at eval time to
                       replace class_embed logits with trained-linear-classifier logits.
"""

import os
import torch


class FeatureStore:
    """Accumulates raw per-proposal features + labels per class.

    Save format: torch.save dict with keys {'features', 'labels', 'n_classes',
    'd_model', 'max_per_class', 'count'}, so future refactors of this class
    don't invalidate saved features.
    """

    def __init__(self, n_classes: int, d_model: int = 256, max_per_class: int = 2000):
        self.n_classes = n_classes
        self.d_model = d_model
        # max_per_class caps storage per class so the file stays tractable.
        # 80 classes × 2000 × 256 × 4 bytes ≈ 160 MB on disk. Plenty for a
        # linear probe to converge; overfitting isn't a risk at this scale.
        self.max_per_class = int(max_per_class)
        # Per-class lists of tensors; O(1) cap-check via len().
        self._features = [[] for _ in range(n_classes)]
        self._labels = [[] for _ in range(n_classes)]
        # Finalized tensors, set by finalize()
        self.features = None  # (N, d_model)
        self.labels = None    # (N,)
        self.count = None     # (n_classes,) int — per-class sample counts

    def add(self, features: torch.Tensor, class_ids: torch.Tensor):
        """Append features keyed by class id.

        features: (M, d_model) float
        class_ids: (M,) int in [0, n_classes)

        If max_per_class is set, entries for classes already at the cap are
        silently dropped (matches PrototypeStore semantics).
        """
        class_ids = class_ids.long()
        features = features.detach().cpu().float()
        for i in range(class_ids.numel()):
            c = int(class_ids[i].item())
            if self.max_per_class > 0 and len(self._features[c]) >= self.max_per_class:
                continue
            self._features[c].append(features[i])
            self._labels[c].append(c)

    def is_full(self) -> bool:
        """True iff every class has reached max_per_class. Always False when unlimited."""
        if self.max_per_class <= 0:
            return False
        return all(len(lst) >= self.max_per_class for lst in self._features)

    def finalize(self):
        """Flatten per-class lists into single (N, d_model) / (N,) tensors."""
        all_feats = []
        all_labs = []
        count = torch.zeros(self.n_classes, dtype=torch.int64)
        for c in range(self.n_classes):
            count[c] = len(self._features[c])
            if len(self._features[c]) > 0:
                all_feats.append(torch.stack(self._features[c], dim=0))
                all_labs.append(torch.full((len(self._features[c]),), c, dtype=torch.int64))
        if all_feats:
            self.features = torch.cat(all_feats, dim=0)
            self.labels = torch.cat(all_labs, dim=0)
        else:
            self.features = torch.empty(0, self.d_model)
            self.labels = torch.empty(0, dtype=torch.int64)
        self.count = count

    def save(self, path: str):
        if self.features is None:
            self.finalize()
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'features': self.features,
            'labels': self.labels,
            'count': self.count,
            'n_classes': self.n_classes,
            'd_model': self.d_model,
            'max_per_class': self.max_per_class,
        }, path)


class LinearProbeScorer:
    """Score features with a trained linear classifier. Mirrors PrototypeStore.score() API.

    The probe is an nn.Linear(d_model, n_classes). We store weights + biases
    in a plain dict (not pickled module) so the file is forward-compatible.
    """

    def __init__(self):
        self.W = None           # (n_classes, d_model)
        self.b = None           # (n_classes,)
        self.seen = None        # (n_classes,) bool — True for classes seen during training
        self.n_classes = None
        self.d_model = None
        self.meta = {}          # training hyperparameters + accuracy metrics

    @classmethod
    def load(cls, path: str, device=None) -> 'LinearProbeScorer':
        data = torch.load(path, map_location='cpu', weights_only=False)
        obj = cls()
        obj.W = data['W']
        obj.b = data['b']
        obj.seen = data['seen']
        obj.n_classes = int(data['n_classes'])
        obj.d_model = int(data['d_model'])
        obj.meta = data.get('meta', {})
        if device is not None:
            obj.W = obj.W.to(device)
            obj.b = obj.b.to(device)
            obj.seen = obj.seen.to(device)
        return obj

    def score(self, features: torch.Tensor, temperature: float = 10.0,
              unseen_value: float = -10e10) -> torch.Tensor:
        """Score features via trained linear probe: logits = features @ W.T + b.

        features: (..., d_model) — e.g. (B, 300, 256) decoder outputs.
        Returns: (..., n_classes) logits. Unseen classes get `unseen_value`.

        Note: `temperature` is accepted for API compatibility with PrototypeStore
        but IGNORED here — the probe was trained with CE on raw (unnormalized)
        features, so its logits are already at the correct scale. Applying
        temperature would change the distribution seen by downstream sigmoid/
        post_process and invalidate the upper-bound semantics.
        """
        if self.W is None:
            raise RuntimeError('LinearProbeScorer.score() called before load().')
        device = features.device
        W = self.W.to(device)
        b = self.b.to(device)
        logits = features @ W.T + b  # (..., n_classes)
        seen = self.seen.to(device)
        logits = logits.masked_fill(~seen, unseen_value)
        return logits
