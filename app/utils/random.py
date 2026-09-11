"""
Randomness System for TheUnnecessaryFM
Provides reproducible seeded pseudo-random number generation with weighted choice helpers.
"""

import secrets
from typing import Any, List, Optional, Sequence
import numpy as np


def generate_seed() -> int:
    """Generate a cryptographically secure 64-bit random seed."""
    return secrets.randbits(64)


class SeededRNG:
    """
    Wraps numpy.random.Generator with convenient musical and weighted randomness methods,
    while recording all seed state for 100% deterministic reproducibility.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            self.seed = generate_seed()
        else:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Return a random float in [low, high)."""
        return float(self.rng.uniform(low, high))

    def randint(self, low: int, high: int) -> int:
        """Return a random integer in [low, high]. Note: inclusive of high."""
        return int(self.rng.integers(low, high + 1))

    def choice(self, items: Sequence[Any], weights: Optional[Sequence[float]] = None) -> Any:
        """Select a single item, optionally with probability weights."""
        if not items:
            raise ValueError("Cannot choose from an empty sequence")
        if weights is not None:
            w = np.array(weights, dtype=np.float64)
            s = np.sum(w)
            if s <= 0 or np.isnan(s):
                p = np.ones(len(items)) / len(items)
            else:
                p = w / s
            idx = self.rng.choice(len(items), p=p)
        else:
            idx = self.rng.choice(len(items))
        return items[idx]

    def sample(self, items: Sequence[Any], k: int) -> List[Any]:
        """Sample k unique items without replacement."""
        k = min(k, len(items))
        if k <= 0:
            return []
        indices = self.rng.choice(len(items), size=k, replace=False)
        return [items[i] for i in indices]

    def shuffle(self, items: List[Any]) -> List[Any]:
        """Return a shuffled copy of items."""
        copy_items = list(items)
        self.rng.shuffle(copy_items)
        return copy_items

    def chance(self, probability: float) -> bool:
        """Return True with probability in [0, 1]."""
        return float(self.rng.uniform(0.0, 1.0)) < probability

    def gaussian(self, mean: float = 0.0, std: float = 1.0) -> float:
        """Return a normally distributed float."""
        return float(self.rng.normal(mean, std))
