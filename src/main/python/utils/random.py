"""
Seeding utilities for deterministic behavior across libraries.

This module provides a single entry point `set_global_seed(seed)` that
initializes Python's `random`, NumPy, and PyTorch (if installed) PRNGs.

Usage:
    from src.main.python.utils.random import set_global_seed
    set_global_seed(42)
"""

from typing import Dict
import os
import random as _random

import numpy as _np


def _try_seed_torch(seed: int) -> bool:
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Make cuDNN deterministic if present (may affect performance)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        return True
    except Exception:
        return False


def set_global_seed(seed: int) -> Dict[str, bool]:
    """
    Seed Python, NumPy, and Torch (if available) for reproducibility.

    Also sets PYTHONHASHSEED to stabilize hash-based iteration ordering.

    Parameters
    ----------
    seed : int
        Non-negative integer seed.

    Returns
    -------
    dict
        Mapping of component -> success boolean.
    """
    seed = int(seed)
    assert seed >= 0, "Seed must be a non-negative integer"

    os.environ["PYTHONHASHSEED"] = str(seed)

    _random.seed(seed)
    np_ok = False
    try:
        _np.random.seed(seed)
        np_ok = True
    except Exception:
        np_ok = False

    torch_ok = _try_seed_torch(seed)

    return {
        "python": True,
        "numpy": np_ok,
        "torch": torch_ok,
    }
