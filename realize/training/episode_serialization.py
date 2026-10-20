"""Converts tensors and arrays to JSON-serializable values."""
import torch
import numpy as np


def _to_serializable(x):
    """Recursively convert torch tensors / numpy arrays / nested lists to
    plain Python lists; other types pass through unchanged."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [_to_serializable(t) for t in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().tolist()
    return x
