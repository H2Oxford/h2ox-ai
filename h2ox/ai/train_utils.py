import torch
import numpy as np


def get_exponential_weights(horizon: int) -> torch.Tensor:
    # exponential weighting
    wt = np.exp(np.linspace(0, 5, horizon))[::-1]
    wt = wt / np.linalg.norm(wt)
    wt = torch.from_numpy(wt)
    return wt
