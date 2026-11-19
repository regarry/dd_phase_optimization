"""Optional matched scene streams; absent keys preserve historical RNG behavior."""
from contextlib import contextmanager
import random

import numpy as np
import torch


def seed_all(seed):
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:
        raise ValueError('Scene seeds must be nonnegative integers')
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_seed(seed):
    if seed is None:
        yield
        return
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    seed_all(seed)
    try:yield
    finally:
        random.setstate(before[0]);np.random.set_state(before[1]);torch.set_rng_state(before[2])
        if before[3] is not None:torch.cuda.set_rng_state_all(before[3])
