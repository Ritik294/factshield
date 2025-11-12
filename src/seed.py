import random, numpy as np, torch


def set_seed(seed=42):
random.seed(seed)
np.random.seed(seed)
try:
import torch
torch.manual_seed(seed)
except Exception:
pass