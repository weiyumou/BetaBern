"""Shared helpers for the test suite."""
import torch


def make_batch(B: int = 4, S: int = 3, num_items: int = 10, num_skills: int = 3, seed: int = 0):
    """A random ``(item_ids, skill_ids, answers, mask)`` response batch with no padding."""
    g = torch.Generator().manual_seed(seed)
    item_ids = torch.randint(0, num_items, (B, S), generator=g)
    skill_ids = torch.randint(0, num_skills, (B,), generator=g)
    answers = torch.randint(0, 2, (B, S), generator=g)
    mask = torch.ones(B, S, dtype=torch.bool)
    return item_ids, skill_ids, answers, mask
