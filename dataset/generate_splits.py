"""
Generate fixed splits derived from the official training set.
"""

import logging
import shutil
from pathlib import Path
from typing import List

import numpy as np

from algonauts23 import SUBS
from algonauts23.utils import seed_hash

ROOT = Path(__file__).parent
PARENT_SPLIT = "training-official"
DERIVED_SPLITS = {"train": 0.85, "val": 0.1, "testval": 0.05}
SEED = 2023

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)


def generate_splits(
    num_samples: int, split_sizes: List[float], seed: int
) -> List[np.ndarray]:
    """
    Generate reproducible data splits.

    Args:
        num_samples: number of samples
        split_sizes: fractional split sizes summing to one
        seed: random seed

    Returns:
        A list of split indices arrays
    """
    assert sum(split_sizes) == 1.0, "split_sizes must sum to 1"

    split_lengths = np.asarray(split_sizes) * num_samples
    split_ends = np.round(np.cumsum(split_lengths)).astype(int)
    split_starts = np.concatenate([[0], split_ends[:-1]])

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)

    splits = [
        np.sort(indices[start:end]) for start, end in zip(split_starts, split_ends)
    ]
    return splits


if __name__ == "__main__":
    challenge_data_dir = ROOT / "algonauts_2023_challenge_data"
    out_dir = ROOT / "derived_splits"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    for sub in SUBS:
        sub_dir = challenge_data_dir / sub
        sub_out_dir = out_dir / sub
        sub_out_dir.mkdir()

        train_fmri = np.load(
            sub_dir / "training_split" / "training_fmri" / "lh_training_fmri.npy",
            mmap_mode="r",
        )
        num_samples = len(train_fmri)

        seed = seed_hash("generate_splits", PARENT_SPLIT, sub, SEED)
        sizes = list(DERIVED_SPLITS.values())
        indices = generate_splits(num_samples, sizes, seed)

        for split, ind in zip(DERIVED_SPLITS, indices):
            logging.info("%s/%s: %s", sub, split, ind[:10])
            np.save(sub_out_dir / f"{split}_indices.npy", ind)
