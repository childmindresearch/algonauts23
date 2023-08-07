import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from datasets import Dataset, Image
from PIL.Image import Resampling
from tqdm import tqdm

from algonauts23 import SUBS

ROOT = Path(__file__).parent
SEED = 2023
PARENT_SPLITS = {
    "train": "training",
    "val": "training",
    "testval": "training",
    "test": "test",
}

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)


def generate_dataset(root: Path, split: str = "train"):
    algonauts_dir = root / "algonauts_2023_challenge_data"
    parent_split = PARENT_SPLITS[split]

    masks = {sub: load_mask(algonauts_dir, sub) for sub in SUBS}
    group_mask = np.any(list(masks.values()), axis=0)

    for sub_id, sub in enumerate(SUBS):
        logging.info("Generating data for subject %s", sub)

        image_list = load_image_list(algonauts_dir, sub, parent_split)
        logging.info("Num images: %d", len(image_list))

        if parent_split != "test":
            logging.info("Loading fmri activity")
            fmri = load_fmri(algonauts_dir, sub, parent_split, masks[sub], group_mask)
            logging.info("fMRI shape: %s", fmri.shape)
        else:
            fmri = None

        indices = load_indices(root, sub, split)
        if indices is None:
            indices = np.arange(len(image_list))
        logging.info(
            "Split %s num samples: %d\n\tindices: %s", split, len(indices), indices[:10]
        )

        for idx in tqdm(indices):
            path = Path(image_list[idx])
            entities = dict(
                [part.split("-", maxsplit=2) for part in path.stem.split("_")]
            )
            nsd_id = int(entities["nsd"])

            activity = fmri[idx] if fmri is not None else None

            record = {
                "split": split,
                "subject": sub,
                "subject_id": sub_id,
                "sample_id": idx,
                "nsd_id": nsd_id,
                "image": str(path.absolute()),
                "activity": activity,
            }
            yield record


def load_mask(algonauts_dir: Path, sub: str) -> np.ndarray:
    roi_dir = algonauts_dir / sub / "roi_masks"
    mask_lh = np.load(roi_dir / "lh.all-vertices_fsaverage_space.npy")
    mask_rh = np.load(roi_dir / "rh.all-vertices_fsaverage_space.npy")
    mask = np.concatenate([mask_lh, mask_rh]) > 0
    return mask


def load_image_list(algonauts_dir: Path, sub: str, parent_split: str) -> List[Path]:
    image_dir = algonauts_dir / sub / f"{parent_split}_split" / f"{parent_split}_images"
    image_list = sorted(image_dir.glob("*.png"))
    return image_list


def load_fmri(
    algonauts_dir: Path,
    sub: str,
    parent_split: str,
    sub_mask: np.ndarray,
    group_mask: np.ndarray,
) -> np.ndarray:
    fmri_dir = algonauts_dir / sub / f"{parent_split}_split" / f"{parent_split}_fmri"
    fmri_lh = np.load(fmri_dir / f"lh_{parent_split}_fmri.npy")
    fmri_rh = np.load(fmri_dir / f"rh_{parent_split}_fmri.npy")
    fmri = np.concatenate([fmri_lh, fmri_rh], axis=1)
    fmri = sub_to_group(fmri, sub_mask, group_mask)
    return fmri


def load_indices(root: Path, sub: str, split: str) -> Optional[np.ndarray]:
    indices_path = root / "derived_splits" / sub / f"{split}_indices.npy"
    if not indices_path.exists():
        return None
    indices = np.load(indices_path)
    return indices


def sub_to_group(
    data: np.ndarray,
    sub_mask: np.ndarray,
    group_mask: np.ndarray,
) -> np.ndarray:

    # Most subjects have complete data, so sub mask is same as group mask
    if np.all(sub_mask == group_mask):
        return data

    projected = np.zeros(data.shape[:-1] + (group_mask.sum(),), dtype=data.dtype)
    mask = sub_mask[group_mask]
    projected[..., mask] = data
    return projected


def get_transforms(img_size: int):
    def transforms(examples):
        examples["image"] = [
            image.resize((img_size, img_size), resample=Resampling.BICUBIC)
            for image in examples["image"]
        ]
        return examples

    return transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        metavar="SPLIT",
        type=str,
        choices=["train", "val", "testval", "test"],
        default="train",
        help="dataset split",
    )
    parser.add_argument(
        "--img_size",
        "--sz",
        metavar="H",
        type=int,
        default=256,
        help="optional image size to resize to",
    )
    parser.add_argument(
        "--workers",
        "-j",
        metavar="N",
        type=int,
        default=4,
        help="number of data workers",
    )
    args = parser.parse_args()

    dset = Dataset.from_generator(
        generate_dataset,
        gen_kwargs={"root": ROOT, "split": args.split},
    )

    dset = dset.cast_column("image", Image())
    if args.img_size > 0:
        dset = dset.map(
            get_transforms(args.img_size),
            batched=True,
            batch_size=256,
            num_proc=args.workers,
        )

    if args.split == "train":
        dset = dset.shuffle(seed=SEED)

    out_dir = (
        ROOT
        / "processed"
        / f"size-{args.img_size if args.img_size > 0 else 'native'}"
        / args.split
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    dset.save_to_disk(out_dir, num_proc=args.workers)
