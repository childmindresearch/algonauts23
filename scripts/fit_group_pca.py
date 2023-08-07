import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from hf_argparser import HfArg, HfArgumentParser
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from timm.utils import random_seed

from algonauts23 import ALGONAUTS_DATA_DIR
from algonauts23.visualize import Visualizer, plot_maps

plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 150

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)


@dataclass
class Args:
    out_dir: str = HfArg(aliases=["-o"], help="path to output directory")
    data_dir: str = HfArg(
        default=str(ALGONAUTS_DATA_DIR), help="path to algonauts root data directory"
    )
    n_components: int = HfArg(aliases=["-d"], default=1024, help="number of components")
    seed: int = HfArg(default=42, help="random seed")


def main(args: Args):
    start_time = time.monotonic()
    random_seed(args.seed)

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    logging.info("Loading data")
    train_data = load_activity(args, "train")
    logging.info("Train shape: %s", train_data.shape)
    val_data = load_activity(args, "val")
    logging.info("Val shape: %s", val_data.shape)

    logging.info("Fitting pca: n_components: %d", args.n_components)
    pca = PCA(n_components=args.n_components, svd_solver="randomized")
    pca.fit(train_data)

    val_pred = pca.inverse_transform(pca.transform(val_data))
    score = r2_score(val_data, val_pred)
    logging.info("Val r2: %.3f", score)

    logging.info("Generating components figure")
    # HACK: subj01 has complete data so it can work as a group visualizer
    visualizer = Visualizer(
        "subj01", root=Path(args.data_dir) / "algonauts_2023_challenge_data"
    )
    fig_path = out_dir / f"group_pca_d-{args.n_components}.png"
    plot_pca_basis(visualizer, pca, out_path=fig_path, n_plot_components=32, nrow=4)

    logging.info("Saving pca weights")
    state_path = out_dir / f"group_pca_d-{args.n_components}.pt"
    # weight: (n_features, n_components)
    # bias: (n_features,)
    state = {
        "weight": torch.as_tensor(pca.components_.T, dtype=torch.float32).contiguous(),
        "bias": torch.as_tensor(pca.mean_, dtype=torch.float32),
    }
    torch.save(state, state_path)

    logging.info("Done! Run time: %.0fs", time.monotonic() - start_time)


def load_activity(args: Args, split: str) -> np.ndarray:
    ds = load_from_disk(Path(args.data_dir) / "processed" / "size-native" / split)
    ds = ds.select_columns(["activity"])
    ds.set_format("numpy")
    data = ds[:]["activity"]
    return data


def plot_pca_basis(
    visualizer: Visualizer,
    pca: PCA,
    out_path: Path,
    n_plot_components: int = 8,
    nrow: int = 1,
):
    basis = []
    titles = []
    for ii in range(n_plot_components):
        u = pca.components_[ii]
        u = u / np.max(np.abs(u))
        basis.append(u)

        varp = pca.explained_variance_ratio_[ii]
        title = f"$u_{{{ii}}}$ (var % = {100 * varp:.1f})"
        titles.append(title)

    plot_maps(
        visualizer=visualizer,
        maps=basis,
        titles=titles,
        nrow=nrow,
        cmap="turbo",
        vmin=-1.0,
        vmax=1.0,
        out_path=out_path,
    )


if __name__ == "__main__":
    args: Args
    parser = HfArgumentParser(Args)
    (args,) = parser.parse_args_into_dataclasses()
    main(args)
