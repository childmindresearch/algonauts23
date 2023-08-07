"""
Cortical parcellation utils.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve

import nibabel as nib
import numpy as np

PARC_CACHE_DIR = Path.home() / ".cache/parcellations"


@dataclass
class Parcellation:
    """
    An ROI parcellation consisting of an integer label array, a mapping of
    indices to string names, and an optional colormap.

    Example::

        parc = Parcellation(label, mapping)
        mask = parc.get("PPA")
    """

    label: np.ndarray
    mapping: Optional[Dict[int, str]] = None
    colormap: Optional[Dict[int, Tuple[int, int, int]]] = None

    def __post_init__(self):
        self.label = self.label.astype(int)

        if self.mapping is None:
            uniq = np.unique(self.label)
            self.mapping = {ii: str(ii) for ii in uniq.tolist()}

        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.sizes = {k: self.get(k).sum() for k in self.mapping}

    def get(self, roi: Optional[Union[int, str]] = None) -> np.ndarray:
        """
        Get the mask for the ROI index or string name.

        If `roi` is `None` or `"all"`, returns the mask of the entire group.
        """
        if roi in {None, "all"}:
            return self.label > 0
        elif isinstance(roi, int):
            return self.label == roi
        else:
            return self.label == self.reverse_mapping[roi]

    def names(self) -> List[str]:
        """
        List of ROI names.
        """
        return list(self.reverse_mapping.keys())

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self):
        return iter(self.reverse_mapping)

    @classmethod
    def from_masks(
        cls,
        masks: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    ) -> "Parcellation":
        """
        Create a parcellation from a group of masks. Each point is assigned to
        the smallest mask containing it.
        """
        if not isinstance(masks, dict):
            masks = {str(ii + 1): mask for ii, mask in enumerate(masks)}
        names = list(masks.keys())
        masks = np.asarray(list(masks.values()))

        # prepend unknown mask
        masks = np.concatenate([np.ones((1, masks.shape[1]), dtype="bool"), masks])

        # assign each dimension to the smallest region containing it
        mask_sizes = np.where(masks, masks.sum(axis=1, keepdims=True), np.inf)
        label = np.argmin(mask_sizes, axis=0)

        mapping = {ii + 1: name for ii, name in enumerate(names)}
        return cls(label=label, mapping=mapping)

    @classmethod
    def from_fs_annot(
        cls,
        lh_path: Union[str, Path],
        rh_path: Union[str, Path],
    ) -> "Parcellation":

        lh_label, lh_colors, lh_names = _load_fs_annot(lh_path)
        rh_label, rh_colors, rh_names = _load_fs_annot(rh_path)

        hemi_id_offset = len(lh_names)
        label = np.concatenate([lh_label, rh_label + hemi_id_offset])
        colors = lh_colors + rh_colors
        names = lh_names + rh_names

        mapping = {ii: name for ii, name in enumerate(names)}
        colormap = {ii: color for ii, color in enumerate(colors)}
        return cls(label=label, mapping=mapping, colormap=colormap)


def _load_fs_annot(path: Union[str, Path]):
    label, colors, names = nib.freesurfer.read_annot(path)

    # Check that label ids are sequential starting from 0
    ids = np.unique(label)
    assert np.all(ids == np.arange(len(ids))), "Unexpected annot label IDs"
    # Check that colors and names match labels
    assert (
        len(colors) == len(names) == len(ids)
    ), "Annot colors and/or names don't match label IDs"

    names = [_as_str(name) for name in names]
    colors = [tuple(color[:3]) for color in colors]
    return label, colors, names


def _as_str(s: Union[str, bytes]):
    if isinstance(s, bytes):
        s = s.decode()
    return str(s)


def load_fsaverage_schaefer(parcels: int = 200, networks: int = 7) -> Parcellation:
    """
    Load a Schaefer parcellation for fsaverage space.
    """
    assert parcels in {ii * 100 for ii in range(1, 11)}, f"Invalid parcels {parcels}"
    assert networks in {7, 17}, f"Invalid networks {networks}"

    annot_paths = {}
    for hemi in ["lh", "rh"]:
        filename, url = _get_schaefer_url(hemi=hemi, parcels=parcels, networks=networks)
        annot_paths[hemi] = _maybe_download(filename, url)
    return Parcellation.from_fs_annot(
        lh_path=annot_paths["lh"], rh_path=annot_paths["rh"]
    )


def _get_schaefer_url(
    hemi: str = "lh", parcels: int = 200, networks: int = 7
) -> Tuple[str, str]:

    name = f"{hemi}.Schaefer2018_{parcels}Parcels_{networks}Networks_order.annot"
    url = (
        "https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/"
        "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/"
        f"FreeSurfer5.3/fsaverage/label/{name}"
    )
    return name, url


def _maybe_download(
    name: str, url: str, cache_dir: Union[str, Path] = PARC_CACHE_DIR
) -> Path:
    path = Path(cache_dir) / name
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)
    return path
