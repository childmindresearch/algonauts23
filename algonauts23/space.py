from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from shapely import MultiPolygon
from sklearn.neighbors import NearestNeighbors

from algonauts23 import ALGONAUTS_RAW_DIR, ROI_GROUPS, SUBS
from algonauts23.parcellation import Parcellation, load_fsaverage_schaefer
from algonauts23.surface import load_fsaverage_flat


class AlgonautsSpace:
    """
    Algonauts fMRI surface space, consisting of a flat surface patch and a set
    of ROI parcellations.

    Example::

        space = AlgonautsSpace(sub="subj01")
        mask = space.get_roi(group="streams", roi="midventral")
        poly = space.get_roi_poly(group="streams", roi="midventral")
    """

    def __init__(self, sub: str, root: Union[str, Path] = ALGONAUTS_RAW_DIR):
        assert sub in SUBS, f"Invalid sub {sub}"

        self.sub = sub
        self.root = Path(root)

        surf = load_fsaverage_flat()
        mask, hemi_offset = load_challenge_mask(root, sub)
        groups = list(ROI_GROUPS.keys())
        parcs = {group: load_roi_group(root, sub, group) for group in groups}
        patch = surf.extract_patch(mask=mask)
        group_mask, _ = load_challenge_mask(root, "subj01")

        self.mask = mask
        self.hemi_offset = hemi_offset
        self.parcs = parcs
        self.patch = patch
        self.group_mask = group_mask

    def get_roi(
        self,
        group: str,
        roi: Optional[Union[int, str]] = None,
        hemi: Optional[str] = None,
    ) -> np.ndarray:
        """
        Get an individual ROI mask belonging to a group (possibly empty),
        optionally restricted to a hemisphere.

        If roi is None, returns the mask of the entire group.
        If hemi is None, return both hemis.
        """
        parc = self.parcs[group]
        mask = parc.get(roi)

        if hemi not in {None, "both"}:
            mask = self.mask_hemi(mask, hemi=hemi)
        return mask

    def mask_hemi(self, data: np.ndarray, hemi: str = "lh"):
        assert hemi in {"lh", "rh"}, f"Invalid hemi {hemi}"
        assert data.shape[-1] == len(self), "Data doesn't match space"

        data = data.copy()
        if hemi == "lh":
            data[..., self.hemi_offset :] = 0
        else:
            data[..., : self.hemi_offset] = 0
        return data

    def split_hemi(self, data: np.ndarray):
        assert data.shape[-1] == len(self), "Data doesn't match space"

        data_lh = data[..., : self.hemi_offset]
        data_rh = data[..., self.hemi_offset :]
        return data_lh, data_rh

    @lru_cache()
    def get_roi_poly(
        self,
        group: str,
        roi: Optional[Union[int, str]] = None,
        hemi: Optional[str] = None,
    ) -> Optional[MultiPolygon]:
        """
        Get an individual ROI as a polygon.
        """
        mask = self.get_roi(group, roi=roi, hemi=hemi)
        if mask.sum() == 0:
            return None
        return self.patch.roi_to_poly(mask=mask)

    def list_rois(
        self, group: Optional[str] = None
    ) -> Union[List[str], Dict[str, List[str]]]:
        """
        List of available ROIs for one group or all groups.
        """
        if group is not None:
            return self.parcs[group].names()
        return {group: self.list_rois(group) for group in self.parcs}

    def roi_groups(self) -> List[str]:
        """
        List of ROI groups.
        """
        return list(self.parcs.keys())

    def project(self, data: np.ndarray) -> np.ndarray:
        """
        Project data from fsaverage space or group challenge space to the subject's
        challenge space. The last dimension of data should match one of the two source
        spaces.
        """
        dim = data.shape[-1]
        if dim == len(self.mask):
            mask = self.mask
        elif dim == self.group_mask.sum():
            mask = self.mask[self.group_mask]
        else:
            raise ValueError(f"Data has invalid dimension {dim}")

        data = data[..., mask]
        return data

    def project_parc(
        self,
        parc: Parcellation,
        reindex: bool = True,
        max_regions: Optional[int] = None,
    ) -> np.ndarray:
        """
        Project a parcellation onto the challenge space, with the option to re-index
        labels and keep only the top k largest regions.
        """
        label = self.project(parc.label)

        uniq, counts = np.unique(label, return_counts=True)
        if max_regions and max_regions < len(uniq):
            # Re-assign small regions to the top-k largest using nearest neighbors
            topk = uniq[np.argsort(-counts)[:max_regions]]
            mask = np.isin(label, topk)
            nbrs = NearestNeighbors().fit(self.patch.points[mask])
            neigh_ind = nbrs.kneighbors(
                self.patch.points, n_neighbors=1, return_distance=False
            )
            label = label[mask][neigh_ind[:, 0]]

        if reindex:
            _, label = np.unique(label, return_inverse=True)
        return label

    def __len__(self) -> int:
        return len(self.patch)


def load_challenge_mask(root: Union[str, Path], sub: str) -> Tuple[np.ndarray, int]:
    """
    Load algonauts challenge mask for subject sub. Return mask array and hemisphere
    offset.
    """
    roi_dir = Path(root) / sub / "roi_masks"
    mask_lh = np.load(roi_dir / "lh.all-vertices_fsaverage_space.npy")
    mask_rh = np.load(roi_dir / "rh.all-vertices_fsaverage_space.npy")
    mask = np.concatenate([mask_lh, mask_rh]) > 0
    hemi_offset = mask_lh.sum()
    return mask, hemi_offset


def load_roi_group(root: Union[str, Path], sub: str, group: str) -> Parcellation:
    """
    Load algonauts ROI group for subject sub.
    """
    roi_dir = Path(root) / sub / "roi_masks"
    label_lh = np.load(roi_dir / f"lh.{group}_challenge_space.npy")
    label_rh = np.load(roi_dir / f"rh.{group}_challenge_space.npy")
    label = np.concatenate([label_lh, label_rh])

    mapping = np.load(roi_dir / f"mapping_{group}.npy", allow_pickle=True)
    mapping: dict = mapping.item()
    # Drop Unknown region from mapping
    mapping.pop(0)

    parc = Parcellation(label, mapping)
    return parc


def load_schaefer_patches(space: AlgonautsSpace, parcels: int = 800) -> np.ndarray:
    """
    Load Schaefer parcellation patches as a label array with ``parces // 8``
    patches.
    """
    parc = load_fsaverage_schaefer(parcels=parcels)
    # The challenge ROI is a bit more than 1/8 of full cortex. Restricting to
    # this many patches gives a nice round consistent number and helps to
    # consolidate the small cut off boundary patches.
    num_patches = parcels // 8
    label = space.project_parc(parc, reindex=True, k=num_patches)
    return label
