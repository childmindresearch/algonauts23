"""
Cortical surface utils.

Example::

    surf = load_fsaverage_flat()
    rois = cortex.get_roi_verts("fsaverage")
    roi_polys = {k: surf.roi_to_poly(indices=v) for k, v in rois.items()}
    patch = surf.extract_patch(rois["V1"])
"""

from dataclasses import dataclass
from typing import Optional

import cortex
import numpy as np
import shapely


@dataclass
class Surface:
    """
    A triangulated surface.
    """

    points: np.ndarray
    polys: np.ndarray

    def __post_init__(self):
        self.polys = self.polys.astype(np.int64)

        assert self.points.ndim == 2 and self.points.shape[1] in {
            2,
            3,
        }, "Invalid points; expected shape (num_points, {2, 3})"

        assert (
            self.polys.ndim == 2 and self.polys.shape[1] == 3
        ), "Invalid polys; expected shape (num_polys, 3)"

        assert self.polys.min() >= 0 and self.polys.max() < len(
            self.points
        ), "Invalid indices in polys"

    def merge(self, other: "Surface") -> "Surface":
        """
        Merge surface with another.
        """
        points = np.concatenate([self.points, other.points])
        polys = np.concatenate([self.polys, other.polys + len(self)])
        return Surface(points, polys)

    def roi_indices_to_mask(self, indices: np.ndarray) -> np.ndarray:
        """
        ROI indices to mask.
        """
        assert (
            indices.ndim == 1 and indices.min() >= 0 and indices.max() < len(self)
        ), "Invalid indices"

        mask = np.zeros(len(self), dtype=bool)
        mask[indices] = True
        return mask

    def roi_mask_to_indices(self, mask: np.ndarray) -> np.ndarray:
        """
        ROI mask to indices.
        """
        assert mask.shape == (len(self),), "Invalid mask"
        return mask.nonzero()[0]

    def roi_to_poly(
        self,
        *,
        indices: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        simplify_tolerance: Optional[float] = None,
    ) -> shapely.MultiPolygon:
        """
        Convert an ROI to a shapely multi polygon.

        Args:
            indices: ROI indices
            mask: ROI mask, if indices not given
            simplify_tolerance: shapely polygon simplification tolerance in
                surface coordinate units.
        """
        assert not (indices is None and mask is None), "indices or mask is required"

        if indices is not None:
            mask = self.roi_indices_to_mask(indices)

        assert mask.ndim == 1 and len(mask) == len(
            self
        ), "Invalid mask; expected shape (num_points,)"

        poly_mask = mask[self.polys]
        poly_mask = np.all(poly_mask, axis=1)
        mask_polys = self.polys[poly_mask]
        mask_poly_pts = self.points[mask_polys]

        geoms = shapely.polygons(mask_poly_pts)
        boundary = shapely.unary_union(geoms)
        if simplify_tolerance:
            boundary = boundary.simplify(simplify_tolerance)
        if not isinstance(boundary, shapely.MultiPolygon):
            boundary = shapely.MultiPolygon([boundary])
        return boundary

    def extract_patch(
        self,
        *,
        indices: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> "Surface":
        """
        Extract the surface patch for the ROI indices or mask.
        """
        assert not (indices is None and mask is None), "indices or mask is required"

        if indices is not None:
            mask = self.roi_indices_to_mask(indices)

        mask_points = self.points[mask]
        mask_indices = np.cumsum(mask) - 1
        poly_mask = mask[self.polys]
        poly_mask = np.all(poly_mask, axis=1)
        mask_polys = self.polys[poly_mask]
        mask_polys = mask_indices[mask_polys]
        return Surface(mask_points, mask_polys)

    def __len__(self) -> int:
        return len(self.points)


def load_fsaverage_flat(hemisphere: Optional[str] = None) -> Surface:
    """
    Load canonical fsaverage flat surface from pycortex.

    Args:
        hemisphere: "lh" or "rh". If None, load both and concatenate.
    """
    assert hemisphere in {None, "lh", "rh"}, "Invalid hemisphere"

    if hemisphere is None:
        surf_lh = load_fsaverage_flat(hemisphere="lh")
        surf_rh = load_fsaverage_flat(hemisphere="rh")
        surf = surf_lh.merge(surf_rh)
    else:
        points, polys = cortex.db.get_surf("fsaverage", "flat", hemisphere=hemisphere)

        # keep only x, y
        points = points[:, :2].copy()

        # Shift lh to (-inf, -2), rh to (2, inf)
        padding = 2.0
        if hemisphere == "lh":
            points[:, 0] = points[:, 0] - points[:, 0].max() - padding
        else:
            points[:, 0] = points[:, 0] - points[:, 0].min() + padding
        surf = Surface(points, polys)
    return surf
