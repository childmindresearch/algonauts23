"""
Resample scattered data to a fixed grid using Gaussian weighted averaging.

Example::

    points = np.random.rand(100, 2)
    label = np.random.randint(0, 10, 100)

    resampler = Resampler(pixel_size=0.1, rect=(0, 1, 0, 1)).fit(points)
    label_img = resampler.transform(label, categorical=True)
    label_inv = resampler.inverse(label_img)
"""

from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors


class Bbox(NamedTuple):
    """
    Bounding box with format (xmin, xmax, ymin, ymax).
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float


class Resampler:
    """
    Resample scattered data to a fixed grid using Gaussian weighted averaging.

    Args:
        pixel_size: size of desired pixels in original units.
        padding: bounding box padding in pixel units.
        rect: Bounding box `(left, right, bottom, top)`. Overrides `padding`.
        fwhm: Gaussian FWHM in pixel units.
    """

    def __init__(
        self,
        pixel_size: float,
        padding: float = 4.0,
        rect: Optional[Bbox] = None,
        fwhm: float = 1.0,
    ):
        self.pixel_size = pixel_size
        self.padding = padding
        self.rect = None if rect is None else Bbox(*rect)
        self.fwhm = fwhm
        # https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/
        self._sigma = (pixel_size * fwhm) / 2.35482004503

        self._reset_data()

    def _reset_data(self):
        self.points_ = None
        self.bbox_ = None
        self.grid_ = None
        self.x_ = None
        self.y_ = None
        self.weight_ = None
        self.density_ = None
        self.mask_ = None
        self.point_mask_ = None

    @property
    def grid_shape_(self) -> Optional[Tuple[int, int]]:
        """
        Shape of grid, (height, width).
        """
        return None if self.grid_ is None else self.grid_.shape[:2]

    @property
    def grid_size_(self) -> Optional[Tuple[int, int]]:
        """
        Size of grid, (width, height).
        """
        return None if self.grid_ is None else self.grid_.shape[1::-1]

    @property
    def flat_grid_(self) -> Optional[np.ndarray]:
        """
        Flattened grid, shape (height * width, 2).
        """
        return None if self.grid_ is None else self.grid_.reshape(-1, 2)

    def fit(self, points: np.ndarray) -> "Resampler":
        """
        Fit resampler to scattered points, shape (n_points, 2).
        """
        self._reset_data()
        self.points_ = points
        self.grid_, self.bbox_ = self.fit_grid(
            points, self.pixel_size, padding=self.padding, rect=self.rect
        )
        self.x_ = np.ascontiguousarray(self.grid_[0, :, 0])
        self.y_ = np.ascontiguousarray(self.grid_[:, 0, 1])

        # Sparse nearest neighbors graph
        nbrs = NearestNeighbors()
        nbrs.fit(points)
        radius = 3 * self._sigma
        weight = nbrs.radius_neighbors_graph(
            self.flat_grid_, radius=radius, mode="distance"
        )

        # Gaussian averaging weights
        weight.data = np.exp(-0.5 * weight.data**2 / self._sigma**2)
        density = np.asarray(weight.sum(axis=1))
        weight = weight.multiply(1 / np.where(density == 0, 1e-8, density))
        self.weight_ = weight.tocsr()
        self.density_ = density.reshape(self.grid_shape_)
        self.mask_ = (density > 0).reshape(self.grid_shape_)

        # Mask of points contained in bbox
        self.point_mask_ = (
            (points[:, 0] >= self.bbox_.xmin)
            & (points[:, 0] <= self.bbox_.xmax)
            & (points[:, 1] >= self.bbox_.ymin)
            & (points[:, 1] <= self.bbox_.ymax)
        )
        return self

    def transform(
        self,
        data: np.ndarray,
        categorical: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Transform an array of data, shape (n_points, d), into a sequence of
        images, shape (h, w, d).
        """
        self._check_fit()
        if categorical is None:
            categorical = np.issubdtype(data.dtype, np.integer) or np.issubdtype(
                data.dtype, np.bool_
            )
        assert data.ndim in (1, 2) and len(data) == len(
            self.points_
        ), "invalid data; expected shape (n_points, d) or (n_points,)"
        assert not categorical or (
            data.ndim == 1 or data.shape[1] == 1
        ), "multiple categorical frames not supported"

        # Apply the Gaussian weighted averaging using sparse matrix multiply.
        shape = data.shape
        if categorical:
            data, uniq = label_to_one_hot(data)
        image = self.weight_ @ (data[:, None] if data.ndim == 1 else data)
        if categorical:
            image = one_hot_to_label(image, uniq)
        image = image.reshape(self.grid_shape_ + shape[1:])
        return image

    def inverse(
        self,
        image: np.ndarray,
        categorical: Optional[bool] = None,
        interpolation: str = "nearest",
    ) -> np.ndarray:
        """
        Transform image data back onto scattered points using interpolation.
        """
        self._check_fit()
        if categorical is None:
            categorical = np.issubdtype(image.dtype, np.integer) or np.issubdtype(
                image.dtype, np.bool_
            )
        assert image.shape[:2] == self.grid_shape_, "image doesn't match grid"
        assert image.ndim in {
            2,
            3,
        }, "invalid image shape; expected (h, w), or (h, w, d)"
        assert interpolation in {
            "nearest",
            "linear",
        }, f"invalid interpolation {interpolation}; expected 'nearest' or 'linear'"

        if categorical:
            image, uniq = label_to_one_hot(image)
        # reversing x, y since interpn expects 'ij' ordering
        points = self.points_[:, [1, 0]]
        data = np.zeros((len(points),) + image.shape[2:], dtype=image.dtype)
        data[self.point_mask_] = interpolate.interpn(
            (self.y_, self.x_), image, points[self.point_mask_], method=interpolation
        )
        if categorical:
            data = one_hot_to_label(data, uniq).astype(image.dtype)
        return data

    def apply_mask(self, image: np.ndarray, fill_value: Any = np.nan) -> np.ndarray:
        """
        Apply the valid point mask to the image.
        """
        self._check_fit()
        assert image.shape[:2] == self.grid_shape_, "image doesn't match grid"
        assert image.ndim in {
            2,
            3,
        }, "invalid image shape; expected (h, w), or (h, w, d)"
        mask = self.mask_ if image.ndim == 2 else self.mask_[..., None]
        image = np.where(mask, image, fill_value)
        return image

    def _check_fit(self):
        assert self.points_ is not None, "resampler still needs to be fit"

    @staticmethod
    def fit_grid(
        points: np.ndarray,
        pixel_size: float,
        padding: float,
        rect: Optional[Bbox] = None,
    ) -> Tuple[np.ndarray, Bbox]:
        """
        Fit a pixel grid to scattered points with desired padding and pixel size.

        Args:
            points: array of `(x, y)` points, shape `(num_points, 2)`.
            pixel_size: pixel size in data units.
            padding: padding in pixel units.
            rect: Bounding box `(left, right, bottom, top)`. Overrides `padding`.

        Returns:
            grid: `(x, y)` grid array, shape `(height, width, 2)`.
            bbox: grid bounding box `(xmin, xmax, ymin, ymax)`.
        """
        if rect is None:
            xmin, ymin = points.min(axis=0)
            xmax, ymax = points.max(axis=0)

            padding = padding * pixel_size
            xmin = xmin - padding
            xmax = xmax + padding
            ymin = ymin - padding
            ymax = ymax + padding
        else:
            xmin, xmax, ymin, ymax = rect

        w = int(np.round((xmax - xmin) / pixel_size))
        h = int(np.round((ymax - ymin) / pixel_size))
        xmin = pixel_size * np.round(xmin / pixel_size)
        ymin = pixel_size * np.round(ymin / pixel_size)
        xmax = xmin + pixel_size * w
        ymax = ymin + pixel_size * h

        # TODO: having the origin of the grid be lower left is annoying for
        # saving as png.
        x = xmin + pixel_size * np.arange(0.5, w)
        y = ymin + pixel_size * np.arange(0.5, h)
        grid = np.meshgrid(x, y)
        grid = np.stack(grid, axis=-1)
        return grid, Bbox(xmin, xmax, ymin, ymax)


def label_to_one_hot(label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a categorical label to one-hot representation. Return the one-hot
    representation and the unique label values.
    """
    shape = label.shape
    label = label.flatten()
    uniq, label = np.unique(label, return_inverse=True)
    one_hot = np.zeros((len(label), len(uniq)))
    one_hot[np.arange(len(label)), label] = 1.0
    if len(shape) > 1:
        one_hot = one_hot.reshape(shape + (len(uniq),))
    return one_hot, uniq


def one_hot_to_label(
    one_hot: np.ndarray, uniq: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert a one-hot to categorical representation.
    """
    label = np.argmax(one_hot, axis=-1)
    if uniq is not None:
        label = uniq[label]
    return label
