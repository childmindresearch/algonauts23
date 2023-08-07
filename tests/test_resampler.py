from typing import Optional, Tuple

import numpy as np
import pytest

from algonauts23.resample import Bbox, Resampler


@pytest.fixture
def points() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random(size=(100, 2))


@pytest.fixture
def label() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 10, size=(100,))


@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
@pytest.mark.parametrize(
    "rect,shape", [(None, (107, 103)), (Bbox(0.0, 1.0, 0.0, 1.0), (100, 100))]
)
def test_resampler(
    points: np.ndarray,
    label: np.ndarray,
    interpolation: str,
    rect: Optional[Bbox],
    shape: Tuple[int, int],
):
    resampler = Resampler(0.01, rect=rect).fit(points)
    img = resampler.transform(label, categorical=True)
    assert img.shape == resampler.grid_shape_ == shape

    img = resampler.apply_mask(img, fill_value=0)
    label2 = resampler.inverse(img, categorical=True, interpolation=interpolation)
    assert np.all(label == label2)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
