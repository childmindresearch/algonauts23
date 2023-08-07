import numpy as np
import pytest
from scipy.spatial import Delaunay

from algonauts23.surface import Surface


@pytest.fixture
def surf() -> Surface:
    rng = np.random.default_rng(42)
    points = rng.random(size=(400, 2))
    tri = Delaunay(points)
    polys = tri.simplices
    return Surface(points, polys)


def test_extract_patch(surf: Surface):
    mask = np.all(surf.points < 0.5, axis=1)
    patch = surf.extract_patch(mask=mask)
    assert np.all(patch.polys >= 0) and np.all(patch.polys < len(patch))


if __name__ == "__main__":
    pytest.main(["-s", __file__])
