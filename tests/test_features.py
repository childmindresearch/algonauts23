from pathlib import Path

import h5py
import numpy as np
import pytest

from algonauts23.features import H5Writer


def test_feature_writer(tmp_path: Path):
    path = tmp_path / "test.h5"
    writer = H5Writer(path)

    data = []
    with writer as writer:
        writer.create_dataset("dummy/A", shape=(100, 50), dtype="float32")

        for _ in range(100):
            x = np.random.randn(1, 50)
            writer.put("dummy/A", x)
            data.append(x)
    data = np.concatenate(data)

    f = h5py.File(path, mode="r")
    data2 = np.asarray(f["dummy/A"])
    assert np.allclose(data, data2)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
