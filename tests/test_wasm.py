"""The WebAssembly build runs the same Rust code as the Python extension:
it must find exactly the same spikes."""

import json
import shutil
import subprocess
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from eegspike.detect import detect_candidates

WASM = resources.files("eegspike").joinpath("eegspike_core.wasm")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@pytest.mark.parametrize("seed", range(3))
def test_wasm_matches_numpy(tmp_path, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 15, (6, 20000))
    for t in rng.integers(100, 19900, 25):
        x[rng.integers(0, 6), t : t + 12] += np.hanning(12) * 250
    f = tmp_path / "x.f64"
    x.astype("<f8").tofile(f)
    smooth_n, refractory_n, k = 5, 50, 8.0
    out = subprocess.run(
        [
            "node",
            str(Path(__file__).with_name("wasm_run.mjs")),
            str(WASM),
            str(f),
            "6",
            "20000",
            str(k),
            str(smooth_n),
            str(refractory_n),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    w = json.loads(out.stdout)
    ch, s, sc = detect_candidates(x, 250.0, k=k, smooth_ms=20, refractory_ms=200, backend="numpy")
    assert np.array_equal(w["ch"], ch) and np.array_equal(w["s"], s)
    assert np.allclose(w["sc"], sc)
