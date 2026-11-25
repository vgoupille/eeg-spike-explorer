import numpy as np
import pandas as pd
import pytest

from eegspike.detect import detect_candidates, group_events, rust_available, _detect_numpy
from eegspike.connectivity import spike_cooccurrence, connectivity_graph

FS = 250.0


def synthetic(n_ch=6, seconds=60, spikes_at=(10.0, 25.5, 40.2), seed=0):
    """Background noise + triangular 50 ms spikes on channels 0-2."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    x = rng.normal(0, 15, (n_ch, n))
    w = int(0.05 * FS)
    tri = np.concatenate([np.linspace(0, 1, w // 2), np.linspace(1, 0, w - w // 2)]) * 250
    for t in spikes_at:
        s = int(t * FS)
        x[:3, s:s + w] += tri
    return x


def test_detects_planted_spikes():
    x = synthetic()
    ch, s, sc = detect_candidates(x, FS, k=15, backend="numpy")
    cand = pd.DataFrame({"channel": [f"C{c}" for c in ch], "sample": s, "time": s / FS, "score": sc})
    ev = group_events(cand, FS, min_channels=3)
    assert len(ev) == 3
    assert np.allclose(sorted(ev.time), [10.0, 25.5, 40.2], atol=0.06)


@pytest.mark.skipif(not rust_available(), reason="eegspike_rs not built")
@pytest.mark.parametrize("seed", range(5))
def test_rust_matches_numpy(seed):
    x = synthetic(seed=seed) + np.random.default_rng(seed).normal(0, 5, (6, 15000))
    a = detect_candidates(x, FS, k=8, backend="numpy")
    b = detect_candidates(x, FS, k=8, backend="rust")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.allclose(a[2], b[2])


def test_cooccurrence_and_graph():
    names = ["A", "B", "C", "D"]
    ev = pd.DataFrame({"channels": ["A;B", "A;B;C", "C;D"]})
    m = spike_cooccurrence(ev, names)
    assert np.allclose(m, m.T)
    assert m[0, 1] == pytest.approx(1.0)        # A and B always together
    assert m[0, 3] == 0.0
    g = connectivity_graph(m, names, density=0.5)
    assert g.has_edge("A", "B")
