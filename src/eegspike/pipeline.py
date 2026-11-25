"""The full analysis in one call, shared by `eegspike run` and `eegspike review`."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mne
import numpy as np

from .connectivity import compute_connectivity
from .detect import SpikeResult, detect_candidates, detect_spikes, rust_available
from .io import load_eeg


@dataclass
class Analysis:
    raw: mne.io.BaseRaw
    spikes: SpikeResult
    mats: dict
    timings: dict


def analyze(
    edf: str,
    tmin=None,
    tmax=None,
    k: float = 15.0,
    min_channels: int = 3,
    backend: str = "auto",
    benchmark: bool = True,
) -> Analysis:
    raw = load_eeg(edf, tmin=tmin, tmax=tmax)
    spikes = detect_spikes(raw, k=k, min_channels=min_channels, backend=backend)
    timings: dict = {}
    if benchmark:
        # time both detection cores on the same data and check they agree
        xr = raw.copy().filter(3, 35, verbose="error").get_data() * 1e6
        res = {}
        for b in ["numpy"] + (["rust"] if rust_available() else []):
            best = np.inf
            for _ in range(5):
                t = time.perf_counter()
                res[b] = detect_candidates(xr, raw.info["sfreq"], k=k, backend=b)
                best = min(best, time.perf_counter() - t)
            timings[f"detect_{b}_ms"] = best * 1000
        if "rust" in res:
            timings["rust_matches_numpy"] = bool(
                all(np.array_equal(res["numpy"][i], res["rust"][i]) for i in (0, 1))
            )
    mats = compute_connectivity(raw, spikes.events)
    return Analysis(raw, spikes, mats, timings)
