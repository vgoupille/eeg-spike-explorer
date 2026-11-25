"""Automatic detection of interictal epileptiform spikes.

The detector is deliberately simple and transparent (it is a demo, not a
clinically validated tool):

1. The signal is band-passed (default 3-35 Hz) to keep the sharp
   component of spikes and drop slow waves and muscle activity.
2. The Teager-Kaiser energy operator, ``x[t]^2 - x[t-1] x[t+1]``, is large
   where the signal is both high in amplitude and high in frequency,
   which is what a spike (20-70 ms) looks like.
3. A robust per-channel threshold, ``median + k * MAD / 0.6745``, adapts to
   each electrode's background activity.
4. Local maxima above threshold, separated by a refractory period, are
   spike candidates on each channel.
5. Candidates seen on at least ``min_channels`` channels within a short
   window are merged into one *event* (a spike has a spatial field).

Steps 2-4 have two identical implementations: NumPy (reference) and Rust
(``eegspike_rs``, compiled with PyO3/maturin, multithreaded).
"""
from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd

try:  # optional Rust core
    import eegspike_rs as _rs
except ImportError:  # pragma: no cover
    _rs = None


def rust_available() -> bool:
    return _rs is not None


def _median(v: np.ndarray) -> float:
    return float(np.median(v))


def _detect_numpy(x: np.ndarray, k: float, smooth_n: int, refractory_n: int):
    """Reference implementation (must match rust/src/lib.rs)."""
    chs, samples, scores = [], [], []
    w = max(int(smooth_n), 1)
    h = w // 2
    for c, xc in enumerate(x):
        n = xc.size
        if n < 3:
            continue
        tk = np.zeros(n)
        tk[1:-1] = xc[1:-1] ** 2 - xc[:-2] * xc[2:]
        # centered moving average over [t-h, t-h+w), zero padded
        csum = np.concatenate(([0.0], np.cumsum(tk)))
        lo = np.clip(np.arange(n) - h, 0, n)
        hi = np.clip(np.arange(n) - h + w, 0, n)
        e = (csum[hi] - csum[lo]) / w
        med = _median(e)
        mad = _median(np.abs(e - med))
        thr = med + k * mad / 0.6745
        cand = np.flatnonzero(
            (e[1:-1] > thr) & (e[1:-1] > e[:-2]) & (e[1:-1] >= e[2:])
        ) + 1
        kept: list[list[float]] = []
        for t in cand:
            v = e[t]
            if kept and t - kept[-1][0] < refractory_n:
                if v > kept[-1][1]:
                    kept[-1] = [t, v]
                continue
            kept.append([t, v])
        for t, v in kept:
            chs.append(c)
            samples.append(int(t))
            scores.append(v / thr)
    return (np.asarray(chs, dtype=np.int64), np.asarray(samples, dtype=np.int64),
            np.asarray(scores, dtype=float))


def detect_candidates(x: np.ndarray, sfreq: float, k: float = 8.0,
                      smooth_ms: float = 20.0, refractory_ms: float = 200.0,
                      backend: str = "auto"):
    """Per-channel spike candidates on an array (n_channels, n_times).

    Returns (channel_idx, sample_idx, score) arrays. ``backend`` is
    "auto" (Rust if installed), "rust" or "numpy".
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    smooth_n = max(1, int(round(smooth_ms * sfreq / 1000)))
    refractory_n = max(1, int(round(refractory_ms * sfreq / 1000)))
    use_rust = backend == "rust" or (backend == "auto" and _rs is not None)
    if use_rust:
        if _rs is None:
            raise ImportError("eegspike_rs is not installed (see rust/README)")
        return _rs.detect(x, k, smooth_n, refractory_n)
    return _detect_numpy(x, k, smooth_n, refractory_n)


@dataclass
class SpikeResult:
    candidates: pd.DataFrame   # one row per (channel, spike)
    events: pd.DataFrame       # one row per multi-channel event
    ch_names: list[str]
    sfreq: float
    params: dict


def group_events(cand: pd.DataFrame, sfreq: float, window_ms: float = 80.0,
                 min_channels: int = 2) -> pd.DataFrame:
    """Merge per-channel candidates that occur within ``window_ms`` of each
    other into events seen on at least ``min_channels`` channels."""
    if cand.empty:
        return pd.DataFrame(columns=["time", "sample", "n_channels", "channels",
                                     "lead_channel", "score"])
    win = window_ms * sfreq / 1000
    c = cand.sort_values("sample").reset_index(drop=True)
    groups = np.cumsum(np.r_[0, np.diff(c["sample"].to_numpy()) > win])
    rows = []
    for _, g in c.groupby(groups):
        chans = sorted(set(g["channel"]))
        if len(chans) < min_channels:
            continue
        lead = g.loc[g["score"].idxmax()]
        rows.append({
            "time": lead["sample"] / sfreq,
            "sample": int(lead["sample"]),
            "n_channels": len(chans),
            "channels": ";".join(chans),
            "lead_channel": lead["channel"],
            "score": float(g["score"].max()),
        })
    return pd.DataFrame(rows)


def detect_spikes(raw: mne.io.BaseRaw, band: tuple[float, float] = (3.0, 35.0),
                  k: float = 15.0, smooth_ms: float = 20.0,
                  refractory_ms: float = 200.0, window_ms: float = 80.0,
                  min_channels: int = 3, backend: str = "auto") -> SpikeResult:
    """Detect interictal spikes on an MNE Raw object."""
    sfreq = raw.info["sfreq"]
    xr = raw.copy().filter(*band, verbose="error").get_data() * 1e6  # µV
    ch_idx, samp, score = detect_candidates(xr, sfreq, k, smooth_ms,
                                            refractory_ms, backend)
    names = np.array(raw.ch_names)
    cand = pd.DataFrame({
        "channel": names[ch_idx] if len(ch_idx) else np.array([], dtype=str),
        "sample": samp,
        "time": samp / sfreq,
        "score": score,
    })
    events = group_events(cand, sfreq, window_ms, min_channels)
    params = dict(band=band, k=k, smooth_ms=smooth_ms, refractory_ms=refractory_ms,
                  window_ms=window_ms, min_channels=min_channels,
                  backend="rust" if (backend == "rust" or (backend == "auto" and _rs)) else "numpy")
    return SpikeResult(cand, events, list(raw.ch_names), sfreq, params)
