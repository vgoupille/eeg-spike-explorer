"""Loading and preprocessing EEG recordings with MNE-Python."""

from __future__ import annotations

import mne
import numpy as np

# Classic 18-channel longitudinal bipolar montage ("double banana") + Fz-Cz, Cz-Pz
DOUBLE_BANANA = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "Fz-Cz",
    "Cz-Pz",
]


def load_eeg(
    path: str,
    channels: list[str] | None = None,
    l_freq: float = 1.0,
    h_freq: float = 70.0,
    line_freq: float = 50.0,
    sfreq: float = 250.0,
    tmin: float | None = None,
    tmax: float | None = None,
) -> mne.io.BaseRaw:
    """Read an EDF file and return a filtered, resampled MNE Raw (in volts).

    Parameters
    ----------
    channels : channel names to keep. Default: the double-banana bipolar
        montage if all its channels exist, otherwise every EEG channel.
    """
    raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
    if tmin is not None or tmax is not None:
        raw.crop(tmin or 0.0, tmax)
    if channels is None:
        channels = DOUBLE_BANANA if set(DOUBLE_BANANA) <= set(raw.ch_names) else None
    if channels is not None:
        raw.pick(channels)
    else:
        raw.pick("eeg")
    raw.load_data(verbose="error")

    # Some EDF exports declare no physical unit: MNE then reads microvolts as
    # volts. Real scalp EEG is ~1e-5 V, so values ~1 or larger mean "µV".
    if np.median(np.abs(raw._data)) > 1e-2:
        raw.apply_function(lambda x: x * 1e-6)

    if line_freq and line_freq < raw.info["sfreq"] / 2:
        raw.notch_filter(line_freq, verbose="error")
    raw.filter(l_freq, h_freq, verbose="error")
    if sfreq and sfreq < raw.info["sfreq"]:
        raw.resample(sfreq, verbose="error")
    return raw
