"""eegspike: detection of interictal epileptiform spikes and
inter-electrode connectivity on scalp EEG, built on MNE-Python.

Pipeline: load EDF -> filter -> detect spikes (NumPy or Rust core)
-> connectivity (wPLI, correlation, spike co-occurrence) -> graph
-> interactive WebGPU viewer.
"""
from .io import load_eeg
from .detect import detect_spikes, group_events, rust_available
from .connectivity import compute_connectivity, connectivity_graph

__version__ = "0.1.0"
__all__ = [
    "load_eeg", "detect_spikes", "group_events", "rust_available",
    "compute_connectivity", "connectivity_graph",
]
