"""Inter-electrode connectivity and graph analysis.

Three complementary matrices (n_channels x n_channels):

* ``wpli_<band>``: weighted Phase Lag Index (MNE-Connectivity) on 2-s epochs.
  It ignores zero-lag coupling, so it is robust to volume conduction.
* ``corr``: Pearson correlation of the filtered signals (simple baseline).
* ``spike_cooc``: how often two channels take part in the same spike event
  (Jaccard index). It maps the spatial spread of the irritative zone.
"""
from __future__ import annotations

import mne
import networkx as nx
import numpy as np
import pandas as pd

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def wpli(raw: mne.io.BaseRaw, bands=BANDS, epoch_s: float = 2.0) -> dict[str, np.ndarray]:
    from mne_connectivity import spectral_connectivity_epochs

    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_s, preload=True,
                                          verbose="error")
    fmin = tuple(b[0] for b in bands.values())
    fmax = tuple(b[1] for b in bands.values())
    con = spectral_connectivity_epochs(
        epochs, method="wpli", mode="multitaper", sfreq=raw.info["sfreq"],
        fmin=fmin, fmax=fmax, faverage=True, verbose="error",
    )
    data = con.get_data(output="dense")  # (n, n, n_bands), lower triangle
    out = {}
    for i, name in enumerate(bands):
        m = data[:, :, i]
        m = m + m.T
        np.fill_diagonal(m, 0)
        out[f"wpli_{name}"] = m
    return out


def correlation(raw: mne.io.BaseRaw) -> np.ndarray:
    m = np.corrcoef(raw.get_data())
    np.fill_diagonal(m, 0)
    return np.abs(m)


def spike_cooccurrence(events: pd.DataFrame, ch_names: list[str]) -> np.ndarray:
    n = len(ch_names)
    idx = {c: i for i, c in enumerate(ch_names)}
    member = np.zeros((len(events), n), dtype=bool)
    for r, chans in enumerate(events.get("channels", [])):
        for c in str(chans).split(";"):
            if c in idx:
                member[r, idx[c]] = True
    inter = member.T.astype(float) @ member.astype(float)
    count = member.sum(0).astype(float)
    union = count[:, None] + count[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(m, 0)
    return m


def compute_connectivity(raw: mne.io.BaseRaw, events: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    mats = {"corr": correlation(raw)}
    mats.update(wpli(raw))
    if events is not None:
        mats["spike_cooc"] = spike_cooccurrence(events, list(raw.ch_names))
    return mats


def connectivity_graph(mat: np.ndarray, ch_names: list[str], density: float = 0.15) -> nx.Graph:
    """Keep the strongest ``density`` fraction of edges and build a graph.
    Node attributes: strength, degree centrality, betweenness."""
    n = len(ch_names)
    iu = np.triu_indices(n, 1)
    w = mat[iu]
    k = max(1, int(round(density * w.size)))
    thr = np.sort(w)[-k]
    g = nx.Graph()
    g.add_nodes_from(ch_names)
    for i, j, v in zip(*iu, w):
        if v >= thr and v > 0:
            g.add_edge(ch_names[i], ch_names[j], weight=float(v))
    strength = dict(g.degree(weight="weight"))
    btw = nx.betweenness_centrality(g, weight=None)
    deg = nx.degree_centrality(g)
    nx.set_node_attributes(g, strength, "strength")
    nx.set_node_attributes(g, btw, "betweenness")
    nx.set_node_attributes(g, deg, "degree_centrality")
    return g


def graph_table(g: nx.Graph) -> pd.DataFrame:
    return (pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient="index")
            .rename_axis("channel").sort_values("strength", ascending=False))
