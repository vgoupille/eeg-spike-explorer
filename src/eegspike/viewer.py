"""Export results to a single self-contained HTML viewer (WebGPU + fallback)."""

from __future__ import annotations

import base64
import json
from importlib import resources
from pathlib import Path

import mne
import numpy as np

from .connectivity import connectivity_graph

# Approximate 2D positions of 10-20 electrodes (nose up, unit circle).
POS_1020 = {
    "Fp1": (-0.31, 0.95),
    "Fp2": (0.31, 0.95),
    "Fpz": (0.0, 1.0),
    "F7": (-0.81, 0.59),
    "F3": (-0.42, 0.52),
    "Fz": (0.0, 0.47),
    "F4": (0.42, 0.52),
    "F8": (0.81, 0.59),
    "T3": (-1.0, 0.0),
    "T7": (-1.0, 0.0),
    "C3": (-0.5, 0.0),
    "Cz": (0.0, 0.0),
    "C4": (0.5, 0.0),
    "T4": (1.0, 0.0),
    "T8": (1.0, 0.0),
    "T5": (-0.81, -0.59),
    "P7": (-0.81, -0.59),
    "P3": (-0.42, -0.52),
    "Pz": (0.0, -0.47),
    "P4": (0.42, -0.52),
    "T6": (0.81, -0.59),
    "P8": (0.81, -0.59),
    "O1": (-0.31, -0.95),
    "O2": (0.31, -0.95),
    "Oz": (0.0, -1.0),
}


def _channel_pos(name: str):
    parts = [p.strip() for p in name.replace("EEG", "").split("-")]
    pts = [POS_1020[p] for p in parts if p in POS_1020]
    if not pts:
        return None
    return [float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))]


MATRIX_LABELS = {
    "spike_cooc": "Co-occurrence des pointes",
    "wpli_delta": "wPLI delta (1-4 Hz)",
    "wpli_theta": "wPLI thêta (4-8 Hz)",
    "wpli_alpha": "wPLI alpha (8-13 Hz)",
    "wpli_beta": "wPLI bêta (13-30 Hz)",
    "corr": "Corrélation |r|",
}


def build_payload(raw: mne.io.BaseRaw, spikes, mats: dict, source: str, timings: dict | None = None) -> dict:
    x = np.asarray(raw.get_data()) * 1e6  # µV
    # int16 quantisation keeps the file small (~2 bytes / sample)
    scale = float(max(np.abs(x).max() / 32000, 0.01))
    q = np.round(x / scale).astype("<i2")
    # detection band, so the browser can re-run the detector (WebAssembly)
    xd = np.asarray(raw.copy().filter(*spikes.params["band"], verbose="error").get_data()) * 1e6
    dscale = float(max(np.abs(xd).max() / 32000, 1e-3))
    qd = np.round(xd / dscale).astype("<i2")
    ch = list(raw.ch_names)
    idx = {c: i for i, c in enumerate(ch)}
    cand = spikes.candidates
    events = [
        {
            "t": round(float(r.time), 4),
            "s": int(r.sample),
            "n": int(r.n_channels),
            "lead": idx[r.lead_channel],
            "score": round(float(r.score), 2),
            "chs": [idx[c] for c in r.channels.split(";")],
        }
        for r in spikes.events.itertuples()
    ]
    order = [k for k in MATRIX_LABELS if k in mats]
    graphs = {}
    for k in order:
        g = connectivity_graph(mats[k], ch)
        graphs[k] = {
            "strength": [round(g.nodes[c]["strength"], 4) for c in ch],
            "betweenness": [round(g.nodes[c]["betweenness"], 4) for c in ch],
            "edges": [[idx[a], idx[b], round(d["weight"], 4)] for a, b, d in g.edges(data=True)],
        }
    return {
        "meta": {
            "source": source,
            "sfreq": raw.info["sfreq"],
            "n_ch": len(ch),
            "n_t": int(x.shape[1]),
            "duration": float(x.shape[1] / raw.info["sfreq"]),
            "ch_names": ch,
            "scale": scale,
            "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spikes.params.items()},
            "timings": timings or {},
        },
        "signal": base64.b64encode(q.tobytes()).decode(),
        "det_signal": base64.b64encode(qd.tobytes()).decode(),
        "det_scale": dscale,
        "cand": {"ch": [idx[c] for c in cand["channel"]], "s": cand["sample"].astype(int).tolist()},
        "events": events,
        "matrices": {
            k: {"label": MATRIX_LABELS[k], "values": np.round(mats[k], 4).ravel().tolist()} for k in order
        },
        "graphs": graphs,
        "pos": [_channel_pos(c) for c in ch],
    }


def render_viewer(payload: dict, standalone: bool = True, review: bool = False) -> str:
    """Fill the HTML template. ``standalone`` adds the <html>/<head> skeleton so
    the file opens directly in a browser; ``review`` turns on the review panel,
    which talks to the API of ``eegspike review``."""
    template = resources.files("eegspike").joinpath("viewer_template.html").read_text("utf-8")
    html = template.replace("/*__EEG_DATA__*/null", json.dumps(payload, separators=(",", ":")))
    html = html.replace("/*__EEG_REVIEW__*/null", '{"api":"/api/review"}' if review else "null")
    wasm = resources.files("eegspike").joinpath("eegspike_core.wasm")
    if wasm.is_file():
        html = html.replace('/*__WASM__*/""', json.dumps(base64.b64encode(wasm.read_bytes()).decode()))
    if standalone:
        html = (
            '<!doctype html>\n<html lang="fr">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "<style>html{color-scheme:light dark}body{margin:0}</style>\n</head>\n<body>\n"
            + html
            + "\n</body>\n</html>\n"
        )
    return html


def write_viewer(payload: dict, out: str | Path, standalone: bool = True) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_viewer(payload, standalone), encoding="utf-8")
    return out
