"""Command line: eegspike run recording.edf -o results/"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eegspike", description="Détection de pointes intercritiques et connectivité EEG")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyser un fichier EDF")
    r.add_argument("edf")
    r.add_argument("-o", "--out", default="results")
    r.add_argument("--tmin", type=float)
    r.add_argument("--tmax", type=float)
    r.add_argument("-k", type=float, default=15.0, help="seuil en MAD (défaut 15)")
    r.add_argument("--min-channels", type=int, default=3)
    r.add_argument("--backend", choices=["auto", "rust", "numpy"], default="auto")
    a = ap.parse_args(argv)

    from . import load_eeg, detect_spikes, rust_available
    from .connectivity import compute_connectivity, connectivity_graph, graph_table
    from .detect import detect_candidates
    from .viewer import build_payload, write_viewer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    raw = load_eeg(a.edf, tmin=a.tmin, tmax=a.tmax)
    t_load = time.perf_counter() - t0
    print(f"[1/4] {Path(a.edf).name}: {len(raw.ch_names)} canaux, {raw.times[-1]:.0f} s, {raw.info['sfreq']:.0f} Hz ({t_load:.1f} s)")

    spikes = detect_spikes(raw, k=a.k, min_channels=a.min_channels, backend=a.backend)
    ev = spikes.events
    print(f"[2/4] {len(spikes.candidates)} candidats, {len(ev)} événements "
          f"({len(ev) / (raw.times[-1] / 60):.1f}/min), détecteur {spikes.params['backend']}")

    # quick benchmark of the two detection cores on the same data
    timings = {}
    xr = raw.copy().filter(3, 35, verbose="error").get_data() * 1e6
    backends = ["numpy"] + (["rust"] if rust_available() else [])
    res = {}
    for b in backends:
        best = np.inf
        for _ in range(5):
            t = time.perf_counter()
            res[b] = detect_candidates(xr, raw.info["sfreq"], k=a.k, backend=b)
            best = min(best, time.perf_counter() - t)
        timings[f"detect_{b}_ms"] = best * 1000
    if "rust" in res:
        same = all(np.array_equal(res["numpy"][i], res["rust"][i]) for i in (0, 1))
        timings["rust_matches_numpy"] = bool(same)
        print(f"      Rust {timings['detect_rust_ms']:.1f} ms vs NumPy {timings['detect_numpy_ms']:.1f} ms, résultats identiques : {same}")

    mats = compute_connectivity(raw, ev)
    g = connectivity_graph(mats["spike_cooc"], raw.ch_names)
    hubs = graph_table(g)
    print(f"[3/4] connectivité : {', '.join(mats)} ; canaux centraux (co-occurrence) : {', '.join(hubs.index[:3])}")

    spikes.candidates.to_csv(out / "spike_candidates.csv", index=False)
    ev.to_csv(out / "spike_events.csv", index=False)
    hubs.to_csv(out / "graph_hubs.csv")
    np.savez_compressed(out / "connectivity.npz", ch_names=np.array(raw.ch_names), **mats)
    summary = {
        "file": Path(a.edf).name, "duration_s": float(raw.times[-1]),
        "n_events": int(len(ev)), "events_per_min": float(len(ev) / (raw.times[-1] / 60)),
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spikes.params.items()},
        "timings": timings, "top_channels": hubs.index[:5].tolist(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    html = write_viewer(build_payload(raw, spikes, mats, Path(a.edf).name, timings), out / "viewer.html")
    print(f"[4/4] résultats dans {out}/ ; ouvrir {html}")


if __name__ == "__main__":
    main()
