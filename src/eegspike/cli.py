"""Command line.

eegspike run recording.edf -o results/      # analyse + static viewer
eegspike review recording.edf -o results/   # review application in the browser
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _common(p):
    p.add_argument("edf")
    p.add_argument("-o", "--out", default="results")
    p.add_argument("--tmin", type=float)
    p.add_argument("--tmax", type=float)
    p.add_argument("-k", type=float, default=15.0, help="seuil en MAD (défaut 15)")
    p.add_argument("--min-channels", type=int, default=3)
    p.add_argument("--backend", choices=["auto", "rust", "numpy"], default="auto")


def cmd_run(a):
    from .connectivity import connectivity_graph, graph_table
    from .pipeline import analyze
    from .viewer import build_payload, write_viewer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = analyze(a.edf, a.tmin, a.tmax, a.k, a.min_channels, a.backend)
    raw, spikes, mats, timings = res.raw, res.spikes, res.mats, res.timings
    ev = spikes.events
    dur = raw.times[-1]
    print(f"[1/4] {Path(a.edf).name}: {len(raw.ch_names)} canaux, {dur:.0f} s, {raw.info['sfreq']:.0f} Hz")
    print(
        f"[2/4] {len(spikes.candidates)} candidats, {len(ev)} événements "
        f"({len(ev) / (dur / 60):.1f}/min), détecteur {spikes.params['backend']}"
    )
    if "detect_rust_ms" in timings:
        print(
            f"      Rust {timings['detect_rust_ms']:.1f} ms vs NumPy {timings['detect_numpy_ms']:.1f} ms, "
            f"résultats identiques : {timings['rust_matches_numpy']}"
        )
    hubs = graph_table(connectivity_graph(mats["spike_cooc"], raw.ch_names))
    print(
        f"[3/4] connectivité : {', '.join(mats)} ; "
        f"canaux centraux (co-occurrence) : {', '.join(hubs.index[:3])}"
    )

    spikes.candidates.to_csv(out / "spike_candidates.csv", index=False)
    ev.to_csv(out / "spike_events.csv", index=False)
    hubs.to_csv(out / "graph_hubs.csv")
    np.savez_compressed(out / "connectivity.npz", ch_names=np.array(raw.ch_names), **mats)
    summary = {
        "file": Path(a.edf).name,
        "duration_s": float(dur),
        "n_events": int(len(ev)),
        "events_per_min": float(len(ev) / (dur / 60)),
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spikes.params.items()},
        "timings": timings,
        "top_channels": hubs.index[:5].tolist(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    html = write_viewer(build_payload(raw, spikes, mats, Path(a.edf).name, timings), out / "viewer.html")
    print(f"[4/4] résultats dans {out}/ ; ouvrir {html}")


def cmd_review(a):
    try:
        import uvicorn

        from .app import build_review_app
    except ImportError:  # pragma: no cover
        raise SystemExit("Le mode relecture demande FastAPI : pip install 'eegspike[app]'") from None
    print(f"Analyse de {Path(a.edf).name}…")
    app = build_review_app(
        a.edf, a.out, tmin=a.tmin, tmax=a.tmax, k=a.k, min_channels=a.min_channels, backend=a.backend
    )
    url = f"http://127.0.0.1:{a.port}"
    print(f"Relecture prête : {url}  (décisions enregistrées dans {a.out}/review.json, Ctrl+C pour quitter)")
    if not a.no_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="eegspike", description="Détection de pointes intercritiques et connectivité EEG"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyser un fichier EDF et générer viewer.html")
    _common(r)
    rv = sub.add_parser("review", help="relire les pointes détectées dans le navigateur")
    _common(rv)
    rv.add_argument("--port", type=int, default=8765)
    rv.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    {"run": cmd_run, "review": cmd_review}[a.cmd](a)


if __name__ == "__main__":
    main()
