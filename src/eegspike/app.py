"""Review application: a small FastAPI server around the viewer.

    eegspike review recording.edf -o results/

serves the interactive viewer with a review panel; every decision is saved to
``results/review.json`` immediately.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .review import STATUSES, ReviewStore
from .viewer import build_payload, render_viewer


class StatusIn(BaseModel):
    status: str


class AddIn(BaseModel):
    t: float


def create_app(payload: dict, store: ReviewStore, out_dir: str | Path, stem: str) -> FastAPI:
    app = FastAPI(title="EEG Spike Explorer · relecture")
    html = render_viewer(payload, review=True)
    duration = payload["meta"]["duration"]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return html

    @app.get("/api/review")
    def get_review():
        return store.state()

    @app.put("/api/review/events/{sample}")
    def set_status(sample: int, body: StatusIn):
        if body.status not in STATUSES:
            raise HTTPException(422, f"status must be one of {STATUSES}")
        try:
            store.set_status(sample, body.status)
        except KeyError:
            raise HTTPException(404, f"no detected event at sample {sample}")
        return store.stats()

    @app.post("/api/review/added", status_code=201)
    def add(body: AddIn):
        if not 0 <= body.t <= duration:
            raise HTTPException(422, "t is outside the recording")
        item = store.add(body.t)
        return {"item": item, "stats": store.stats()}

    @app.delete("/api/review/added/{item_id}")
    def remove(item_id: str):
        try:
            store.remove(item_id)
        except KeyError:
            raise HTTPException(404, f"no added spike {item_id}")
        return store.stats()

    @app.post("/api/review/export")
    def export():
        paths = store.export(out_dir, stem)
        return {"files": {k: Path(v).name for k, v in paths.items()}, "stats": store.stats()}

    return app


def build_review_app(edf: str, out_dir: str | Path = "results", **analysis_kw) -> FastAPI:
    from .pipeline import analyze

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = analyze(edf, **analysis_kw)
    res.spikes.events.to_csv(out / "spike_events.csv", index=False)
    payload = build_payload(res.raw, res.spikes, res.mats, Path(edf).name, res.timings)
    store = ReviewStore.open(out / "review.json", res.raw.info["sfreq"], res.spikes.events)
    return create_app(payload, store, out, Path(edf).stem)
