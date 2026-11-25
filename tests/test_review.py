import json

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from eegspike.app import create_app  # noqa: E402
from eegspike.review import ReviewStore  # noqa: E402

EVENTS = pd.DataFrame({
    "time": [1.0, 2.5, 4.0], "sample": [250, 625, 1000], "n_channels": [3, 5, 4],
    "channels": ["A;B;C", "A;B;C;D;E", "B;C;D;E"], "lead_channel": ["A", "C", "D"],
    "score": [2.0, 3.5, 1.4],
})
PAYLOAD = {"meta": {"duration": 10.0}}


@pytest.fixture
def client(tmp_path):
    store = ReviewStore.open(tmp_path / "review.json", 250.0, EVENTS)
    return TestClient(create_app(PAYLOAD, store, tmp_path, "rec")), tmp_path


def test_page_has_review_enabled(client):
    c, _ = client
    html = c.get("/").text
    assert '"api":"/api/review"' in html


def test_decisions_are_saved_and_counted(client):
    c, tmp = client
    assert c.put("/api/review/events/250", json={"status": "accepted"}).status_code == 200
    s = c.put("/api/review/events/625", json={"status": "rejected"}).json()
    assert s["reviewed"] == 2 and s["precision"] == 0.5
    saved = json.loads((tmp / "review.json").read_text())
    assert saved["decisions"] == {"250": "accepted", "625": "rejected"}
    # back to pending removes the decision
    s = c.put("/api/review/events/625", json={"status": "pending"}).json()
    assert s["reviewed"] == 1 and s["precision"] == 1.0


def test_bad_requests(client):
    c, _ = client
    assert c.put("/api/review/events/999", json={"status": "accepted"}).status_code == 404
    assert c.put("/api/review/events/250", json={"status": "maybe"}).status_code == 422
    assert c.post("/api/review/added", json={"t": 42.0}).status_code == 422


def test_add_remove_and_resume(client):
    c, tmp = client
    r = c.post("/api/review/added", json={"t": 3.2})
    assert r.status_code == 201 and r.json()["stats"]["added"] == 1
    item = r.json()["item"]
    c.put("/api/review/events/1000", json={"status": "accepted"})
    # a new store on the same file resumes the review
    again = ReviewStore.open(tmp / "review.json", 250.0, EVENTS)
    assert again.decisions == {1000: "accepted"} and again.added[0]["t"] == 3.2
    assert c.delete(f"/api/review/added/{item['id']}").json()["added"] == 0
    assert c.delete("/api/review/added/m99").status_code == 404


def test_export(client):
    c, tmp = client
    c.put("/api/review/events/250", json={"status": "accepted"})
    c.put("/api/review/events/625", json={"status": "rejected"})
    c.post("/api/review/added", json={"t": 7.5})
    files = c.post("/api/review/export").json()["files"]
    bids = pd.read_csv(tmp / files["bids"], sep="\t")
    assert list(bids.columns) == ["onset", "duration", "trial_type", "value"]
    assert np.allclose(bids.onset, [1.0, 7.5]) and list(bids.value) == ["validated", "added"]
    import mne
    ann = mne.read_annotations(tmp / files["mne"])
    assert sorted(set(ann.description)) == ["spike/accepted", "spike/added", "spike/pending", "spike/rejected"]
