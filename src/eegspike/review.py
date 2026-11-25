"""Review store: an expert validates, rejects or adds spikes.

Decisions are saved to ``review.json`` after every change, so a review can
be stopped and resumed. Events are identified by their sample index, which
stays stable as long as the recording and the sampling rate do not change.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

STATUSES = ("pending", "accepted", "rejected")


@dataclass
class ReviewStore:
    path: Path
    sfreq: float
    events: pd.DataFrame  # detector output (spike_events)
    decisions: dict[int, str] = field(default_factory=dict)  # sample -> status
    added: list[dict] = field(default_factory=list)  # [{"id", "t"}]
    reviewer: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---- persistence ----------------------------------------------------
    @classmethod
    def open(cls, path: str | Path, sfreq: float, events: pd.DataFrame) -> ReviewStore:
        path = Path(path)
        store = cls(path=path, sfreq=sfreq, events=events)
        if path.exists():
            saved = json.loads(path.read_text("utf-8"))
            known = set(int(s) for s in events["sample"]) if len(events) else set()
            store.decisions = {
                int(k): v for k, v in saved.get("decisions", {}).items() if int(k) in known and v in STATUSES
            }
            store.added = saved.get("added", [])
            store.reviewer = saved.get("reviewer", "")
        return store

    def save(self) -> None:
        data = {
            "sfreq": self.sfreq,
            "reviewer": self.reviewer,
            "decisions": {str(k): v for k, v in sorted(self.decisions.items())},
            "added": self.added,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), "utf-8")
        tmp.replace(self.path)

    # ---- edits ------------------------------------------------------------
    def set_status(self, sample: int, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if sample not in set(int(s) for s in self.events["sample"]):
            raise KeyError(sample)
        with self._lock:
            if status == "pending":
                self.decisions.pop(sample, None)
            else:
                self.decisions[sample] = status
            self.save()

    def add(self, t: float) -> dict:
        with self._lock:
            n = max([int(a["id"][1:]) for a in self.added] + [0]) + 1
            item = {"id": f"m{n}", "t": round(float(t), 4)}
            self.added.append(item)
            self.added.sort(key=lambda a: a["t"])
            self.save()
        return item

    def remove(self, item_id: str) -> None:
        with self._lock:
            before = len(self.added)
            self.added = [a for a in self.added if a["id"] != item_id]
            if len(self.added) == before:
                raise KeyError(item_id)
            self.save()

    # ---- summaries ----------------------------------------------------------
    def stats(self) -> dict:
        acc = sum(v == "accepted" for v in self.decisions.values())
        rej = sum(v == "rejected" for v in self.decisions.values())
        n = int(len(self.events))
        return {
            "n_events": n,
            "reviewed": acc + rej,
            "accepted": acc,
            "rejected": rej,
            "added": len(self.added),
            # share of reviewed detections that the expert confirmed
            "precision": acc / (acc + rej) if acc + rej else None,
            # share of true spikes (confirmed + missed) that the detector found,
            # only meaningful once the whole recording has been reviewed
            "recall_if_complete": acc / (acc + len(self.added)) if acc + len(self.added) else None,
        }

    def state(self) -> dict:
        return {
            "decisions": {str(k): v for k, v in self.decisions.items()},
            "added": self.added,
            "stats": self.stats(),
        }

    def table(self) -> pd.DataFrame:
        """One row per spike, detector and expert combined."""
        rows = []
        for r in self.events.to_dict("records"):
            rows.append(
                {
                    "onset": float(r["time"]),
                    "duration": 0.0,
                    "source": "detector",
                    "status": self.decisions.get(int(r["sample"]), "pending"),
                    "lead_channel": r["lead_channel"],
                    "n_channels": int(r["n_channels"]),
                    "score": float(r["score"]),
                }
            )
        for a in self.added:
            rows.append(
                {
                    "onset": float(a["t"]),
                    "duration": 0.0,
                    "source": "expert",
                    "status": "added",
                    "lead_channel": "",
                    "n_channels": 0,
                    "score": float("nan"),
                }
            )
        return pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)

    def export(self, out_dir: str | Path, stem: str) -> dict[str, str]:
        """Write the review as CSV, BIDS events.tsv and MNE annotations."""
        import mne

        out = Path(out_dir)
        t = self.table()
        paths = {}
        p = out / f"{stem}_review.csv"
        t.to_csv(p, index=False)
        paths["csv"] = str(p)
        # BIDS: onset/duration/trial_type, only the spikes the expert stands by
        kept = t[t.status.isin(["accepted", "added"])]
        bids = pd.DataFrame(
            {
                "onset": kept.onset.round(4),
                "duration": 0.0,
                "trial_type": "spike",
                "value": kept.source.map({"detector": "validated", "expert": "added"}),
            }
        )
        p = out / f"{stem}_events.tsv"
        bids.to_csv(p, sep="\t", index=False)
        paths["bids"] = str(p)
        ann = mne.Annotations(
            onset=t.onset.to_numpy(),
            duration=t.duration.to_numpy(),
            description=[f"spike/{s}" for s in t.status],
        )
        p = out / f"{stem}_spikes-annot.fif"
        ann.save(p, overwrite=True, verbose="error")
        paths["mne"] = str(p)
        return paths
