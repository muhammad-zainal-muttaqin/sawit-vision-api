"""SQLite log of every detection request."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    filename    TEXT,
    width       INTEGER,
    height      INTEGER,
    n_objects   INTEGER NOT NULL,
    counts_json TEXT    NOT NULL,
    latency_ms  REAL    NOT NULL,
    conf        REAL    NOT NULL
);
"""


class DetectionLog:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, filename: str | None, width: int, height: int, counts: dict[str, int],
            latency_ms: float, conf: float) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO detections (created_at, filename, width, height, n_objects, counts_json, latency_ms, conf)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), filename, width, height,
                 sum(counts.values()), json.dumps(counts), round(latency_ms, 2), conf),
            )
            return int(cur.lastrowid)

    def recent(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "counts": json.loads(r["counts_json"])} for r in rows]

    def stats(self) -> dict:
        totals: dict[str, int] = {}
        with self._conn() as c:
            rows = c.execute("SELECT counts_json, latency_ms FROM detections").fetchall()
        for r in rows:
            for k, v in json.loads(r["counts_json"]).items():
                totals[k] = totals.get(k, 0) + v
        lat = sorted(r["latency_ms"] for r in rows)
        return {
            "requests": len(rows),
            "objects_by_class": totals,
            "latency_ms_median": lat[len(lat) // 2] if lat else None,
        }
