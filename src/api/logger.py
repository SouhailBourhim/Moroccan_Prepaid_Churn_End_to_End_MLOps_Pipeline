"""SQLite-backed prediction logger for the churn API.

Writes one row to `prediction_requests` and one row per subscriber to
`prediction_rows` for every /predict call. Designed for local development
and staging — swap for a cloud sink (BigQuery, S3 parquet) in production.

The database path comes from CHURN_PREDICTIONS_DB (see app.py); in Docker it
lives on a named volume so the log — and the drift baseline comparison that
reads it — survives container restarts.

WAL journal mode is enabled so multiple uvicorn workers can write concurrently
without locking errors. Rows older than the retention window are pruned at
startup.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger as _log

_CREATE_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS prediction_requests (
    request_id  TEXT    PRIMARY KEY,
    ts          TEXT    NOT NULL,
    model_name  TEXT,
    n_subscribers INTEGER,
    threshold   REAL,
    latency_ms  REAL
);

CREATE TABLE IF NOT EXISTS prediction_rows (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id        TEXT    NOT NULL,
    subscriber_idx    INTEGER NOT NULL,
    churn_probability REAL,
    churn_prediction  INTEGER,
    features_json     TEXT,
    subscriber_id     TEXT
);

CREATE INDEX IF NOT EXISTS idx_rows_request ON prediction_rows (request_id);
CREATE INDEX IF NOT EXISTS idx_requests_ts  ON prediction_requests (ts);
"""


def _round(value: float | None, digits: int) -> float | None:
    return round(float(value), digits) if value is not None else None


class PredictionLogger:
    """Thread-safe SQLite prediction logger."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_CREATE_SQL)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns introduced after a database was first created."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(prediction_rows)")}
        if "subscriber_id" not in cols:
            conn.execute("ALTER TABLE prediction_rows ADD COLUMN subscriber_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rows_subscriber ON prediction_rows (subscriber_id)"
        )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def log_request(
        self,
        request_id: str,
        model_name: str,
        n_subscribers: int,
        threshold: float,
        latency_ms: float,
        probs: list[float],
        predictions: list[bool],
        raw_inputs: list[dict[str, Any]],
        subscriber_ids: list[str | None] | None = None,
    ) -> None:
        """Persist one request and its per-subscriber scores.

        Errors are swallowed and logged — a DB failure must never fail a
        prediction response.
        """
        ts = datetime.now(UTC).isoformat()
        ids = subscriber_ids if subscriber_ids is not None else [None] * len(probs)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO prediction_requests "
                    "(request_id, ts, model_name, n_subscribers, threshold, latency_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (request_id, ts, model_name, n_subscribers, threshold, latency_ms),
                )
                conn.executemany(
                    "INSERT INTO prediction_rows "
                    "(request_id, subscriber_idx, churn_probability,"
                    " churn_prediction, features_json, subscriber_id) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (request_id, i, float(p), int(pred), json.dumps(raw), sid)
                        for i, (p, pred, raw, sid) in enumerate(
                            zip(probs, predictions, raw_inputs, ids, strict=True)
                        )
                    ],
                )
        except Exception as exc:
            _log.error(f"PredictionLogger: failed to write request {request_id}: {exc}")

    def summary(self, since_hours: int = 24) -> dict[str, Any]:
        """All-time totals plus traffic statistics over the last `since_hours`."""
        since = (datetime.now(UTC) - timedelta(hours=since_hours)).isoformat()
        with self._connect() as conn:
            total_preds: int = conn.execute(
                "SELECT COUNT(*) FROM prediction_rows"
            ).fetchone()[0]
            total_requests: int = conn.execute(
                "SELECT COUNT(*) FROM prediction_requests"
            ).fetchone()[0]
            win_rows = conn.execute(
                "SELECT COUNT(*), AVG(r.churn_probability), AVG(r.churn_prediction) "
                "FROM prediction_rows r "
                "JOIN prediction_requests q ON r.request_id = q.request_id "
                "WHERE q.ts >= ?",
                (since,),
            ).fetchone()
            win_requests = conn.execute(
                "SELECT COUNT(*), AVG(latency_ms) FROM prediction_requests WHERE ts >= ?",
                (since,),
            ).fetchone()

        return {
            "total_predictions": total_preds,
            "total_requests": total_requests,
            "window_hours": since_hours,
            "predictions_in_window": int(win_rows[0]),
            "requests_in_window": int(win_requests[0]),
            "mean_churn_probability": _round(win_rows[1], 4),
            "churn_flag_rate": _round(win_rows[2], 4),
            "mean_latency_ms": _round(win_requests[1], 2),
        }

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent `limit` scored rows."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT q.request_id, q.ts, q.model_name, q.threshold, q.latency_ms,
                       r.subscriber_idx, r.subscriber_id, r.churn_probability,
                       r.churn_prediction
                FROM prediction_rows r
                JOIN prediction_requests q ON r.request_id = q.request_id
                ORDER BY q.ts DESC, r.subscriber_idx ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "request_id": row[0],
                "timestamp": row[1],
                "model_name": row[2],
                "threshold": row[3],
                "latency_ms": row[4],
                "subscriber_idx": row[5],
                "subscriber_id": row[6],
                "churn_probability": row[7],
                "churn_prediction": bool(row[8]),
            }
            for row in rows
        ]

    def fetch_window(
        self, since_hours: int, max_rows: int
    ) -> list[tuple[dict[str, Any], float]]:
        """Most recent (raw inputs, churn probability) pairs within the window."""
        since = (datetime.now(UTC) - timedelta(hours=since_hours)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT r.features_json, r.churn_probability FROM prediction_rows r "
                "JOIN prediction_requests q ON r.request_id = q.request_id "
                "WHERE q.ts >= ? ORDER BY r.id DESC LIMIT ?",
                (since, max_rows),
            ).fetchall()
        return [(json.loads(row[0]), float(row[1])) for row in rows]

    def prune(self, retention_days: int) -> int:
        """Delete requests (and their rows) older than `retention_days`. Returns rows removed."""
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        with self._connect() as conn:
            removed = conn.execute(
                "DELETE FROM prediction_rows WHERE request_id IN "
                "(SELECT request_id FROM prediction_requests WHERE ts < ?)",
                (cutoff,),
            ).rowcount
            conn.execute("DELETE FROM prediction_requests WHERE ts < ?", (cutoff,))
        return int(removed)
