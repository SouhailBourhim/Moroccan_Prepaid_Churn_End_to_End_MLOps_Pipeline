"""SQLite-backed prediction logger for the churn API.

Writes one row to `prediction_requests` and one row per subscriber to
`prediction_rows` for every /predict call. Designed for local development
and staging — swap for a cloud sink (BigQuery, S3 parquet) in production.

WAL journal mode is enabled so multiple uvicorn workers can write concurrently
without locking errors.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator

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
    features_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_rows_request ON prediction_rows (request_id);
CREATE INDEX IF NOT EXISTS idx_requests_ts  ON prediction_requests (ts);
"""


class PredictionLogger:
    """Thread-safe SQLite prediction logger."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_CREATE_SQL)

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
    ) -> None:
        """Persist one request and its per-subscriber scores.

        Errors are swallowed and logged — a DB failure must never fail a
        prediction response.
        """
        ts = datetime.now(timezone.utc).isoformat()
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
                    "(request_id, subscriber_idx, churn_probability, churn_prediction, features_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (request_id, i, float(p), int(pred), json.dumps(raw))
                        for i, (p, pred, raw) in enumerate(
                            zip(probs, predictions, raw_inputs)
                        )
                    ],
                )
        except Exception as exc:
            _log.error(f"PredictionLogger: failed to write request {request_id}: {exc}")

    def summary(self, since_hours: int = 24) -> dict[str, Any]:
        """Return aggregate stats across all logged predictions."""
        since = (
            datetime.now(timezone.utc) - timedelta(hours=since_hours)
        ).isoformat()
        with self._connect() as conn:
            total_preds: int = conn.execute(
                "SELECT COUNT(*) FROM prediction_rows"
            ).fetchone()[0]
            total_requests: int = conn.execute(
                "SELECT COUNT(*) FROM prediction_requests"
            ).fetchone()[0]
            recent_preds: int = conn.execute(
                "SELECT COUNT(*) FROM prediction_rows r "
                "JOIN prediction_requests q ON r.request_id = q.request_id "
                "WHERE q.ts >= ?",
                (since,),
            ).fetchone()[0]
            mean_prob_row = conn.execute(
                "SELECT AVG(churn_probability) FROM prediction_rows"
            ).fetchone()[0]
            flag_rate_row = conn.execute(
                "SELECT AVG(churn_prediction) FROM prediction_rows"
            ).fetchone()[0]
            mean_lat_row = conn.execute(
                "SELECT AVG(latency_ms) FROM prediction_requests"
            ).fetchone()[0]

        return {
            "total_predictions": total_preds,
            "total_requests": total_requests,
            f"predictions_last_{since_hours}h": recent_preds,
            "mean_churn_probability": (
                round(float(mean_prob_row), 4) if mean_prob_row is not None else None
            ),
            "churn_flag_rate": (
                round(float(flag_rate_row), 4) if flag_rate_row is not None else None
            ),
            "mean_latency_ms": (
                round(float(mean_lat_row), 2) if mean_lat_row is not None else None
            ),
        }

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent `limit` scored rows."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT q.request_id, q.ts, q.model_name, q.threshold, q.latency_ms,
                       r.subscriber_idx, r.churn_probability, r.churn_prediction
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
                "churn_probability": row[6],
                "churn_prediction": bool(row[7]),
            }
            for row in rows
        ]
