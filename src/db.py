import sqlite3
from pathlib import Path
from typing import Optional

DATABASE_PATH = Path(__file__).resolve().parent.parent / "data" / "scheduler.db"

APPOINTMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    patient_name TEXT NOT NULL,
    doctor_name TEXT NOT NULL,
    scheduled_at TEXT NOT NULL,
    summary TEXT,
    appointment_notes TEXT
);
"""


def init_db() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        conn.executescript(APPOINTMENTS_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _check_for_conflicting_appointments(scheduled_at: str, doctor_name: str) -> bool:
    """Return True if the doctor already has an appointment at the given time."""
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE scheduled_at = ? AND doctor_name = ?",
            (scheduled_at, doctor_name),
        ).fetchone()
        return count > 0
    finally:
        conn.close()


def insert_appointment(
    patient_name: str,
    doctor_name: str,
    scheduled_at: str,
    summary: str,
    appointment_notes: str = "",
) -> tuple[Optional[dict], Optional[sqlite3.Error | ValueError]]:
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        conflict = _check_for_conflicting_appointments(scheduled_at, doctor_name)
        if conflict:
            return (None, ValueError("Conflicting appointment found, ask the user to schedule a different time."))

        cur = conn.execute(
            "INSERT INTO appointments (patient_name, doctor_name, scheduled_at, summary, appointment_notes) VALUES (?, ?, ?, ?, ?) RETURNING *",
            (patient_name, doctor_name, scheduled_at, summary, appointment_notes),
        )
        row = cur.fetchone()
        conn.commit()
        return (row[0], None)
    except sqlite3.Error as e:
        return (None, e)
    finally:
        conn.close()


def select_appointments() -> list[dict]:
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        return conn.execute("SELECT * FROM appointments").fetchall()
    finally:
        conn.close()


def select_appointments_by_patient(patient_name: str, future_only: bool = False) -> list[dict]:
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        return conn.execute(f"SELECT * FROM appointments WHERE patient_name = ? {"AND scheduled_at > datetime('now')" if future_only else ''}", (patient_name,)).fetchall()
    finally:
        conn.close()
