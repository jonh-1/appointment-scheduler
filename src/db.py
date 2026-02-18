import sqlite3
from pathlib import Path
from typing import Optional

database_path = Path(__file__).resolve().parent.parent / "data" / "scheduler.db"

appointments_schema = """
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

def get_db_path() -> Path:
    return database_path


def init_db() -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(database_path)
    try:
        conn.executescript(appointments_schema)
        conn.commit()
    finally:
        conn.close()


def check_for_conflicting_appointments(scheduled_at: str, doctor_name: str) -> bool:
    """Return True if the doctor already has an appointment at the given time."""
    conn = sqlite3.connect(database_path)
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
    conn = sqlite3.connect(database_path)
    try:
        conflict = check_for_conflicting_appointments(scheduled_at, doctor_name)
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
    conn = sqlite3.connect(database_path)
    try:
        return conn.execute("SELECT * FROM appointments").fetchall()
    finally:
        conn.close()


def select_appointments_by_patient(patient_name: str, future_only: bool = False) -> list[dict]:
    conn = sqlite3.connect(database_path)
    try:
        return conn.execute(f"SELECT * FROM appointments WHERE patient_name = ? {"AND scheduled_at > datetime('now')" if future_only else ''}", (patient_name,)).fetchall()
    finally:
        conn.close()
