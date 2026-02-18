import sqlite3
from pathlib import Path
from typing import Optional

database_path = Path(__file__).resolve().parent.parent / "data" / "scheduler.db"

appointments_schema = """
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    patient_name TEXT NOT NULL,
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


def insert_appointment(patient_name: str, scheduled_at: str, summary: str, appointment_notes: str = "") -> tuple[Optional[dict], Optional[sqlite3.Error]]:
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "INSERT INTO appointments (patient_name, scheduled_at, summary, appointment_notes) VALUES (?, ?, ?, ?) RETURNING *",
            (patient_name, scheduled_at, summary, appointment_notes),
        )
        row = cur.fetchone()
        conn.commit()
        return (row, None)
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
