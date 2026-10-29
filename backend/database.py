"""SQLite Database and Data Isolation Layer for CreditPulse.
Guarantees strict per-user isolation: User A can NEVER view User B's cases or CSV files.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "creditpulse_app.db"

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def hash_pw(pw: str) -> str:
    return hashlib.sha256(f"cp_salt_{pw}".encode("utf-8")).hexdigest()

def log_audit_event(username: str, action: str, details: str = "", user_id: Optional[int] = None):
    with get_db() as conn:
        conn.execute("""
        INSERT INTO audit_events (user_id, username, action, details)
        VALUES (?, ?, ?, ?)
        """, (user_id, username, action, details))
        conn.commit()

def get_recent_audit_events(limit: int = 50) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
        SELECT id, user_id, username, action, details, created_at
        FROM audit_events
        ORDER BY id DESC
        LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

def get_loan_review_state(loan_id: int) -> dict:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM loan_reviews WHERE loan_id = ?", (loan_id,)).fetchone()
        if row:
            return dict(row)
        return {"loan_id": loan_id, "status": "normal", "note": "", "updated_at": None, "updated_by": None}

def set_loan_review_state(loan_id: int, status: str, note: str, username: str):
    with get_db() as conn:
        conn.execute("""
        INSERT INTO loan_reviews (loan_id, status, note, updated_at, updated_by)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
        ON CONFLICT(loan_id) DO UPDATE SET
            status = excluded.status,
            note = excluded.note,
            updated_at = CURRENT_TIMESTAMP,
            updated_by = excluded.updated_by
        """, (loan_id, status, note, username))
        conn.commit()

def get_human_flag(loan_id: int) -> dict:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM human_loan_flags WHERE loan_id = ?", (loan_id,)).fetchone()
        if row:
            d = dict(row)
            d["is_flagged"] = bool(d["is_flagged"])
            return d
        return {
            "loan_id": loan_id,
            "is_flagged": False,
            "severity": "None",
            "reason": "",
            "note": "",
            "flagged_by": "",
            "created_at": None,
            "updated_at": None,
            "resolved_at": None,
            "resolved_by": None
        }

def set_human_flag(loan_id: int, severity: str, reason: str, note: str, flagged_by: str) -> dict:
    with get_db() as conn:
        conn.execute("""
        INSERT INTO human_loan_flags (loan_id, is_flagged, severity, reason, note, flagged_by, updated_at, resolved_at, resolved_by)
        VALUES (?, 1, ?, ?, ?, ?, CURRENT_TIMESTAMP, NULL, NULL)
        ON CONFLICT(loan_id) DO UPDATE SET
            is_flagged = 1,
            severity = excluded.severity,
            reason = excluded.reason,
            note = excluded.note,
            flagged_by = excluded.flagged_by,
            updated_at = CURRENT_TIMESTAMP,
            resolved_at = NULL,
            resolved_by = NULL
        """, (loan_id, severity, reason, note, flagged_by))
        conn.commit()
    return get_human_flag(loan_id)

def resolve_human_flag(loan_id: int, resolved_by: str, resolution_note: str = "") -> dict:
    with get_db() as conn:
        conn.execute("""
        UPDATE human_loan_flags SET
            is_flagged = 0,
            resolved_at = CURRENT_TIMESTAMP,
            resolved_by = ?,
            note = note || ' [RESOLVED: ' || ? || ']'
        WHERE loan_id = ?
        """, (resolved_by, resolution_note, loan_id))
        conn.commit()
    return get_human_flag(loan_id)

def get_all_human_flags() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
        SELECT * FROM human_loan_flags ORDER BY is_flagged DESC, updated_at DESC
        """).fetchall()
        return [dict(r) for r in rows]

def add_simulated_stream_record(record: dict) -> dict:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO simulated_stream_records (
            record_id, loan_id, account_id, borrower_name, event_type, amount, balance,
            model_version, predicted_risk, risk_category, model_flag, human_flag, processing_status, features_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get("record_id", 101),
            record["loan_id"],
            record["account_id"],
            record.get("borrower_name", f"Facility #{record['loan_id']}"),
            record.get("event_type", "Transaction Debited"),
            record.get("amount", 0.0),
            record.get("balance", 0.0),
            record.get("model_version", "M6_multitask_ens5"),
            record["predicted_risk"],
            record.get("risk_category", "Moderate"),
            1 if record["predicted_risk"] >= 0.0285 else 0,
            1 if record.get("human_flag") else 0,
            "Processed",
            json.dumps(record.get("features", {}))
        ))
        conn.commit()
        record_id = cur.lastrowid
        row = conn.execute("SELECT * FROM simulated_stream_records WHERE id = ?", (record_id,)).fetchone()
        return dict(row)

def get_simulated_stream_records(limit: int = 50) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
        SELECT * FROM simulated_stream_records ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

def reset_simulated_stream_records():
    with get_db() as conn:
        conn.execute("DELETE FROM simulated_stream_records")
        conn.commit()

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            full_name TEXT NOT NULL,
            department TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            case_name TEXT NOT NULL,
            dataset_filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            notes TEXT,
            total_facilities INTEGER NOT NULL,
            avg_risk_6m REAL NOT NULL,
            critical_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS case_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            loan_id INTEGER NOT NULL,
            borrower_name TEXT NOT NULL,
            sector TEXT NOT NULL,
            principal_amount REAL NOT NULL,
            dpd_status TEXT NOT NULL,
            risk_6m REAL NOT NULL,
            risk_tier TEXT NOT NULL,
            hazard_json TEXT NOT NULL,
            drivers_json TEXT NOT NULL,
            raw_row_json TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases (id) ON DELETE CASCADE
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS loan_reviews (
            loan_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'normal',
            note TEXT DEFAULT '',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_by TEXT DEFAULT 'analyst'
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS human_loan_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            loan_id INTEGER NOT NULL UNIQUE,
            is_flagged INTEGER NOT NULL DEFAULT 1,
            severity TEXT NOT NULL, -- 'Watch', 'Concern', 'Critical'
            reason TEXT NOT NULL,
            note TEXT DEFAULT '',
            flagged_by TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            resolved_by TEXT
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS simulated_stream_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id INTEGER NOT NULL,
            loan_id INTEGER NOT NULL,
            account_id INTEGER NOT NULL,
            borrower_name TEXT NOT NULL,
            event_type TEXT NOT NULL,
            amount REAL NOT NULL,
            balance REAL NOT NULL,
            model_version TEXT NOT NULL,
            predicted_risk REAL NOT NULL,
            risk_category TEXT NOT NULL,
            model_flag INTEGER NOT NULL,
            human_flag INTEGER NOT NULL DEFAULT 0,
            processing_status TEXT NOT NULL DEFAULT 'Processed',
            features_json TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Seed default demonstration users if not present
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users WHERE username = 'analyst'")
        if cur.fetchone()[0] == 0:
            conn.execute("""
            INSERT INTO users (username, email, password_hash, role, full_name, department)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                "analyst",
                "risk_desk@sovereignbank.com",
                hash_pw("analyst123"),
                "Standard Risk Analyst",
                "Dr. Alex Mercer",
                "Institutional Risk & Surveillance"
            ))
            
        cur.execute("SELECT COUNT(*) FROM users WHERE username = 'examiner'")
        if cur.fetchone()[0] == 0:
            conn.execute("""
            INSERT INTO users (username, email, password_hash, role, full_name, department)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                "examiner",
                "examiner@federalreserve.gov",
                hash_pw("examiner123"),
                "Senior Risk Examiner",
                "Sarah Lin, CFA",
                "Federal Supervisory & CECL Audit"
            ))
        
        # Seed demonstration human flags if none exist
        cur.execute("SELECT COUNT(*) FROM human_loan_flags")
        if cur.fetchone()[0] == 0:
            demo_flags = [
                (5316, 1, "Concern", "Cash-flow deficit & falling liquidity reserves", "Customer operating account had 2 overdraft days this quarter.", "Dr. Alex Mercer"),
                (5365, 1, "Critical", "Acute credit distress & multiple negative balance days", "Debt service installments exceeding 75% of inflow. Immediate intervention recommended.", "Sarah Lin, CFA"),
                (4959, 1, "Watch", "Customer sector volatility in fleet logistics", "Monitoring fuel surcharge impact on debt service capacity.", "Dr. Alex Mercer"),
            ]
            for lf in demo_flags:
                conn.execute("""
                INSERT INTO human_loan_flags (loan_id, is_flagged, severity, reason, note, flagged_by)
                VALUES (?, ?, ?, ?, ?, ?)
                """, lf)

        conn.commit()

init_db()
