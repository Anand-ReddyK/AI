"""
SQLite Database manager for Customer Support & Account Operations.
Maintains presistent accounts and an immutable audit log of sensitive actions.
"""

import sqlite3
from pathlib import Path
from sqlite3 import Connection
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "support_accounts.db"


def get_db_connection() -> Connection:
    """Returns a connection to the SQLite database with raw factory enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database schem if tables do not exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Accounts Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                tier TEXT NOT NULL CHECK(tier IN ('Free', 'Pro', 'Enterprise')),
                balance_usd REAL NOT NULL DEFAULT 0.0,
                active_api_key TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Audit Log Table (Tracks refunds and key rotations)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id TEXT NOT NULL,
                action TEXT NOT NULL,
                amount_usd REAL DEFAULT 0.0,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (account_id) REFERENCES accounts(account_id)
            )
        """)

        conn.commit()


def get_account_by_id(account_id: str) -> dict[str, Any] | None:
    """Fetches an account record by account_id"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def execute_refund(account_id: str, amount_usd: float, reason: str) -> dict[str, Any]:
    """
    Credits/adds balance to the customer account and writes an audit log entry.
    NOTE: In 7.0 there are NO CEILINGS OR CHECKS on amount_usdd!
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE accounts SET balance_usd = balance_usd + ? WHERE account_id = ?",
            (amount_usd, account_id),
        )
        cursor.execute(
            "INSERT INTO audit_log (account_id, action, amount_usd, details) VALUES (?, ?, ?, ?)",
            (account_id, "REFUND_ISSUED", amount_usd, reason),
        )

        conn.commit()

    return {
        "status": "SUCCESS",
        "account_id": account_id,
        "refunded_usd": amount_usd,
        "reason": reason,
    }


def update_api_key(account_id: str, new_key: str) -> dict[str, Any]:
    """Updates customer's API key and logs the action"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE accounts SET active_api_key = ? WHERE account_id = ?",
            (new_key, account_id),
        )
        cursor.execute(
            "INSERT INTO audit_log (account_id, action, details) VALUES (?, ?, ?)",
            (account_id, "API_KEY_ROTATED", f"New key issued: {new_key[:8]}"),
        )

        conn.commit()

    return {"status": "SUCCESS", "account_id": account_id, "new_key": new_key}


def get_recent_audit_log(limit: int = 5) -> list[dict[str, Any]]:
    """Fetches recent audit log entries for inspection."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        )

        return [dict(r) for r in cursor.fetchall()]
