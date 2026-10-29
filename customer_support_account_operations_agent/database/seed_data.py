"""
Seeds mock accounts across Free, Pro, and Enterprise tiers for testing.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from customer_support_account_operations_agent.database.db import get_db_connection, init_db

SAMPLE_ACCOUNTS = [
    {
        "account_id": "acc_free_101",
        "name": "Alice Green",
        "email": "alice.green@startup.io",
        "tier": "Free",
        "balance_usd": 0.0,
        "active_api_key": "sk_live_free_1019283849102"
    },
    {
        "account_id": "acc_pro_202",
        "name": "Bob Vance (Vance Refrigeration)",
        "email": "bob@vancerefrig.com",
        "tier": "Pro",
        "balance_usd": 240.0,
        "active_api_key": "sk_live_pro_2028374619284"
    },
    {
        "account_id": "acc_enterprise_303",
        "name": "Global Logistics Corp",
        "email": "procurement@globallogistics.com",
        "tier": "Enterprise",
        "balance_usd": 5400.0,
        "active_api_key": "sk_live_ent_3039182736451"
    }
]


def seed():
    """Initializes DB and inserts or replaces default mock accounts."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for acc in SAMPLE_ACCOUNTS:
            cursor.execute("""
                INSERT OR REPLACE INTO accounts (account_id, name, email, tier, balance_usd, active_api_key)
                VALUES (:account_id, :name, :email, :tier, :balance_usd, :active_api_key)
            """, acc)
        conn.commit()
    print("[Database] Seeded 3 mock accounts into SQLite successfully.")


if __name__ == "__main__":
    seed()
