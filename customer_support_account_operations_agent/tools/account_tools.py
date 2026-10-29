"""
Account Operations Tools:
Exposes tools for customer account lookup, refund processing, and API key rotation
"""
import sys
import uuid
from pathlib import Path
from langchain_core.tools import tool

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from customer_support_account_operations_agent.database.db import (
    get_account_by_id,
    execute_refund,
    update_api_key
)


@tool
def get_customer_account(account_id: str) -> str:
    """
    Retrieves the customer's account profile, tier, current balance, and active API key.
    Use this tool whenever a customer asks about their account details or before issuing refunds.
    """
    account = get_account_by_id(account_id)
    if not account:
        return f"Error: No account found for account_id: '{account_id}'"
    

    return (
        f"Account ID: {account['account_id']}\n"
        f"Name: {account['name']}\n"
        f"Email: {account['email']}\n"
        f"Tier: {account['tier']}\n"
        f"Current Balance: ${account['balance_usd']:.2f}\n"
        f"Active API Key: {account['active_api_key']}"
    )


@tool
def issue_refund(account_id: str, amount_usd: float, reason: str) -> str:
    """
    Issues a monetary refund to the specified customer account and updates the database.
    """
    # WARNING: Currently this has no ceiling or verification checks!
    account = get_account_by_id(account_id)
    if not account:
        return f"Error: Cannot issue refund. Account '{account_id}' does not exist."

    
    result = execute_refund(account_id, amount_usd, reason)
    return (
        f"Successfully processed refund of ${amount_usd:.2f} for account {account_id}. "
        f"Reason recorded: '{reason}'."
    )


@tool
def rotate_api_key(account_id: str) -> str:
    """
    Rotates the customer's active API key, invalidating the old key and generating a new token.
    """
    account = get_account_by_id(account_id)
    if not account:
        return f"Error: Cannot rotate key. Account '{account_id}' does not exist."
    
    new_token = f"sk_live_{account['tier'].lower()}_{uuid.uuid4().hex[:16]}"
    update_api_key(account_id, new_token)
    return f"API Key for account {account_id} has been rotated. New Key: {new_token}"