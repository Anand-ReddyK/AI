"""
Policy Ingestion Pipeline:
Reads authentic HubSpot ToS PDF, chunks text, and indexes into Qdrant collection 'billing_policies'
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import pymupdf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chunking_comparison_lab.index_to_qdrant import (
    index_chunks_to_local_qdrant,
    DEFAULT_LOCAL_DB_PATH
)
from chunking_comparison_lab.strategies.structure_aware_chunking import extract_and_chunk_structure_aware


POLICY_PDF_PATH = Path(__file__).resolve().parent.parent / "policies" / "HubSpot ToS April 14, 2026.pdf"
COLLECTION_NAME = "billing_policies"

def ingest():
    """Ingests policy PDF using structure-aware chunking and indexess into Qdrant."""
    if not POLICY_PDF_PATH.exists():
        raise FileNotFoundError(f"Policy PDF not found at {POLICY_PDF_PATH}")
    
    print(f"[Ingest] Running structure-aware chunking on '{POLICY_PDF_PATH.name}'...")

    raw_chunks = extract_and_chunk_structure_aware(
        pdf_path=str(POLICY_PDF_PATH),
        chunk_size=800,
        overlap=100
    )

    # Format chunks for Qdrant indexing
    formatted_chunks = []
    for c in raw_chunks:
        formatted_chunks.append({
            "chunk_id": f"hubspot_tos_{c["chunk_id"]}",
            "text": c["text"],
            "metadata": {
                "source": POLICY_PDF_PATH.name,
                "page": c.get("page_number", -1),
                "header_path": c.get("header_path", "Root"),
                "chunk_type": c.get("chunk_type", "text"),
                "char_count": c.get("char_count", 0),
            }
        })
    
    print(f"[Ingest] Created {len(formatted_chunks)} structure-aware chunks.")
    print(f"[Ingest] Indexing into Qdrant collection '{COLLECTION_NAME}'...")

    count = index_chunks_to_local_qdrant(
        collection_name=COLLECTION_NAME,
        chunks=formatted_chunks,
        storage_path=DEFAULT_LOCAL_DB_PATH,
        recreate_collection=True
    )
    print(f"[Ingest] Successfully indexed {count} structure-aware chunks into Qdrant.")

    OUTPUT_JSON_PATH = PROJECT_ROOT / "customer_support_account_operations_agent" / "chunked_data" / "hubspot_policies_chunks.json"
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(formatted_chunks, f, indent=4, ensure_ascii=False)

    print(f"\n[SAVED] {len(formatted_chunks)} chunks saved to {OUTPUT_JSON_PATH}")



if __name__ == "__main__":
    ingest()
    