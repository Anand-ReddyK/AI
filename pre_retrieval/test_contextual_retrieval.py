"""
Test Script: Demonstrates Anthropic Contextual Retrieval on orphan chunks.
Shows how adding situational context allows keyword (BM25) and semantic search
to match queries that were previously impossible to retrieve.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pre_retrieval.contextual_retrival import ContextualEnricher


# Document summary providing overall situational context
DOCUMENT_SUMMARY = """
Document: Acme Cloud Distributed Infrastructure & Database Reliability Guide (Platform v4.2).
This document covers production database replication tiers, Kafka event stream retention, 
caching strategies with Redis Cluster, and disaster recovery SLA specifications 
for payment processing and core microservices.
"""

# Sample chunks: Both are classic "orphan chunks" from engineering docs
# Chunk 1 describes failover and timeout parameters without naming the specific database (Aurora PostgreSQL).
# Chunk 2 describes compaction and consumer thresholds without naming the messaging engine (Apache Kafka).
SAMPLE_CHUNKS = [
    {
        "id": 1,
        "page_number": 14,
        "header_path": "High Availability > Replication & Failover",
        "text": (
            "For multi-region clusters, the heartbeat timeout is set to 2500ms with a maximum "
            "retry count of 3. If an unrecoverable failure occurs, the secondary replica promotes to primary "
            "within 30 seconds, maintaining an RPO of < 1 second. During split-brain mitigation, write consistency "
            "falls back to strict local quorum."
        )
    },
    {
        "id": 2,
        "page_number": 28,
        "header_path": "Event Ingestion Tier > Segment Compaction & Monitoring",
        "text": (
            "Log segments are automatically compacted when tombstone markers exceed 20% of the segment volume. "
            "Traffic spikes up to 50,000 msg/sec are buffered in memory before flushing to NVMe volumes. "
            "Consumer group lag exceeding 5 minutes triggers an automated P2 incident alert."
        )
    }
]



def test_single_chunk_context():
    print("=" * 70)
    print("TEST 1: CONTEXT GENERATION FOR AN ORPHAN CHUNK")
    print("=" * 70)

    enricher = ContextualEnricher()
    orphan_chunk = SAMPLE_CHUNKS[0]

    print(f"\n[Original Chunk Text]:\n\"{orphan_chunk['text']}\"\n")
    print("Notice: The text above contains latency and failover SLA metrics ('2500ms', 'RPO < 1s'), but does NOT")
    print("mention the database technology (Aurora / PostgreSQL) or service name.\n")

    # Generate situational context prefix
    context_prefix = enricher.generate_chunk_context(
        document_context_summary=DOCUMENT_SUMMARY,
        chunk_text=orphan_chunk["text"]
    )

    print(f"[Generated Situational Context Prefix]:\n-> {context_prefix}\n")

    enriched_text = f"{context_prefix}\n\n{orphan_chunk['text']}"
    print(f"[Final Enriched Chunk Stored in DB]:\n\"\"\"\n{enriched_text}\n\"\"\"\n")

    # Verification: Did the prefix add the missing technical keywords?
    target_keywords = ["aurora", "postgresql", "database", "replication"]
    print("[Keyword Visibility Check]:")
    for kw in target_keywords:
        in_original = kw in orphan_chunk["text"].lower()
        in_enriched = kw in enriched_text.lower()
        print(f"  Keyword '{kw}': in original = {in_original} | in enriched = {in_enriched}")


def test_batch_enrichment():
    print("\n" + "=" * 70)
    print("TEST 2: BATCH ENRICHMENT OF MULTIPLE CHUNKS")
    print("=" * 70)

    enricher = ContextualEnricher()
    enriched = enricher.enrich_chunks(
        chunks=SAMPLE_CHUNKS,
        document_context_summary=DOCUMENT_SUMMARY,
        max_chunks_to_process=2
    )

    for c in enriched:
        print(f"\n--- Chunk ID: {c['id']} (Page {c['page_number']}) ---")
        print(f"Context Prefix : {c.get('context_prefix')}")
        print(f"Enriched Text  : {c.get('text')[:180]}...\n")


if __name__ == "__main__":
    test_single_chunk_context()
    test_batch_enrichment()
