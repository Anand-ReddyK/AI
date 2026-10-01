"""
Central configuration for Qdrant collection names, chunk data paths, and evaluation settings.
"""
from pathlib import Path

# Base directory for chunking_comparison_lab
LAB_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Qdrant Collection Names (Single Source of Truth)
# ---------------------------------------------------------------------------
COLL_COVID_6PAGE_NAIVE = "covid_6page_naive"
COLL_COVID_6PAGE_RECURSIVE = "covid_6page_recursive"
COLL_COVID_6PAGE_STRUCTURE = "covid_6page_structure"

# ---------------------------------------------------------------------------
# Strategy Files Mapping (Used by index_to_qdrant.py)
# ---------------------------------------------------------------------------
STRATEGY_COLLECTION_TO_CHUNKED_FILE_COVID_6PAGE = {
    COLL_COVID_6PAGE_NAIVE: str(LAB_DIR / "chunked_data/naive/naive_chunks-covid-6-page-rag.json"),
    COLL_COVID_6PAGE_RECURSIVE: str(LAB_DIR / "chunked_data/recursive/recursive_chunks-covid-6-page-rag.json"),
    COLL_COVID_6PAGE_STRUCTURE: str(LAB_DIR / "chunked_data/structure_aware/structure_aware_chunks-covid-6-page-rag.json"),
}

# ---------------------------------------------------------------------------
# Strategies Mapping for Evaluation (Used by evaluate_retrieval.py)
# Label -> Qdrant Collection Name
# ---------------------------------------------------------------------------
STRATEGIES_TO_COLLECTIONS_COVID_6PAGE = {
    "Naive (Fixed Size)": COLL_COVID_6PAGE_NAIVE,
    "Recursive (Page-Aware)": COLL_COVID_6PAGE_RECURSIVE,
    "Structure-Aware": COLL_COVID_6PAGE_STRUCTURE,
}

# ---------------------------------------------------------------------------
# Golden Dataset Path
# ---------------------------------------------------------------------------
GOLDEN_DATASET_COVID_6PAGE = str(LAB_DIR / "golden_dataset/covid_6page.json")
