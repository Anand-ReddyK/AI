"""
Instead of dumping raw characters through `pypdf`, Strategy 3 uses a two-phase pipeline

[PDF Document]
      │
      ▼ (Phase 1: Layout-Aware Parsing via pymupdf4llm)
[Structured Markdown]
   • Tables converted to GFM tables: | Col1 | Col2 |
   • Headings converted to: # Header 1, ## Header 2
      │
      ▼ (Phase 2: Header-Aware Splitting)
[Structured Chunks]
   • Chunks split on header boundaries
   • Full header breadcrumbs stored in metadata ("Header 1 > Header 2")
   • Tables flagged with chunk_type: "table"
   • Page numbers tracked


Problems:
 - Not detecting the multiple page tables correctly (especialy when some kind a text comes at the end of a page where the page's table continues to the next page)
 - When a Heading section spills to another page then it is not being tracked and is being considred as "Root Document"
"""

from collections.abc import Callable
from typing import Optional
import os
import json
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 1. Configuration
CHUNK_SIZE = 800  # Slightly larger to allow complete tables and paragraphs to stay intact
OVERLAP = 100



def is_table_chunk(text: str) -> bool:
    """Detects if a chunk contains a Markdown table."""
    
    lines = text.strip().split("\n")
    pipe_lines = [line for line in lines if line.strip().startswith("|") and line.strip().endswith("|")]

    return len(pipe_lines) >= 2 and any("---" in line for line in pipe_lines)


def extract_and_chunk_structure_aware(
    pdf_path: str, 
    chunk_size: int = 800, 
    overlap: int = 100,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> list[dict]:
    """
    Extracts and chunks a PDF file using layout-aware parsing.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Maximum size of each chunk.
        overlap: Overlap between consecutive chunks.
        progress_callback: Optional callback function to report progress.

    Returns:
        List of chunk dictionaries.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    if progress_callback:
        progress_callback(0, 0, "Extracting layout-aware Markdown from PDF....")
    
    # Step 1: Extract PDF as page-by-page structured Markdown
    # pymupdf4llm understands bounding boxes, headers, and table structures
    # print("Extracting layout-aware Markdown from PDF....")
    page_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    total_pages = len(page_data)

    # Step 2: Define header hierarchy to track
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # Keep the header text inside the chunk so LLM sees it!
    )

    # Fallback splitter for sections that exceed chunk_size
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " "]
    )

    all_chunks = []
    global_chunk_id = 0

    for idx, item in enumerate(page_data):
        page_num = item["metadata"]["page_number"]
        page_md = item["text"]

        if progress_callback:
            progress_callback(idx + 1, total_pages, f"Processing page {page_num} of {total_pages}...")

        if not page_md.strip():
            continue

        # Split on Markdown headers first
        header_splits = markdown_splitter.split_text(page_md)

        for split in header_splits:
            content = split.page_content
            header_meta = split.metadata

            # Build readable breadcrumb string: "Header 1 > Header 2 > Header 3..."
            breadcrumb = " > ".join(header_meta.values()) if header_meta else "Root Document"

            # Check if this section contains a table
            has_table = is_table_chunk(content)
            
            # If it's a table, or if it fits within chunk_size, keep it intact!
            if has_table or len(content) <= chunk_size:
                all_chunks.append({
                    "chunk_id": global_chunk_id,
                    "page_number": page_num,
                    "header_path": breadcrumb,
                    "chunk_type": "table" if has_table else "text",
                    "char_count": len(content),
                    "text": content
                })
                global_chunk_id += 1
            else:
                # Sub-chunk large text sections that don't have tables
                sub_splits = fallback_splitter.split_text(content)
                for sub in sub_splits:
                    all_chunks.append({
                        "chunk_id": global_chunk_id,
                        "page_number": page_num,
                        "header_path": breadcrumb,
                        "chunk_type": "text",
                        "char_count": len(sub),
                        "text": sub
                    })
                    global_chunk_id += 1
    
    if progress_callback:
        progress_callback(total_pages, total_pages, f"Completed: {len(all_chunks)} chunks created from {total_pages} pages.")
    
    return all_chunks



if __name__ == "__main__":

    # 1. Configuration
    PDF_PATH = "data_sources/european-parliament-272-pages-rag.pdf"
    OUTPUT_JSON_PATH = "chunked_data/structure_aware/structure_aware_chunks-european-parliament-272-pages-rag.json"

    # 2. Ensure output folder exists
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    # 3. Run the chunker
    chunks = extract_and_chunk_structure_aware(PDF_PATH, CHUNK_SIZE, OVERLAP)

    # 4. Print Diagnostics
    print("=" * 60)
    print(f"STRUCTURE-AWARE REPORT: {os.path.basename(PDF_PATH)}")
    print("=" * 60)
    print(f"Total chunks created : {len(chunks)}")
    table_count = sum(1 for c in chunks if c["chunk_type"] == "table")
    print(f"Total tables detected: {table_count}")
    print("=" * 60)


    # 5. Inspection: Find and print a sample table chunk
    table_sample = next((c for c in chunks if c["chunk_type"] == "table"), None)
    if table_sample:
        print("\n--- [INSPECTION: Preserved Table Chunk with Metadata] ---")
        print(f"Chunk ID    : {table_sample['chunk_id']}")
        print(f"Page Number : {table_sample['page_number']}")
        print(f"Header Path : {table_sample['header_path']}")
        print(f"Type        : {table_sample['chunk_type']}")
        print(f"Content     :\n{'-'*50}\n{table_sample['text'][:600]}...\n{'-'*50}")


    # 6. Save to JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

    print(f"\n[SAVED] {len(chunks)} chunks saved to {OUTPUT_JSON_PATH}")