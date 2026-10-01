"""
Instead of blindly slicing at character index 500, we uses a hierarchy of separators in order

Step 1 (\n\n): It first tries to split by double newlines (paragraphs). 
    If a paragraph is under 500 characters, it keeps it intact as a complete chunk!

Step 2 (\n): If a paragraph is larger than 500 characters, it recurses down to 
    single newlines (sentences or lines).

Step 3 (" "): If a line is still larger than 500 characters, it recurses down to 
    spaces (word boundaries).

Step 4 (""): Only if a single word is larger than 500 characters 
    (e.g., a massive URL or base64 string) will it chop mid-word as a last resort.
"""

import os
import json
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Configuration (Matching Strategy Naive chuncking for a fair test)
PDF_PATH = "data_sources/european-parliament-272-pages-rag.pdf"
OUTPUT_JSON_PATH = "chunked_data/recursive/recursive_chunks-european-parliament-272-pages-rag.json"

CHUNK_SIZE = 500
OVERLAP = 100

def extract_and_chunk_by_page(pdf_path: str, chunck_size: int, overlap: int) -> list[dict]:
    """
    Extracts text page-by-page and applies RecursiveCharacterTextSplitter.
    Attaches page_number metadata to every chunk.
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    # Initialize the recursive splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunck_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " "],
        length_function=len,
    )
    
    # Read PDF
    reader = PdfReader(pdf_path)
    all_chunks = []
    global_chunk_id = 0

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_text = page.extract_text()

        if not page_text or not page_text.strip():
            continue

        # Split this page's text recursively
        page_splits = text_splitter.split_text(page_text)

        for spilt in page_splits:
            all_chunks.append({
                "chunk_id": global_chunk_id,
                "page_number": page_num,
                "char_count": len(spilt),
                "text": spilt
            })
            global_chunk_id += 1

    return all_chunks


# 2. Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)


# 3. Run the chunker
chunks = extract_and_chunk_by_page(PDF_PATH, CHUNK_SIZE, OVERLAP)

# 4. Print Diagnostics
print("=" * 60)
print(f"RECURSIVE CHUNKING REPORT: {os.path.basename(PDF_PATH)}")
print("=" * 60)
print(f"Total chunks created : {len(chunks)}")
print(f"Configured size      : {CHUNK_SIZE} characters")
print(f"Configured overlap   : {OVERLAP} characters")
print("=" * 60)


# 5. Inspect Boundary Splits (Notice how it breaks at spaces/newlines now)
print("\n--- [INSPECTION 1: The Boundary Transition] ---")
chunk_0 = chunks[0]
chunk_1 = chunks[1]
print(f"[CHUNK {chunk_0['chunk_id']} (Page {chunk_0['page_number']}) LAST 80 CHARS]:")
print(f"...{repr(chunk_0['text'][-80:])}")
print(f"\n[CHUNK {chunk_1['chunk_id']} (Page {chunk_1['page_number']}) FIRST 80 CHARS]:")
print(f"{repr(chunk_1['text'][:80])}...")
print("\n--- [INSPECTION 2: Sample Chunk with Metadata] ---")
sample = chunks[min(2, len(chunks)-1)]
print(f"Chunk ID    : {sample['chunk_id']}")
print(f"Page Number : {sample['page_number']}")
print(f"Length      : {sample['char_count']} chars")
print(f"Content     :\n{'-'*40}\n{sample['text']}\n{'-'*40}")


# 6. Save chunks to JSON file
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=4, ensure_ascii=False)
print(f"\n[SAVED] {len(chunks)} chunks saved to {OUTPUT_JSON_PATH}")