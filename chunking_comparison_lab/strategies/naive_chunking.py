import os
from pypdf import PdfReader
import json

PDF_PATH = "data_sources/european-parliament-272-pages-rag.pdf"
OUTPUT_JSON_PATH = "chunked_data/naive/naive_chunks-european-parliament-272-pages-rag.json"

CHUNK_SIZE = 500 # Number of characters per chunk
OVERLAP = 100    # Number of characters to overlap between chunks

def extract_raw_text(pdf_path: str) -> str:
    """Extracts raw text from all pages of a PDF file"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    return full_text


def naive_fixed_chuncking(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """Chunks text into fixed-size chunks with overlap"""
    
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")
    
    chunks = []
    total_length = len(text)
    step = chunk_size - overlap

    for i in range(0, total_length, step):
        chunk_text = text[i : i + chunk_size]

        chunks.append({
            "chunk_id": len(chunks),
            "start_char": i,
            "end_char": min(i+chunk_size, total_length),
            "char_count": len(chunk_text),
            "text": chunk_text,
        })

        if i + chunk_size >= total_length:
            break

    return chunks


# 2. Run the chunker
raw_text = extract_raw_text(PDF_PATH)
chunks = naive_fixed_chuncking(raw_text, CHUNK_SIZE, OVERLAP)
# 3. Print Diagnostics
print("=" * 60)
print(f"NAIVE FIXED-SIZE CHUNKING REPORT: {os.path.basename(PDF_PATH)}")
print("=" * 60)
print(f"Total raw characters : {len(raw_text)}")
print(f"Total chunks created : {len(chunks)}")
print(f"Configured size      : {CHUNK_SIZE} characters")
print("=" * 60)
# 4. Inspect Boundary Splits (Find mid-word / broken cuts)
print("\n--- [INSPECTION 1: The Boundary Cut Problem] ---")
print("Look at how Chunk 0 ends and Chunk 1 begins:\n")
chunk_0 = chunks[0]
chunk_1 = chunks[1]
print(f"[CHUNK {chunk_0['chunk_id']} LAST 80 CHARS]:")
print(f"...{repr(chunk_0['text'][-80:])}")
print(f"\n[CHUNK {chunk_1['chunk_id']} FIRST 80 CHARS]:")
print(f"{repr(chunk_1['text'][:80])}...")
print("\n--- [INSPECTION 2: Sample Raw Chunk (No metadata)] ---")
sample = chunks[min(2, len(chunks)-1)]
print(f"Chunk ID    : {sample['chunk_id']}")
print(f"Length      : {sample['char_count']} chars")
print(f"Metadata    : None (No page number, no chapter, no headers)")
print(f"Content     :\n{'-'*40}\n{sample['text']}\n{'-'*40}")

# 5. Save chunks to JSON file
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=4, ensure_ascii=False)

print(f"\n[SAVED] {len(chunks)} chunks saved to {OUTPUT_JSON_PATH}")
