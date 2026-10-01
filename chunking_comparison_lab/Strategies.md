# **The 3 Strategies to Implement**

## Naive Fixed-Size Chunking (The Control Baseline)

**What it does:** Hard-slices the raw text every $N$ characters (e.g., 500 characters) with zero overlap and zero awareness of sentences, tables, or pages.

**Why build it:** Serves as your scientific benchmark. It demonstrates firsthand why basic string slicing breaks words, shreds numbers in financial tables, and produces zero metadata for source citations.

## Page-Aware Recursive Character Chunking (The Industry Workhorse)

**What it does:** Iterates page-by-page through the PDF, extracting text and applying hierarchical splitting (\n\n paragraphs $\to$ \n sentences $\to$ spaces) with a sliding overlap window (e.g., 800 characters size, 150 characters overlap). It attaches the basic page_number to each chunk's metadata.

**Why build it:** This is the default in 90% of production tutorials (e.g., LangChain's RecursiveCharacterTextSplitter). You will see how clean narrative paragraphs become, but you'll also discover where it fails when a table spans across a chunk boundary or when context from a previous section header is lost.

## Structure-Aware Markdown & Table Chunking (The Production Standard)

**What it does:** Uses a layout-aware PDF parser (such as pymupdf4llm) to convert the PDF into structured Markdown. It preserves tables as unified Markdown blocks `| col1 | col2 |`, detects header hierarchy (`#`, `##`, `###`), and splits on section boundaries while attaching rich metadata:

- `page_number`
- `header_path` (e.g. Chapter 3 > Section 2)
- `chunk_type` (table vs text)

**Why build it:** This directly solves your requirements for small and large tables, makes 272-page PDFs searchable without losing section context, and provides the exact breadcrumb data needed for citations in your future "Talk to PDF" app.


# **How to Run Experiment Across PDFs**

Once the diagnostic script are ready:

- **Test 1 (6-Page PDF):** Run it on the short document to verify chunk counts, basic boundary transitions, and metadata tags.
- **Test 2 (Table Stress-Test):** Pick a page with a small table and a large multi-row table. Inspect how Strategy 1, Strategy 2, and Strategy 3 formatted and stored that table.
- **Test 3 (272-Page PDF):** Observe the memory, processing time, and whether Strategy 3 successfully attaches chapter/section breadcrumbs to chunks on page 150+ where Strategy 2 only knows the raw page number.