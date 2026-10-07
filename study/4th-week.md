# Proposed Week 4 Execution Roadmap
You can break your Week 4 work into 5 focused milestones:

## Milestone 4.1: Implement Hybrid Search (BM25 + Dense Vectors)

- Add sparse BM25 indexing alongside dense embeddings in your Qdrant pipeline.
- Implement Reciprocal Rank Fusion (RRF) to combine ranking scores.
- Verify on acronyms/numbers (e.g. testing queries with specific clinical codes or table figures where dense search currently fails).

**The Problem**: Dense embeddings (all-MiniLM-L6-v2) represent semantic meaning well, but fail on exact part numbers, clinical codes, drugs, acronyms (e.g., RT-PCR, D-Dimer), and specific names.

**The Solution**: Combine semantic vector search with keyword-based BM25, fused via Reciprocal Rank Fusion (RRF).

**Architecture Recommendation**:
- Syllabus recommendation: Uses LangChain EnsembleRetriever.
- My Industry Recommendation: Do this natively in Qdrant! Qdrant supports dual vector spaces (dense + sparse via FastEmbed BM42 / BM25) and server-side fusion in a single query. Doing this natively with qdrant-client keeps your latency low, eliminates dependency bloat, and looks significantly more impressive on your resume than relying on generic framework wrappers.


## Milestone 4.2: Add Cross-Encoder Reranking

- Add a reranker step using sentence-transformers (CrossEncoder).
- Pipeline: Retrieve Top-20 via Hybrid -> Rerank to Top-4 -> Feed to Groq LLM.
- Measure retrieval precision before and after reranking.

**The Concept:**
- **Bi-encoders** (like MiniLM or OpenAI embeddings) encode query and chunks separately into vectors. They are fast for candidate retrieval (N dot products), but cannot compare token-to-token interactions between the query and text.
- **Cross-encoders** take `(query, chunk_text)` together through full transformer attention layers to compute an exact relevance score. They are slower, but far more accurate.

**Implementation Strategy:**
- Retrieve **Top-25 to 30** candidates cheaply via Hybrid Search.
- Pass those 25 candidates through a Cross-Encoder to re-score and select the Top-4 to 5 for generation.
- **Tools to use:** `cross-encoder/ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-base` via `sentence-transformers` (runs completely free and locally on your CPU/GPU), or the `Cohere Rerank API`.


## Milestone 4.3: Implement Contextual Prepending / Query Rewriting & Solve the Table Issue

- Resolve the table chunking dilemma from `ToDo.md`.
- Test prepending breadcrumbs/context to each chunk before embedding.

**Contextual Retrieval:** Anthropic published a major technique in late 2024: before embedding any chunk, run a lightweight prompt through a fast LLM (e.g., Groq) that generates a 50–80 token situational summary of where this chunk lives in the document, and prepend it to the chunk.
- **Example:** Instead of just "The company grew 14% year over year", the chunk becomes: "In Q3 2024 earnings report for ACME Corp, under the cloud services section: The company grew 14% year over year".
- This solves chunk isolation without requiring massive token overlaps.

**Query Rewriting / HyDE (Hypothetical Document Embeddings):**
- If a user asks a short or ambiguous question, prompt Groq to generate a hypothetical answer or decomposed sub-queries, then embed that to retrieve more relevant context.



## Milestone 4.4: Install & Run RAGAS Evaluation

- Install ragas and run an evaluation run on your golden dataset.
- Compute Faithfulness, Answer Relevance, Context Precision, and Context Recall.

**The 4 Key RAGAS Metrics:**
- **Faithfulness:** Does the answer rely only on the retrieved context (no hallucinations)?
- **Answer Relevance:** Does the answer directly address the user's question?
- **Context Precision:** Are the retrieved chunks actually relevant (signal vs. noise ratio)?
- **Context Recall:** Did retrieval catch all necessary facts to answer the question?


## Milestone 4.5: Upgrade `app.py` to RAG v2 & Document Results

- Update your Streamlit application to give users toggles for:
    - Search mode: Dense Only vs. Hybrid (Dense + BM25)
    - Reranking: On / Off
- Publish a comparison report or markdown table showing:
    - Baseline (RAG v1) vs. Enhanced (RAG v2) accuracy and recall.


# Week 4 Study Questions

## Episode 4.1: Hybrid Search (BM25 + Dense Vectors)

- **"Why not just use cosine similarity vector search?"**

- Dense embeddings capture conceptual semantic similarity, but suffer from the 'vocabulary mismatch problem' on specific domain identifiers, acronyms, and part numbers. Hybrid search with BM25 guarantees that exact keyword hits are retrieved, while semantic search catches conceptually relevant passages with zero keyword overlap.

- **"Why RRF instead of linear score combination?"**

- BM25 and Cosine Similarity operate on non-comparable scales—BM25 is unbounded $(0, \infty)$ while Cosine is $[-1, 1]$. Normalizing them requires min-max tuning that breaks when new documents arrive. RRF operates strictly on rank order, making it scale-invariant and robust.

- **"What is the standard value of $k$ in RRF?"**

- $k = 60$, established by Cormack et al. It prevents outlier top ranks from skewing the final result while balancing high-confidence consensus hits.

## Episode 4.2: Cross-Encoder Reranking

- **"What is the fundamental architectural difference between a bi-encoder and a cross-encoder?"**

    - A bi-encoder embeds queries and documents independently into vectors, only comparing them via cosine similarity at the end. A cross-encoder feeds both query and document into the transformer simultaneously, allowing every query token to attend to every document token. Cross-encoders are far more accurate but have an $O(N)$ inference cost per query.

- **"Why can't we use a cross-encoder directly across all our documents?"**

    - If you have 100,000 chunks, passing all 100,000 pairs through a BERT-scale model would take several minutes of GPU time per query. Instead, we use a two-stage retrieval pipeline: bi-encoder/hybrid retrieves the top 20–30 candidates in milliseconds, and the cross-encoder reranks only those 20.

- **"When is reranking worth the extra latency?"**

    - Whenever answer precision and faithfulness matter more than sub-50ms latency—such as enterprise Q&A, legal, or medical search. Adding a 25ms local reranking step can cut hallucinations significantly by ensuring the LLM's top-1 context chunk actually answers the question.

## Milestone 4.3: Implement Contextual Prepending / Query Rewriting & Solve the Table Issue

- **"What is Anthropic's Contextual Retrieval and why does it outperform standard chunking?"**

    - Standard chunking strips context away from passages (the 'orphan chunk' problem). Contextual Retrieval uses an LLM at index time to generate 50–100 tokens of situational context (document title, subject, chapter) and prepends it to each chunk before embedding and BM25 indexing. Anthropic showed this cuts retrieval failure rates by up to 49%.

- **"What is HyDE and when would you use it?"**

    - Hypothetical Document Embeddings (HyDE) prompts an LLM to hallucinate a plausible answer passage to a query, and we embed that passage instead of the question. It helps because answers reside closer to other answers in dense embedding vector space than questions do. It is especially useful for high-level, conceptual, or sparse-keyword queries.

- **"How do you handle multi-part user questions in a RAG pipeline?"**

    - We use Query Decomposition: an LLM breaks a complex comparison question into 2 or 3 independent sub-queries. We retrieve candidates for each sub-query independently, deduplicate the pool, and then use a Cross-Encoder reranker against the original user question to find the best overall context.


## Milestone 4.4: Install & Run RAGAS Evaluation

- **"What is the difference between Context Precision and Context Recall?"**

    - Context Recall measures completeness: did our retrieval capture all facts present in the ground truth? Context Precision measures signal-to-noise: are the chunks containing those facts ranked near the very top of the retrieved list rather than diluted by irrelevant text.

- **"How does RAGAS compute Faithfulness without human labels?"**

    - RAGAS uses an LLM judge to extract all factual claims from the generated answer, then performs automated premise-entailment checks to verify if each claim is directly supported by the retrieved context. Faithfulness is the ratio of supported claims over total claims.

- **"Why is measuring Faithfulness better than just calculating BLEU or ROUGE?"**

    - BLEU and ROUGE only compare n-gram word overlaps against reference text. They cannot detect factual contradictions or subtle hallucinations (e.g. changing '10mg' to '100mg' has high ROUGE overlap but is completely unfaithful).