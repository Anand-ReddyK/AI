"""
After downloading the libraries in requirements.txt:
1. Run this app locally - python -m streamlit run app.py
2. Upload a PDF document in the sidebar
3. The document will be processed and indexed to a local Qdrant instance

This app is a RAG v1 - Structure-Aware Document Q&A Agent. It uses:
1. Structure-Aware chunking
2. Local Qdrant
3. Groq LLM
"""


import os
import re
import tempfile
import time
import streamlit as st

from strategies.structure_aware_chunking import extract_and_chunk_structure_aware
from index_to_qdrant import index_chunks_to_local_qdrant, DEFAULT_LOCAL_DB_PATH
from rag_engine import query_rag_stream


# Page Configuration
st.set_page_config(
    page_title="RAG v1 - Structure-Aware Document Q&A Agent",
    page_icon="📚",
    layout="wide"
)

# Helpers
def sanitize_collection_name(name: str) -> str:
    """Creates a valid Qdrant collectiion name from filename."""
    clean = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    return f"doc_{clean.lower()}"[:50]

# Session State Initialization
if "message" not in st.session_state:
    st.session_state.message = []
if "current_collection" not in st.session_state:
    st.session_state.current_collection = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}


# Sidebar: Document Ingestion & Parameters

with st.sidebar:
    st.title("⚙️ Document Ingestion")
    st.caption("Episode 3.5: Layout-Aware Chunking + Local Qdrant + Groq")

    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

    with st.expander("🛠️ Chunking Parameters", expanded=True):
        chunk_size = st.slider("Chunk Size (characters)", min_value=400, max_value=2000)
        overlap = st.slider("Chunk overlap (characters)", min_value=0, max_value=300, value=100, step=25)
        top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=10, value=4, step=1)
    
    process_btn = st.button("🚀 Process & Index Document", type="primary", use_container_width=True, disabled=uploaded_file is None)

    if process_btn and uploaded_file is not None:
        collection_name = sanitize_collection_name(uploaded_file.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_pdf_path = tmp_file.name
        
        try:
            start_time = time.time()
            with st.status("Processsing PDF....", expanded=True) as status_box:
                progress_bar = st.progress(0)
                status_text = st.empty()


                # Phase 1: Structure-Aware chunking
                def chunk_callback(current, total, message):
                    if total > 0:
                        progress_bar.progress(int(current/total) * 50)
                    status_box.text(f"[Phase 1/2] {message}")
                
                chunks = extract_and_chunk_structure_aware(
                    pdf_path=tmp_pdf_path,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    progress_callback=chunk_callback
                )


                # Phase 2: Embedding & Local Vector Store
                def embed_collback(current, total, message):
                    if total > 0:
                        progress_bar.progress(50 + int(current/total) * 50)
                    status_text.text(f"[Phase 2/2] {message}")
                
                indexed_count = index_chunks_to_local_qdrant(
                    collection_name=collection_name,
                    chunks=chunks,
                    storage_path=DEFAULT_LOCAL_DB_PATH,
                    recreate_collection=True,
                    progress_callback=embed_collback
                )

                elapsed = time.time() - start_time
                status_box.update(label=f"✅ Indexed {indexed_count} chunks in {elapsed:.2f}s!", state="complete", expanded=False)
            
            table_count = sum(1 for c in chunks if c.get("chunk_type") == "table")
            st.session_state.current_collection = collection_name
            st.session_state.doc_stats = {
                "filename": uploaded_file.name,
                "total_chunks": indexed_count,
                "tables_detected": table_count,
                "time_sec": elapsed
            }
            # Clear chat when new document is loaded
            st.session_state.messages = []
            st.rerun()
        except Exception as e:
            st.error(f"Error processing documents: {e}")
        finally:
            if os.path.exists(tmp_pdf_path):
                os.remove(tmp_pdf_path)
    
    if st.session_state.current_collection and st.session_state.doc_stats:
        st.divider()
        st.subheader("📊 Active Document")
        stats = st.session_state.doc_stats
        st.markdown(f"**File:** `{stats['filename']}`")
        st.metric("Total Chunks", stats["total_chunks"])
        st.metric("Tables Preserved", stats["tables_detected"])
        st.caption(f"Indexed in {stats['time_sec']:.2f} seconds")


# Main Panel: Conversation Q&A

st.title("📖 Document Q&A Assistant")
st.markdown("Ask anything about your document")

if not st.session_state.current_collection:
    st.info("👆 Upload a PDF in the sidebar and press **'Process & Index Document'** to start.")
else:
    # Display Chat History

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 View Retrieved Sources & Citations"):
                    for i, s in enumerate(msg["sources"]):
                        payload = s.get("payload", {})
                        score = s.get("score", 0.0)
                        page = payload.get("page_number", "N/A")
                        header = payload.get("header_path", "Root")
                        chunk_type = payload.get("chunk_type", "text")
                        
                        badge = "📊 TABLE" if chunk_type == "table" else "📝 TEXT"
                        st.markdown(f"**Source {i+1}** | {badge} | **Page {page}** | `Score: {score:.3f}`")
                        st.caption(f"Path: `{header}`")
                        st.text(payload.get("text", "")[:400] + ("..." if len(payload.get("text", "")) > 400 else ""))
                        st.divider()
    # Chat Input
    if user_query := st.chat_input("Ask a question about this document..."):
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        # Generate Assistant Response
        with st.chat_message("assistant"):
            try:
                stream_gen, retrieved_chunks = query_rag_stream(
                    collection_name=st.session_state.current_collection,
                    query=user_query,
                    top_k=top_k,
                    storage_path=DEFAULT_LOCAL_DB_PATH
                )
                
                # Stream the response in real time
                response_text = st.write_stream(stream_gen)
                # Show sources immediately under current response
                if retrieved_chunks:
                    with st.expander("🔍 View Retrieved Sources & Citations"):
                        for i, s in enumerate(retrieved_chunks):
                            payload = s.get("payload", {})
                            score = s.get("score", 0.0)
                            page = payload.get("page_number", "N/A")
                            header = payload.get("header_path", "Root")
                            chunk_type = payload.get("chunk_type", "text")
                            
                            badge = "📊 TABLE" if chunk_type == "table" else "📝 TEXT"
                            st.markdown(f"**Source {i+1}** | {badge} | **Page {page}** | `Score: {score:.3f}`")
                            st.caption(f"Path: `{header}`")
                            st.text(payload.get("text", "")[:400] + ("..." if len(payload.get("text", "")) > 400 else ""))
                            st.divider()
                # Save Assistant Message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": retrieved_chunks
                })
            except Exception as e:
                st.error(f"Error generating answer: {e}")

    