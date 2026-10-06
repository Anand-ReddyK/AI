"""
Anthropic Contextual Retrieval:
Generate 50-80 tokens of situational context for each chunk based on the full document,
then prepend it to the chunk prior to vector and BM25 indexing.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_FAST_MODEL = "openai/gpt-oss-20b"


class ContextualEnricher:
    def __init__(self, model: str = DEFAULT_FAST_MODEL):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = model
    
    def generate_chunk_context(
            self,
            document_context_summary: str,
            chunk_text: str
        ) -> str:
        """
        Prompts an LLM to generate situational context for a chunk given the document context.
        Matches the prompt template published in Anthropic's Contextual Retrival paper.
        """

        prompt = f"""
        <document>
        {document_context_summary}
        </document>
        
        Here is the chunk we want to situate within the overall document:
        <document_chunk>
        {chunk_text}
        </document_chunk>
        
        Please write a short, succint context (1-2 sentences, max 50 words) to situate this
        chunk within the overall document for search retrival.
        Mention key document titles, or section identities if relevant.
        Answer ONLY with the succient context and nothing else.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            return response.choices[0].message.content.strip().strip('"')

        except Exception as e:
            print(f"[ContextualEnricher] failed to generate context: {e}")
            return ""
    

    def enrich_chunks(
            self,
            chunks: List[Dict[str, Any]],
            document_context_summary: str,
            max_chunks_to_process: int = 20
        ) -> List[Dict[str, Any]]:
        """
        Enriches chunks by prepending generated situational context to 'text'/
        Preserves original text in payload['original_text'].
        """
        enriched_chunks = []
        limit = min(len(chunks), max_chunks_to_process)
        print(f"[ContextualEnricher] Generating contextual prefixes for {limit} chunks...\n" )

        for i in range(limit):
            chunk = dict(chunks[i])
            original_text = chunk.get("text", "")

            context_prefix = self.generate_chunk_context(
                document_context_summary=document_context_summary,
                chunk_text=original_text
            )

            if context_prefix:
                enriched_text = f"{context_prefix}\n\n{original_text}"
            else:
                enriched_text = original_text
            
            chunk["original_text"] = original_text
            chunk["context_prefix"] = context_prefix
            chunk["text"] = enriched_text # This enriched text is what gets embedded and BM25 indexed
            
            enriched_chunks.append(chunk)

            if (i + 1) % 5 == 0:
                print(f"    Processed {i + 1}/{limit} chunks...")
        
        return enriched_chunks

        
