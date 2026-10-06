"""
Query Transformation Module:
1. Query Rewriting (de-contextualizes conversational questions)
2. HyDE (Hypothetical Document Embeddings for dense search)
3. Multi-Query Decomposition (splits complex questions into sub_queries)
"""

import os
import json
from typing import List, Optional, Dict
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "openai/gpt-oss-20b"


class QueryTransformer:
    def __init__(self, model: str = DEFAULT_MODEL):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = model
    
    def rewrite_query(
            self,
            query: str,
            chat_history: Optional[List[Dict[str, str]]] = None
        ) -> str:
        """
        Rewrites conversational or vague questions into standalone, search-optimized queries.
        Resolves pronouns ('it', 'the second drug', 'their policy') based on chat history.

        Args:
            query: The user's query
            chat_history: Optional list of previous turns in the conversation
        
        Returns:
            Rewritten query
        """

        history_context = ""
        if chat_history:
            formatted = []
            for msg in chat_history[-4:]: # Keep last 4 turns
                role = msg.get('role', "user").capitalize()
                content = msg.get("content", "")
                if content.strip(): # only include non-empty messages
                    formatted.append(f"{role}: {content}")
            history_context = "\n".join(formatted)
        
        prompt = f"""
        You are a seach query optimizer for a document retrieval system.
        Given the user's latest query and optional conversation history, 
        rewrite the query into a single, clear, keyword-rich standalone search query.

        Guidelines:
        - Resolve all pronouns ('it', 'they', 'this drug', 'their policy', ...etc) using the context.
        - Keep domain terminology, acronyms and specific numbers intact.
        - DO NOT answer the query, your job is to rewite the query for better results in a later stage by retrieval system (RAG).
        - return ONLY the rewitten search query string, No explanation about the query or any other text is needed.

        Conversation History:
        {history_context if history_context else 'None'}

        User Query:
        {query}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        return response.choices[0].message.content.strip().strip('"')
    
    # ? Don't we have to send some info about the original collection, like it's summary to help get better hypothetical answers?
    def generate_hyde_passage(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings):
        Generates a hypothetical answer passage. Embedding this hypothetical passage
        often retrives better dense matches than embedding the raw question.
        """

        prompt = f"""
        You are a document writer. Write a short, hypothetical excerpt (2-3 sentences)
        that answers the question below

        DO NOT write introductory phrases like 'Here is an excerpt'.
        Write strictly the factual-sounding excerpt itself.

        Question:
        {query}

        Hypothetical Excerpt (2-3 sentences):
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content.strip().strip('"')
    

    def decompose_query(self, query: str) -> List[str]:
        """
        Decomposes complex or comparision questions into 2-3 simpler sub-queries.
        Example: 'Compare Insurance benifits of Anthem and Blue Cross' ->
        ['Anthem insurance benifits', 'Blue Cross insurance benifits']
        """

        prompt = f"""
        Decompose the following user question into at most 5 distinct, simpler sub-queries (you can give only 2 or 3 sub-queries as well if it is enough to cover the original query).
        
        Each sub-query must be standalone, independent, and focused on retrieving facts to 
        answer part of the original question. Such that the final answer can be 
        synthesized from the results of the sub-queries without loss of information.

        If the query is already simple and cannot be split, return only the original query in the list.

        You must respond in valid JSON matching this exact structure:
        {{
          "queries": ["sub-query 1", "sub-query 2"]
        }}
        
        User Query:
        {query}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        try:
            content = response.choices[0].message.content.strip()
            data = json.loads(content)

            queries = data.get("queries", [])
            if isinstance(queries, list) and queries:
                return [str(q).strip() for q in queries if str(q).strip()]
            
            print(f"[WARNING] Invalid query list: {queries}. Falling back to original query.")
            return [query]
            
        except Exception as e:
            print(f"[WARNING] Query decomposition failed: {e}. Falling back to original query.")
            return [query]


        


        