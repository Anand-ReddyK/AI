In pre_retrieval/query_transformer.py:
- Here we have multiple prosesses to better optimize the user query to get relevent information
- We can build an agent to sit between the user and the RAG pipeline to better understand the user's need and intent, and accordingly apply the query transformer processes

In agents\langgraph_agent\react_interactive_graph.py:
- The LLM uses a tool called ask_human to talk to the user, but it sometimes forgets and just provides a response to ask a question. This reults in the program stopping.
- We need to find a solution, maybe find a different way to know if we have the final answer.