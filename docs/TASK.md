Smart search sistem for a large e-commerce platform

example dataset: sample-products.json

data is multilingual, for different locations.

Short architercture description:
- user facing chatbot
- agent has search tools that it can query based on users input
- prompt should mention not to use original query, but based on the query it should generate proper search queries
- there should be also context for llm to understand what kind of query to make and what fields to used for filtering
    - make this flexible, extandable for late
    - this should be later a kind of rag on the product data, but also with some general knowledge about the products and how to search for them
- make everything modular, so later we could replace each module: llm pipeline, context ingestion, search tools, etc.

Technical details (POC):
- langchain as the main llm framework
    - so you need react agent
    - tools for search 
- simple embeddings search with numpy and cosine similarity
- bm25s as keyword search backup
- facets for filtering results by category, price, etc.
- embeddings-transformers: sentence-transformers/distiluse-base-multilingual-cased-v2
