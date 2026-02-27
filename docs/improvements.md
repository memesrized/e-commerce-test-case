The following structure of the search step:
- sim search works as is
- facets
    - LLM based step
        - it recieves all the facets from the dataset and determines which ones to use for filtering based on the user query
        - also this should be extendable to the following logic:
            - give LLM not the full list, but the most relevant based on something else, this is todo basically, not for current impelementation
- BM25
    - LLM based step
        - it recieves context from context provider
            - now add there some meaningful information from the dataset, categories or short summary of the products (not one by one, but in general to make it more realistic even though it's a demo)
        - also it recieves locality information, so it can determine which market to search for
            - determine language
                - e.g. so if it's FR then it should be 2 queries 1 in english and 1 in french, but if it's US then only english
        - it recieves the original query and determines the keywords to use for the search, this is also extendable to the following logic:

Chatbot:
- guardrails for inappropriate output with stopwords
    - ensure the LLM doesn't generate responses that are off-brand, offensive, overly clinical, or inappropriately explicit?
    - bot don't overthink, this is still a demo

Make a backend version of the search system:
- put it in a separate folder like "backend" or "api"
- and implement logic starting from nlq
- use fastapi
    - put there only retrieval logic with llms
- streamlit for ui chat agent
    - user facing part
- docker compose for running the app and backend together
