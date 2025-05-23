"""Prompts used to extract claims from reference input texts."""

CLAIM_EXTRACTION_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key claims that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each claim in the response should contain the following elements:
- "statement": A comprehensive claim statement that is relevant to the user query and include relevant contextual information (e.g., time, place, specific entities, events or examples). Claims should stand on their own and be independent of each other.
- "sources": A list of all source ids from the data tables that are relevant to the claim.
- "score": An integer score between 0-100 that indicates how important the claim is in answering the user's question. Claims that directly answer the query should be given higher scores than those that only provide background information. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.

The response should be JSON formatted as follows:
{{
    "claims": [
        {{"statement": "Statement of claim 1", "sources": [a list of relevant source ids], "score": importance score value (integer)}},
        {{"statement": "Statement of claim 2", "sources": [a list of relevant source ids], "score": importance score value (integer)}},
    ]
}}

---Data tables---

{context_data}


---Goal---

Generate a response consisting of a list of key claims that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each claim in the response should contain the following elements:
- "statement": A comprehensive claim statement that is relevant to the user query and include relevant contextual information (e.g., time, place, specific entities, events or examples). Claims should stand on their own and be independent of each other.
- "sources": A list of all source ids from the data tables that are relevant to the claim.
- "score": An integer score between 0-100 that indicates how important the claim is in answering the user's question. Claims that directly answer the query should be given higher scores than those that only provide background information. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.

The response should be JSON formatted as follows:
{{
    "claims": [
        {{"statement": "Statement of claim 1", "sources": [a list of relevant source ids], "score": importance score value (integer)}},
        {{"statement": "Statement of claim 2", "sources": [a list of relevant source ids], "score": importance score value (integer)}},
    ]
}}

"""
