# Copyright (c) 2025 Microsoft Corporation.

MAP_SUMMARY_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant tasked with generating a summary of a dataset given a set of source texts.

---Goal---

Generate a summary consisting of a list of key topics discussed in the source texts.

Each key topic in the summary should include:
- Description: A concise description of the topic. If possible, specify the type of source material the topic is based on (e.g., news articles, podcast transcripts, ArXiv research papers, etc.). If the source type cannot be confidently determined, omit it.
- Importance Score: An integer score between 1-100 that indicates how prominent or significant the topic is within the source texts. A higher score means the topic is more frequently mentioned or emphasized.

The summary should be JSON formatted as follows:
{{
    
    "topics": [
        {{"description": "<Description of topic 1>", "score": <Integer score between 1-100>}},
        {{"description": "<Description of topic 2>", "score": <Integer score between 1-100>}},
    ]
}}
"""

MAP_SUMMARY_USER_PROMPT = """
---Source Texts from the Dataset--
{dataset_data}

"""


REDUCE_SUMMARY_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant tasked with generating a summary of a dataset.

---Goal---

Generate a dataset summary of the target length and format that synthesizes all the summaries from multiple analysts who focused on different parts of the dataset.
The summary should include:
1. Data Sources: The primary types of data sources used in the dataset (e.g., news articles, academic papers, social media posts, podcast transcripts, etc.). Base this on the data sources mentioned in the analysts’ summaries. If the source type is unclear or inconsistent, omit it.
2. Prominent Focus: If the dataset has a prominent focus or theme, describe it first along with related subtopics. Then, summarize other topic areas and explain how they are linked to the main focus.
3. Main Topics: List the main topics covered in the analysts' summaries, but do NOT mention the roles of multiple analysts in the summarization process.

---Important Guidelines---
1. Do NOT include specific entities, events, concrete examples, or specific details. Focus on general categories of information.
For example, avoid overly specific summaries like:
- **Economic and Business Developments**: Coverage includes economic activities and business regulations, such as the licensing of cannabis consumption lounges and the challenges faced by rural economies.
Instead, use more general summaries like:
- **Economic and Business Developments**: Coverage includes economic activities and business regulations, such as licensing and economic challenges.
2. Do NOT include any citations.
"""

REDUCE_SUMMARY_USER_PROMPT = """
---Analyst Summaries---
{map_summaries}

---Target response length and format---
{response_type}
"""

NO_DATA_ANSWER = "I am sorry but I am unable to summarize given the provided data."
