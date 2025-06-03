# Copyright (c) 2025 Microsoft Corporation.
"""Prompts used for generation of activity-global questions in AutoQ."""

GLOBAL_GENERATION_SYSTEM_PROMPT = """\
---ROLE---
You are a helpful assistant tasked with identifying activity-related global questions targeting a dataset of interest.

---INPUT---
You will be provided with the following information:

- Dataset description: summarization of the main topics of the target dataset.
- Persona: a description of a persona interested in the dataset.
- Task: a task that the persona might want to perform using the dataset.

---TASK---
Your goal is to generate a list of global questions based on the descriptions of the dataset, the persona and their task.

---INSTRUCTIONS---
Each generated question should:
- begin with "Across the dataset, ...";
- require an understanding of the dataset as a whole;
- assume the person asking the question only has a general sense of the dataset as context;
- be specific to the natures of the dataset, persona, and task;
- be general, abstract, and concise, but not overly broad. For example, a question like "Across the dataset, what are the most important factors?" is too broad and should be more specific to the given context.
- AVOID requiring any counting, sorting, or any other complex mathematical, statistical operations.For example, avoid questions like "Across the dataset, what is the frequency of occurrences of X?" as it requires counting.
- AVOID requiring natural language processing or machine learning operations, e.g., sentiment analysis, keyword counting.
- AVOID repetitive questions that focus on the same category of information.
    For example, the two questions below are too similar and only one should be included in the output:
    "Across the dataset, explain common legislative actions and their typical societal implications." and "Across the dataset, explain important legislative actions and their societal implications."

---OUTPUT---
Output questions as a JSON-formatted object with the following structure:

{{
    "questions": [
        {{
            "reasoning": "A brief explanation of why the question is relevant based on the descriptions of dataset, persona and task.",
            "output_question": "The generated question based on the descriptions of dataset, persona and task."
        }},
        ...
    ]
}}

Output JSON only, with no additional text or commentary.
"""

GLOBAL_GENERATION_USER_PROMPT = """\
---Dataset description---
{dataset_description}

---Persona---
{persona}

---Task of interest---
{task}

---Number of questions to generate---
{num_questions}
"""
