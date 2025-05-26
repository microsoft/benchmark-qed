# Copyright (c) 2025 Microsoft Corporation.
GLOBAL_EXTRACTION_PROMPT = """
You are a helpful assistant tasked with identifying relevant global questions from input questions sharing the same abstract category.

--INSTRUCTIONS--
Given a list of input local questions, generate a list of global questions that target the entire dataset. 

Each global question should:
- begin with 'Across the dataset, ...'. Example: "Across the dataset, what are the most important strategies for improving mental health outcomes?";
- require an understanding of the dataset as a whole;
- be relevant to the abstract category shared by the input local questions;
- be relevant to all the input local questions, not just a subset;
- be general and abstract, short and concise, but not overly broad. For example, a question like "Across the dataset, what are the most important factors?" is too broad and should be more specific to the given input questions.
- assume the person asking only has a general sense of the dataset as context;
- AVOID requiring any counting, sorting, or any other complex mathematical, statistical operations.For example, avoid questions like "Across the dataset, what is the frequency of occurrences of X?" as it requires counting.
- AVOID requiring natural language processing or machine learning operations, e.g., sentiment analysis, keyword counting.

The output question set should be distinct and diverse, avoiding repetitive questions.


--OUTPUT--
Use the following JSON object structure:

{{
    "abstract_category": "The abstract category shared by the input questions.",
    "questions": ["A list of global questions that requires a holistic understanding of the entire dataset, informed by the input local questions."]
}}

Aims for {num_questions} global questions.
"""

GLOBAL_EXTRACTION_INPUT_PROMPT = """\
--INPUT QUESTIONS--
{input_text}

Output the following JSON object structure:

{{
    "abstract_category": "The abstract category shared by the input questions.",
    "questions": ["A list of global questions that requires a holistic understanding of the entire dataset, informed by the input local questions."]
}}

Output a single JSON object only, not a list, with no additional text or commentary. Aim for {num_questions} global questions.
"""
