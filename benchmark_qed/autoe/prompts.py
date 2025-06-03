# Copyright (c) 2025 Microsoft Corporation.
"""File containing the prompts for the evaluation tasks."""

PAIRWISE_EVALUATION_SYSTEM_PROMPT = """
---Role---
You are an impartial judge responsible for grading two answers to a given question.

---Goal---
Given a question and two answers (Answer 1 and Answer 2), determine which answer is better according to the following criterion:

{criteria_name}: {criteria_description}

Your evaluation should include two components:
- **Reasoning**: A detailed explanation of your assessment, grounded in the evaluation criterion.
- **Winner**: Indicate the better answer using one of the following values:
  - 1 if Answer 1 is better
  - 2 if Answer 2 is better
  - 0 if both answers are equally strong or it is not possible to determine a clear winner

---Evaluation Process---
Follow these steps to make your decision:
1. Identify all claims in each answer that are relevant to the question.
2. For each claim, extract the supporting evidence and reasoning provided.
3. Compare the two answers based on the relevance and strength of their claims and supporting evidence, using the criterion specified above.

---Important Guidelines---
- No position biases: The order in which the answers are presented should NOT influence your judgment.
- Ignore length: Do NOT let the length of the answers affect your evaluation.
- Ignore formatting style: Do NOT consider the structure or presentation (e.g., bullet points vs. paragraphs) when judging answer quality. Focus only on the content.

---Output Format---
Format your response as a JSON object with the following structure:
{{
    "reasoning": "A detailed explanation of your assessment of the two answers.",
    "winner": 1 | 2 | 0
}}

"""

PAIRWISE_EVALUATION_USER_PROMPT = """
{score_id}

---Question---
{question}

---Start of Answer 1---
{answer1}
---End of Answer 1---


---Start of Answer 2---
{answer2}
---End of Answer 2---

Determine which answer is better according to the following criterion:
{criteria_name}: {criteria_description}

Format your response as a JSON object with the following structure:
{{
    "reasoning": "A detailed explanation of your assessment of the two answers.",
    "winner": One of the following values: 1 (if Answer 1 is better), 2 (if Answer 2 is better), or 0 (if both answers are equally strong or it is not possible to determine a clear winner).
}}
"""


REFERENCE_EVALUATION_PROMPT = """
---Role---
You are an impartial judge responsible for assessing the quality of a generated answer in directcomparison to a reference answer.

---Context---
The **generated answer** refers to a response produced by an information retrieval system. The **reference answer** serves as a standard or baseline for comparison.

---Goal---
Given a question, a reference answer, and a generated answer, evaluate the generated answer according to the following criterion:

{criteria_name}: {criteria_description}

Assign a score from {score_min} to {score_max} based on how effectively the generated answer satisfies the criterion in relation to the reference answer.

Your evaluation should include two components:
- **Reasoning**: A detailed explanation of your assessment, grounded in the evaluation criterion.
- **Score**: An integer score from {score_min} to {score_max}.

---Important Guidelines---
- Do NOT penalize the generated answer for including additional relevant details not found in the reference answer, as long as they are relevant to the question and do not contradict the reference.
- Do NOT assume that information absent from the reference is incorrect unless it contradicts the reference.
- Focus solely on the content quality as it relates to the evaluation criterion - ignore differences in style, length, or formatting between the generated and reference answers.

---Output Format---
Format your response as a JSON object with the following structure:
{{
    "reasoning": "A detailed explanation of your assessment of the generated answer in relation to the reference answer.",
    "score": <Integer score from {score_min} to {score_max}>
}}

"""

REFERENCE_EVALUATION_USER_PROMPT = """
{score_id}

---Question---
{query}

--- Start of {answer_1_name} Answer---
{answer_1}
--- End of {answer_1_name} Answer---


--- Start of {answer_2_name} Answer---
{answer_2}
--- End of {answer_2_name} Answer---

Assign a score from {score_min} to {score_max} based on how effectively the generated answer satisfies the following criterion in relation to the reference answer:
{criteria_name}: {criteria_description}

Format your response as a JSON object with the following structure:
{{
    "reasoning": "A detailed explanation of your assessment of the generated answer in relation to the evaluation criterion and reference answer.",
    "score": <Integer score from {score_min} to {score_max}>
}}
"""
