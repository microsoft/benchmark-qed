# Copyright (c) 2025 Microsoft Corporation.
"""Scoring configuration models."""

from pathlib import Path

from pydantic import BaseModel, Field


class Condition(BaseModel):
    """BaseModel for a condition description."""

    name: str = Field(description="Name of the condition.")
    answer_base_path: Path = Field(
        description="Path to the JSON file containing the answers for this condition."
    )


class Criteria(BaseModel):
    """BaseModel for a scoring criteria."""

    name: str = Field(description="Name of the criteria.")
    description: str = Field(
        description="Description of the criteria, including how to evaluate it."
        " This should be a detailed explanation of what the criteria means and how to apply it."
    )


def pairwise_scores_criteria() -> list[Criteria]:
    """Create default criteria for scoring."""
    return [
        Criteria(
            name="comprehensiveness",
            description="How much detail does the answer provide to cover all the aspects and details of the question? A comprehensive answer should be thorough and complete, without being redundant or irrelevant. For example, if the question is 'What are the benefits and drawbacks of nuclear energy?', a comprehensive answer would discuss in detail both the positive and negative aspects of nuclear energy, such as its efficiency, environmental impact, safety, cost, etc. An incomplete answer would only provide the benefits of nuclear energy without describing the drawbacks, or a redundant answer would repeat the same information multiple times. Each claim in a comprehensive answer should be well-supported by specific details and reasoning.",
        ),
        Criteria(
            name="diversity",
            description="How varied and rich is the answer in providing different perspectives and insights on the question? A diverse answer should be multi-faceted and multi-dimensional, offering different viewpoints and angles on the question. For example, if the question is 'What are the causes and effects of climate change?', a diverse answer would provide a variety of causes and effects of climate change, such as greenhouse gas emissions, deforestation, natural disasters, biodiversity loss, etc. A biased answer would only provide one perspective or opinion.",
        ),
        Criteria(
            name="empowerment",
            description="How well does the answer help the reader understand and make informed judgement about the topic without being misled or making fallacious assumptions. Evaluate each answer on the quality of answer as it relates to clearly explaining and providing reasoning and sources behind the claims in the answer.",
        ),
        Criteria(
            name="relevance",
            description="How relevant are the claims and information in the answer to the question? A relevant answer should stay focused on the topic, accurately interpret the question's intent, and avoid introducing unrelated or tangential content. For instance, if the question is 'What are the benefits and drawbacks of nuclear energy?', a relevant answer would focus on nuclear energy specifically, rather than discussing general energy sources or unrelated technologies. Irrelevant answers may include off-topic content, vague generalities, or misinterpret the question's purpose.",
        ),
    ]


def reference_scores_criteria() -> list[Criteria]:
    """Create default criteria for scoring."""
    return [
        Criteria(
            name="correctness",
            description="How factually accurate is the generated answer compared to the reference answer? ",
        ),
        Criteria(
            name="completeness",
            description="Does the generated answer cover all necessary details in the reference answer?",
        ),
    ]
