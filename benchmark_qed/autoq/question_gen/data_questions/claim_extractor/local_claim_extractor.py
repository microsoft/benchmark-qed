# Copyright (c) 2025 Microsoft Corporation.
"""Extract claims for a data-local questions."""

import json
import logging
from pathlib import Path
from string import Template
from typing import Any, cast

import pandas as pd

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.autoq.prompts import data_questions
from benchmark_qed.autoq.question_gen.data_questions.claim_extractor.claim_coverage import (
    get_relevant_references,
)
from benchmark_qed.autoq.question_gen.data_questions.claim_extractor.typing import (
    ClaimExtractionResult,
)
from benchmark_qed.config.defaults import LLM_PARAMS
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

PROMPTS_PATH = Path(data_questions.__file__).parent


class DataLocalClaimExtractor:
    """
    Extract claims from a given set of input texts. This serves two purposes.

    1. To measure source coverage of a given question.
    2. Extracted claims can be used as basis for generating a reference answer to the question.
    """

    def __init__(
        self,
        llm: ChatModel,
        llm_params: dict[str, Any] = LLM_PARAMS,
        json_mode: bool = True,
        system_prompt: Template | None = None,
    ) -> None:
        self.llm = llm
        self.system_prompt: Template = system_prompt or load_template_file(
            PROMPTS_PATH / "claim_extraction_system_prompt.txt"
        )

        self.llm_params = llm_params
        self.json_mode = json_mode
        if self.json_mode:
            self.llm_params["response_format"] = {"type": "json_object"}
        else:
            self.llm_params.pop("response_format", None)

    async def aextract_claims(
        self, question_text: str, question_references: list[TextUnit]
    ) -> ClaimExtractionResult:
        """Extract relevant claims for a given question based on the reference texts."""
        context_text, context_df = _build_context(question_references)
        messages = [
            {
                "role": "system",
                "content": self.system_prompt.substitute(context_data=context_text),
            },
            {"role": "user", "content": question_text},
        ]
        result = await self.llm.chat(messages=messages, **self.llm_params)
        response, j = try_parse_json_object(result.output.content)
        if j == {}:
            msg = f"Invalid json response, returning empty claim list: {response}"
            log.warning(msg)
            claims = [{"statement": "", "sources": [], "source_ids": [], "score": 0}]
            return ClaimExtractionResult(
                claims=claims, relevant_references_count=0, reference_coverage=0.0
            )
        parsed_elements = json.loads(response).get("claims")

        if not parsed_elements or not isinstance(parsed_elements, list):
            log.warning("No claims found in the response, returning empty claim list")
            claims = [{"statement": "", "sources": [], "source_ids": [], "score": 0}]
            return ClaimExtractionResult(
                claims=claims, relevant_references_count=0, reference_coverage=0.0
            )

        parsed_elements = [
            {
                "statement": element.get("statement"),
                "sources": element.get("sources"),
                "score": element.get("score", 0),
            }
            for element in parsed_elements
            if element.get("statement", "") != ""
            and isinstance(element.get("score", 0), int)
            and element.get("score", 0) > 0
        ]

        for element in parsed_elements:
            # remove hallucinated sources that are not in the context records
            sources = element.get("sources", [])
            if isinstance(sources, list):
                sources = [str(source) for source in sources]
                source_records = cast(
                    pd.DataFrame, context_df[context_df["source_id"].isin(sources)]
                ).drop_duplicates(subset=["source_id"])
                if source_records.empty:
                    msg = f"All extracted sources are not in the context records: {sources}"
                    log.warning(msg)
                    continue

                # remove any hallucinated sources that are not in the context records
                element["sources"] = cast(
                    pd.DataFrame, source_records[["source_id", "text"]]
                ).to_dict(orient="records")
                element["source_ids"] = [
                    source["source_id"] for source in element["sources"]
                ]
                if len(element["sources"]) < len(sources):
                    pass
            else:
                element["sources"] = []
                element["source_ids"] = []

        # remove elements with relevance score 0
        parsed_elements = [
            element for element in parsed_elements if len(element["sources"]) > 0
        ]
        relevant_references_count = get_relevant_references(parsed_elements)
        return ClaimExtractionResult(
            claims=parsed_elements,
            relevant_references_count=relevant_references_count,
            reference_coverage=relevant_references_count / len(question_references),
        )


def _build_context(question_references: list[TextUnit]) -> tuple[str, pd.DataFrame]:
    """Build context for the claim extraction prompt."""
    if not question_references or len(question_references) == 0:
        return "", pd.DataFrame()

    context_records = []
    for index, reference in enumerate(question_references):
        context_records.append({
            "source_id": reference.short_id or str(index + 1),
            "text": reference.text,
        })
    context_df = pd.DataFrame(context_records)
    return context_df.to_csv(index=False), context_df
