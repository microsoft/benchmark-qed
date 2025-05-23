import asyncio
from typing import Any

import numpy as np

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autoq.data_model.question import Question
from benchmark_qed.autoq.question_gen.data_questions.claim_extractor.local_claim_extractor import (
    DataLocalClaimExtractor,
)
from benchmark_qed.autoq.question_gen.data_questions.claim_extractor.typing import (
    ClaimExtractionResult,
)
from benchmark_qed.llm.type.base import ChatModel


class DataGlobalClaimExtractor:
    """Extract claims relevant to global questions by aggregating claims from associated local questions."""

    def __init__(
        self,
        llm: ChatModel,
        local_questions: list[Question],
        claim_extractor_params: dict[str, Any] = {},
        concurrent_coroutines: int = 32,
    ):
        self.local_questions: dict[str, Question] = {q.text: q for q in local_questions}
        self.local_claim_extractor: DataLocalClaimExtractor = DataLocalClaimExtractor(
            llm=llm, **claim_extractor_params
        )
        self.concurrent_coroutines = concurrent_coroutines
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(
            self.concurrent_coroutines
        )

    async def aextract_claims(
        self, question_text: str, question_references: list[str]
    ) -> ClaimExtractionResult:
        """Extract claims for the given question by extracting claims from each subquery in the question references"""
        mapped_question_references = [
            self.local_questions[ref] for ref in question_references
        ]
        all_claim_results = await asyncio.gather(*[
            self.aextract_claim_subquery(question_text, sub_query)
            for sub_query in mapped_question_references
        ])

        # compute average reference coverage and total number of relevant references
        reference_coverage = float(
            np.mean([
                sub_query_result.reference_coverage
                for sub_query_result in all_claim_results
            ])
        )
        relevant_references_count = sum([
            sub_query_result.relevant_references_count
            for sub_query_result in all_claim_results
        ])

        # flatten the list of claims
        claims = [
            claim
            for sub_query_result in all_claim_results
            for claim in sub_query_result.claims
            if claim["statement"] != "" and len(claim["source_ids"]) > 0
        ]
        return ClaimExtractionResult(
            claims=claims,
            reference_coverage=reference_coverage,
            relevant_references_count=relevant_references_count,
        )

    async def aextract_claim_subquery(
        self, question_text: str, sub_query: Question
    ) -> ClaimExtractionResult:
        if sub_query.references is None or len(sub_query.references) == 0:
            return ClaimExtractionResult(
                claims=[], reference_coverage=0.0, relevant_references_count=0
            )

        async with self.semaphore:
            return await self.local_claim_extractor.aextract_claims(
                question_text, _parse_subquery_references(sub_query.references)
            )


def _parse_subquery_references(references: list[str]) -> list[TextUnit]:
    text_units: list[TextUnit] = []
    for reference in references:
        parsed_reference = reference.replace("Input Text ", "").split(":")
        text_units.append(
            TextUnit(
                id=parsed_reference[0],
                short_id=parsed_reference[0],
                text=":".join(parsed_reference[1:]),
                attributes={},
            )
        )
    return text_units
