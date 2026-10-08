# Copyright (c) 2025 Microsoft Corporation.

from unittest.mock import MagicMock

from graphrag_llm.completion import LLMCompletion

from benchmark_qed.autod.data_processor.embedding import TextEmbedder
from benchmark_qed.autoq.config import AssertionConfig
from benchmark_qed.autoq.question_gen.data_questions.global_question_gen import (
    DataGlobalQuestionGen,
)


def test_claim_extractor_params_are_forwarded_to_local_extractor() -> None:
    llm_params = {"temperature": 0.0}
    assertion_config = AssertionConfig()
    assertion_config.global_.max_assertions = 0

    generator = DataGlobalQuestionGen(
        llm=MagicMock(spec=LLMCompletion),
        text_embedder=MagicMock(spec=TextEmbedder),
        local_questions=[],
        claim_extractor_params={"llm_params": llm_params},
        assertion_config=assertion_config,
        enable_question_validation=False,
    )

    assert generator.claim_extractor.local_claim_extractor.llm_params is llm_params
