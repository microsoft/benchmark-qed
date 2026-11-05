# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Prompt contracts and score plumbing for natural-language negative assertions.

LLM responses are mocked; these tests do not measure a live judge's accuracy.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
import yaml
from graphrag_llm.completion import LLMCompletion
from typer.testing import CliRunner

from benchmark_qed.__main__ import app
from benchmark_qed.autoe import assertion_scores as legacy
from benchmark_qed.autoe import cli as autoe_cli
from benchmark_qed.autoe.assertion import hierarchical, standard
from benchmark_qed.autoe.config import AssertionConfig, NegativeAssertionConfig
from benchmark_qed.autoe.data_model.assertion import (
    AssertionLLMResponse,
    HierarchicalAssertionLLMResponse,
    SupportingAssertionResult,
    SupportingDiscoveryLLMResponse,
)
from benchmark_qed.config.llm_config import AuthType
from benchmark_qed.config.utils import load_template_file

NEGATIVE_ASSERTION = "The answer does not mention Nathan stealing classified info."
POSITIVE_ASSERTION = "The answer identifies Nathan as the project lead."
ANSWER = "Nathan leads the project."


@pytest.mark.parametrize(
    "prompt_name",
    [
        "negative_assertion_system_prompt.txt",
        "negative_hierarchical_assertion_system_prompt.txt",
        "negative_supporting_discovery_system_prompt.txt",
    ],
)
def test_opt_in_judges_explain_negative_assertions(prompt_name: str) -> None:
    """Guard the absence, polarity, scope, and reasoning prompt contract."""
    prompt = load_template_file(standard.ASSERTION_PROMPTS / prompt_name).substitute(
        assertion=NEGATIVE_ASSERTION
    )
    for instruction in (
        "absent from the entire answer",
        "explicit",
        "denial",
        "invert",
        "paraphrases and strong implications",
        "literal text check",
        "does not mention",
        "does not claim",
        "only the answer",
        "rather than inventing a supporting quotation",
    ):
        assert instruction in prompt.lower()
    if prompt_name != "negative_assertion_system_prompt.txt":
        assert "absence is not discovery" in prompt.lower()


@pytest.mark.parametrize("module", [standard, legacy])
@pytest.mark.parametrize("opt_in", [False, True])
@pytest.mark.parametrize(
    ("answer", "judge_score"),
    [(ANSWER, 1), ("Nathan stole classified information.", 0)],
)
async def test_negative_assertion_is_sent_unchanged_and_score_is_not_inverted(
    monkeypatch: pytest.MonkeyPatch,
    module,
    answer: str,
    judge_score: int,
    opt_in: bool,
) -> None:
    response = AssertionLLMResponse(
        score=judge_score, reasoning="Judge's assessment of prohibited content."
    )
    chat = AsyncMock(return_value=SimpleNamespace(formatted_response=response))
    monkeypatch.setattr(module, "chat", chat)

    result = await module.evaluate_assertion(
        llm_client=MagicMock(spec=LLMCompletion),
        assertion=NEGATIVE_ASSERTION,
        question="What is Nathan's role?",
        answer=answer,
        include_score_id_in_prompt=False,
        assessment_system_prompt=(
            load_template_file(
                standard.ASSERTION_PROMPTS / "negative_assertion_system_prompt.txt"
            )
            if opt_in
            else None
        ),
        assessment_user_prompt=(
            load_template_file(
                standard.ASSERTION_PROMPTS / "negative_assertion_user_prompt.txt"
            )
            if opt_in
            else None
        ),
    )

    messages = chat.call_args.kwargs["messages"]
    assert NEGATIVE_ASSERTION in messages[0]["content"]
    assert NEGATIVE_ASSERTION in messages[1]["content"]
    assert answer in messages[1]["content"]
    assert ("negative assertion" in messages[0]["content"]) is opt_in
    prefix = "negative_" if opt_in else ""
    assert messages[0]["content"] == load_template_file(
        standard.ASSERTION_PROMPTS / f"{prefix}assertion_system_prompt.txt"
    ).substitute(assertion=NEGATIVE_ASSERTION)
    assert messages[1]["content"] == load_template_file(
        standard.ASSERTION_PROMPTS / f"{prefix}assertion_user_prompt.txt"
    ).substitute(
        score_id="",
        assertion=NEGATIVE_ASSERTION,
        question="What is Nathan's role?",
        answer=answer,
    )
    assert chat.call_args.kwargs["response_format"] is AssertionLLMResponse
    assert result["assertion"] == NEGATIVE_ASSERTION
    assert result["score"] == judge_score
    assert result["reasoning"] == response.reasoning


@pytest.mark.parametrize("opt_in", [False, True])
def test_mixed_assertions_keep_existing_score_and_output_shape(
    monkeypatch: pytest.MonkeyPatch,
    opt_in: bool,
) -> None:
    """Positive and negative assertions share the existing scoring pipeline."""
    chat = AsyncMock(
        side_effect=[
            SimpleNamespace(
                formatted_response=AssertionLLMResponse(
                    score=score, reasoning=f"Judge score: {score}"
                )
            )
            for score in (1, 0)
        ]
    )
    monkeypatch.setattr(standard, "chat", chat)
    answers = pd.DataFrame([
        {
            "question_id": "q1",
            "question_text": "What is Nathan's role?",
            "answer": ANSWER,
        }
    ])
    assertions = pd.DataFrame([
        {"question_id": "q1", "assertion": assertion}
        for assertion in (POSITIVE_ASSERTION, NEGATIVE_ASSERTION)
    ])
    config_data = {
        "generated": {"name": "test", "answer_base_path": "answers.json"},
        "assertions": {"assertions_path": "assertions.json"},
        "llm_config": {"auth_type": AuthType.AzureManagedIdentity},
    }
    if opt_in:
        config_data["prompt_config"] = {
            "system_prompt": {
                "prompt": standard.ASSERTION_PROMPTS
                / "negative_assertion_system_prompt.txt"
            },
            "user_prompt": {
                "prompt": standard.ASSERTION_PROMPTS
                / "negative_assertion_user_prompt.txt"
            },
        }
    config = AssertionConfig.model_validate(config_data)

    results = standard.get_assertion_scores(
        llm_client=MagicMock(spec=LLMCompletion),
        llm_config=config.llm_config,
        answers=answers,
        assertions=assertions,
        trials=1,
        assessment_system_prompt=config.prompt_config.system_prompt.template,
        assessment_user_prompt=config.prompt_config.user_prompt.template,
    )

    assert results["assertion"].tolist() == [POSITIVE_ASSERTION, NEGATIVE_ASSERTION]
    assert results["score"].tolist() == [1, 0]
    assert set(results.columns) == {
        "score_id",
        "reasoning",
        "score",
        "question",
        "answer",
        "assertion",
        "trial",
    }
    assert chat.await_count == 2
    for call in chat.call_args_list:
        assert ("negative assertion" in call.kwargs["messages"][0]["content"]) is opt_in
    prefix = "negative_" if opt_in else ""
    assert config.prompt_config.system_prompt.prompt == (
        standard.ASSERTION_PROMPTS / f"{prefix}assertion_system_prompt.txt"
    )
    assert config.prompt_config.user_prompt.prompt == (
        standard.ASSERTION_PROMPTS / f"{prefix}assertion_user_prompt.txt"
    )


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("passed", [False, True])
@pytest.mark.parametrize("opt_in", [False, True])
async def test_hierarchical_negative_assertions_keep_pass_fail_meaning(
    monkeypatch: pytest.MonkeyPatch, staged: bool, passed: bool, opt_in: bool
) -> None:
    supporting_results = [
        SupportingAssertionResult(
            id="SA1", reasoning="Prohibited content assessment.", passed=passed
        ),
        SupportingAssertionResult(
            id="SA2", reasoning="Positive fact assessment.", passed=True
        ),
    ]
    response = (
        SupportingDiscoveryLLMResponse(supporting_results=supporting_results)
        if staged
        else HierarchicalAssertionLLMResponse(
            reasoning="Global assessment.",
            global_passed=passed,
            supporting_results=supporting_results,
        )
    )
    chat = AsyncMock(return_value=SimpleNamespace(formatted_response=response))
    monkeypatch.setattr(hierarchical, "chat", chat)
    evaluate = (
        hierarchical.evaluate_supporting_discovery
        if staged
        else hierarchical.evaluate_hierarchical_assertion
    )
    prompt_name = (
        "supporting_discovery_system_prompt.txt"
        if staged
        else "hierarchical_assertion_system_prompt.txt"
    )
    result = await evaluate(
        llm_client=MagicMock(spec=LLMCompletion),
        assertion=NEGATIVE_ASSERTION,
        question="What is Nathan's role?",
        answer=ANSWER,
        supporting_assertions=[NEGATIVE_ASSERTION, POSITIVE_ASSERTION],
        assessment_system_prompt=(
            load_template_file(standard.ASSERTION_PROMPTS / f"negative_{prompt_name}")
            if opt_in
            else None
        ),
    )

    messages = chat.call_args.kwargs["messages"]
    assert ("negative assertions" in messages[0]["content"].lower()) is opt_in
    prefix = "negative_" if opt_in else ""
    assert (
        messages[0]["content"]
        == load_template_file(
            standard.ASSERTION_PROMPTS / f"{prefix}{prompt_name}"
        ).substitute()
    )
    assert f"SA1: {NEGATIVE_ASSERTION}" in messages[1]["content"]
    assert f"SA2: {POSITIVE_ASSERTION}" in messages[1]["content"]
    assert result["supporting_passed"] == [passed, True]
    assert result["supporting_scores"] == [int(passed), 1]
    assert result["has_discovery"] is False
    if not staged:
        assert result["global_score"] == int(passed)


@pytest.mark.parametrize(
    ("config_type", "command", "negative"),
    [
        ("autoe_assertion", "assertion-scores", False),
        ("autoe_negative_assertion", "negative-assertion-scores", True),
    ],
)
@pytest.mark.parametrize("prompt_mode", ["default", "scaffold", "custom"])
def test_scaffold_and_scoring_commands_preserve_prompt_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_type: str,
    command: str,
    negative: bool,
    prompt_mode: str,
) -> None:
    """Run the CLI with real config/input/output handling and mocked LLM calls."""
    runner = CliRunner()
    result = runner.invoke(app, ["config", "init", config_type, str(tmp_path)])
    assert result.exit_code == 0, result.output
    config_path = tmp_path / "settings.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["generated"]["answer_base_path"] = "input/answers.json"
    config["assertions"]["assertions_path"] = "input/assertions.json"
    config["trials"] = 1
    config["cache_config"] = {"type": "none", "storage": None}
    if prompt_mode == "default":
        config.pop("prompt_config")
    elif prompt_mode == "custom":
        config["prompt_config"]["system_prompt"] = {
            "prompt_text": "Custom judge: $${assertion}"
        }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    question = {"question_id": "q1", "question_text": "What is Nathan's role?"}
    (tmp_path / "input" / "answers.json").write_text(
        json.dumps([{**question, "answer": ANSWER}]), encoding="utf-8"
    )
    (tmp_path / "input" / "assertions.json").write_text(
        json.dumps([
            {**question, "assertions": [POSITIVE_ASSERTION, NEGATIVE_ASSERTION]}
        ]),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = MagicMock(spec=LLMCompletion)
    client.metrics_store.get_metrics.return_value = {}
    monkeypatch.setattr(autoe_cli.ModelFactory, "create_chat_model", lambda _: client)
    chat = AsyncMock(
        side_effect=[
            SimpleNamespace(
                formatted_response=AssertionLLMResponse(
                    score=score, reasoning="Assessment."
                )
            )
            for score in (1, 0)
        ]
    )
    monkeypatch.setattr(standard, "chat", chat)
    output = tmp_path / "output"

    result = runner.invoke(app, ["autoe", command, str(config_path), str(output)])

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    scores = pd.read_csv(output / "assertion_scores.csv")
    assert scores["assertion"].tolist() == [POSITIVE_ASSERTION, NEGATIVE_ASSERTION]
    assert scores["score"].tolist() == [1, 0]
    assert (output / "eval_summary.json").exists()
    assert (output / "assertion_summary_by_question.csv").exists()
    assert chat.await_count == 2
    for call in chat.call_args_list:
        system_prompt = call.kwargs["messages"][0]["content"]
        if prompt_mode == "custom":
            assert system_prompt in {
                f"Custom judge: {POSITIVE_ASSERTION}",
                f"Custom judge: {NEGATIVE_ASSERTION}",
            }
        else:
            assert ("negative assertion" in system_prompt) is negative


def test_negative_command_resolves_blob_config_and_forwards_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolver = MagicMock(return_value=tmp_path / "settings.yaml")
    score = MagicMock()
    monkeypatch.setattr(autoe_cli, "resolve_config_path", resolver)
    monkeypatch.setattr(autoe_cli, "_run_single_rag_assertion_scores", score)
    result = CliRunner().invoke(
        app,
        [
            "autoe",
            "negative-assertion-scores",
            "blob://container/evaluation/settings.yaml",
            str(tmp_path / "output"),
            "--account-url",
            "https://example.blob.core.windows.net",
            "--question-id-key",
            "id",
            "--question-text-key",
            "query",
            "--answer-text-key",
            "response",
            "--assertions-key",
            "checks",
            "--no-include-score-id-in-prompt",
            "--print-model-usage",
        ],
    )
    assert result.exit_code == 0, result.output
    assert resolver.call_args.kwargs["account_url"] == (
        "https://example.blob.core.windows.net"
    )
    score.assert_called_once_with(
        config_path=tmp_path / "settings.yaml",
        output=tmp_path / "output",
        print_model_usage=True,
        include_score_id_in_prompt=False,
        question_id_key="id",
        question_text_key="query",
        answer_text_key="response",
        assertions_key="checks",
        config_type=NegativeAssertionConfig,
    )
