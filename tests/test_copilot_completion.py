# Copyright (c) 2025 Microsoft Corporation.
"""Tests for the GitHub Copilot CLI chat provider (no network access)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from copilot.session_events import AssistantUsageData
from graphrag_llm.types import LLMCompletionResponse
from pydantic import BaseModel

from benchmark_qed.config.llm_config import (
    COPILOT_COMPLETION_TYPE,
    LLMConfig,
    LLMProvider,
)
from benchmark_qed.llm import copilot_completion
from benchmark_qed.llm.copilot_completion import (
    CopilotCLICompletion,
    _parse_structured,
    _split_messages,
    _strip_code_fence,
)
from benchmark_qed.llm.factory import ModelFactory

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Answer(BaseModel):
    answer: str
    score: int


class _FakeSession:
    def __init__(self, client: _FakeClient, kwargs: dict[str, Any]) -> None:
        self.client = client
        self.kwargs = kwargs
        self.session_id = f"session-{len(client.sessions)}"
        self._handlers: list[Any] = []
        self.disconnected = False

    def on(self, handler: Any) -> None:
        self._handlers.append(handler)

    async def send_and_wait(self, prompt: str, timeout: float) -> Any:
        self.client.prompts.append(prompt)
        usage = AssistantUsageData(
            model=self.kwargs["model"], input_tokens=10, output_tokens=5
        )
        for handler in self._handlers:
            handler(SimpleNamespace(data=usage, agent_id=None))
        return SimpleNamespace(data=SimpleNamespace(content=self.client.reply))

    async def disconnect(self) -> None:
        self.disconnected = True


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.sessions: list[_FakeSession] = []
        self.prompts: list[str] = []
        self.deleted: list[str] = []

    async def create_session(self, **kwargs: Any) -> _FakeSession:
        session = _FakeSession(self, kwargs)
        self.sessions.append(session)
        return session

    async def delete_session(self, session_id: str) -> None:
        self.deleted.append(session_id)

    async def stop(self) -> None:
        return None


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeClient]:
    client = _FakeClient(reply="Hello!")
    runtime = copilot_completion._CopilotRuntime(cli_path=None, github_token=None)
    runtime._client = client
    runtime._models = {"claude-sonnet-5.5"}
    monkeypatch.setattr(copilot_completion, "_get_runtime", lambda *_: runtime)
    yield client
    runtime.close()


def _copilot_config(**overrides: Any) -> LLMConfig:
    return LLMConfig(
        model="claude-sonnet-5.5", llm_provider=LLMProvider.CopilotChat, **overrides
    )


class TestCopilotConfig:
    def test_does_not_require_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = LLMConfig(model="gpt-5.6-sol", llm_provider="copilot.chat")
        assert config.is_copilot
        assert not config.api_key.get_secret_value()

    def test_to_model_config(self) -> None:
        config = _copilot_config(init_args={"timeout": 30, "cli_path": "copilot"})
        model_config = config.to_model_config()
        assert model_config.type == COPILOT_COMPLETION_TYPE
        assert model_config.model == "claude-sonnet-5.5"
        assert model_config.model_extra == {"timeout": 30, "cli_path": "copilot"}

    def test_embedding_not_supported(self) -> None:
        with pytest.raises(ValueError, match="embeddings"):
            ModelFactory.create_embedding_model(_copilot_config())


class TestHelpers:
    def test_split_string_message(self) -> None:
        assert _split_messages("hi") == (None, "hi")

    def test_split_single_user_message(self) -> None:
        system, prompt = _split_messages([
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ])
        assert system == "Be terse."
        assert prompt == "Hello"

    def test_split_multi_turn(self) -> None:
        system, prompt = _split_messages([
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ])
        assert system is None
        assert "[USER]\nQ1" in prompt
        assert "[ASSISTANT]\nA1" in prompt
        assert prompt.rstrip().endswith("[ASSISTANT]")

    def test_rejects_non_text_content(self) -> None:
        with pytest.raises(ValueError, match="text"):
            _split_messages([
                {"role": "user", "content": [{"type": "image_url", "image_url": {}}]}
            ])

    def test_strip_code_fence(self) -> None:
        assert _strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
        assert _strip_code_fence(' {"a": 1} ') == '{"a": 1}'

    def test_parse_structured_repairs_json(self) -> None:
        parsed = _parse_structured(
            "```json\n{'answer': 'x', 'score': 3,}\n```", _Answer
        )
        assert parsed == _Answer(answer="x", score=3)


class TestCopilotCLICompletion:
    def test_factory_creates_copilot_model(self, fake_client: _FakeClient) -> None:
        llm = ModelFactory.create_chat_model(_copilot_config())
        assert isinstance(llm, CopilotCLICompletion)

    def test_completion(self, fake_client: _FakeClient) -> None:
        llm = ModelFactory.create_chat_model(_copilot_config())
        response = llm.completion(
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hi"},
            ]
        )
        assert isinstance(response, LLMCompletionResponse)
        assert response.usage is not None
        assert response.content == "Hello!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5

        session = fake_client.sessions[0]
        assert session.kwargs["model"] == "claude-sonnet-5.5"
        assert session.kwargs["available_tools"] == []
        assert session.kwargs["system_message"] == {
            "mode": "replace",
            "content": "Be terse.",
        }
        assert session.disconnected
        assert fake_client.deleted == [session.session_id]
        assert fake_client.prompts == ["Hi"]

    def test_structured_completion_async(self, fake_client: _FakeClient) -> None:
        fake_client.reply = '```json\n{"answer": "yes", "score": 7}\n```'
        llm = ModelFactory.create_chat_model(_copilot_config())
        response = asyncio.run(
            llm.completion_async(messages="Question?", response_format=_Answer)
        )
        assert isinstance(response, LLMCompletionResponse)
        assert response.formatted_response == _Answer(answer="yes", score=7)
        system_prompt = fake_client.sessions[0].kwargs["system_message"]["content"]
        assert '"score"' in system_prompt

    def test_unavailable_model(self, fake_client: _FakeClient) -> None:
        llm = ModelFactory.create_chat_model(
            LLMConfig(model="not-a-model", llm_provider="copilot.chat")
        )
        with pytest.raises(ValueError, match="not available"):
            llm.completion(messages="Hi")
        assert fake_client.sessions == []

    def test_options_passed_to_session(self, fake_client: _FakeClient) -> None:
        llm = ModelFactory.create_chat_model(
            _copilot_config(
                init_args={"reasoning_effort": "high", "delete_sessions": False}
            )
        )
        llm.completion(messages="Hi")
        assert fake_client.sessions[0].kwargs["reasoning_effort"] == "high"
        assert fake_client.deleted == []

    def test_streaming_not_supported(self, fake_client: _FakeClient) -> None:
        llm = ModelFactory.create_chat_model(_copilot_config())
        with pytest.raises(ValueError, match="streaming"):
            llm.completion(messages="Hi", stream=True)
