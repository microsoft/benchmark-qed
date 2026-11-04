# Copyright (c) 2025 Microsoft Corporation.
"""Offline consumer integration tests requiring GraphRAG provider-managed auth.

The committed GraphRAG 3.3.0 pin intentionally skips these tests until upstream
support is published. They can also run with an isolated upstream PYTHONPATH.
Only authentication and API responses are stubbed, not the GraphRAG factories.
"""

import asyncio
from unittest.mock import Mock

import litellm
import pytest
from graphrag_llm.completion.lite_llm_completion import LiteLLMCompletion
from graphrag_llm.config.types import AuthMethod
from graphrag_llm.embedding.lite_llm_embedding import LiteLLMEmbedding
from litellm.types.utils import Usage
from pydantic import BaseModel

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.embedding import TextEmbedder
from benchmark_qed.config.llm_config import AuthType, LLMConfig, LLMProvider
from benchmark_qed.llm import chat
from benchmark_qed.llm.factory import ModelFactory

pytestmark = pytest.mark.skipif(
    "provider" not in {method.value for method in AuthMethod},
    reason="Requires an upstream graphrag-llm release with provider-managed auth.",
)


@pytest.fixture(autouse=True)
def prevent_device_flow(monkeypatch):
    """Prevent any real credential cache access, login, or model request."""
    authenticator = Mock()
    authenticator.get_api_key.return_value = "offline-test-token"
    authenticator.get_api_base.return_value = "https://copilot.invalid"
    for module in (
        "litellm.llms.github_copilot.chat.transformation",
        "litellm.llms.github_copilot.embedding.transformation",
    ):
        monkeypatch.setattr(f"{module}.Authenticator", Mock(return_value=authenticator))
    for function in ("completion", "acompletion", "embedding", "aembedding"):
        monkeypatch.setattr(
            litellm,
            function,
            Mock(side_effect=AssertionError("Unstubbed LiteLLM model request")),
        )
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-sent")


def assert_provider_credentials_omitted(kwargs):
    for credential in (
        "api_key",
        "api_token",
        "azure_ad_token",
        "azure_ad_token_provider",
    ):
        assert credential not in kwargs


class Answer(BaseModel):
    answer: str


async def test_real_completion_factory_and_structured_output(monkeypatch):
    requests = []

    async def completion(**kwargs):
        requests.append(kwargs)
        await asyncio.sleep(0)
        return litellm.ModelResponse(
            model="github_copilot/gpt-4o",
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"answer":"grounded"}',
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )

    monkeypatch.setattr(litellm, "acompletion", completion)
    config = LLMConfig(
        llm_provider=LLMProvider.CopilotChat,
        auth_type=AuthType.Provider,
        model="gpt-4o",
    )
    model = ModelFactory.create_chat_model(config)
    assert isinstance(model, LiteLLMCompletion)
    response = await chat(model, "Return an answer.", response_format=Answer)
    assert response.formatted_response == Answer(answer="grounded")
    assert requests[0]["model"] == "github_copilot/gpt-4o"
    assert requests[0]["response_format"] is Answer
    assert_provider_credentials_omitted(requests[0])
    assert model.metrics_store.get_metrics()["total_tokens"] >= 5
    assert model.tokenizer.decode(model.tokenizer.encode("hello")) == "hello"


@pytest.mark.parametrize("batch_size", [1, 2, 10])
async def test_real_embedding_factory_batching_order_and_dimensions(
    monkeypatch, batch_size
):
    requests = []

    async def embedding(**kwargs):
        requests.append(kwargs)
        inputs = kwargs["input"]
        await asyncio.sleep(0.01 if inputs[0] == "0" else 0)
        return litellm.EmbeddingResponse(
            model="github_copilot/text-embedding-3-small",
            data=[
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": [float(text)] * kwargs["dimensions"],
                }
                for index, text in enumerate(inputs)
            ],
            usage=Usage(prompt_tokens=len(inputs), total_tokens=len(inputs)),
        )

    monkeypatch.setattr(litellm, "aembedding", embedding)
    model = ModelFactory.create_embedding_model(
        LLMConfig(
            llm_provider=LLMProvider.CopilotEmbedding,
            auth_type=AuthType.Provider,
            model="text-embedding-3-small",
            call_args={"dimensions": 3},
        )
    )
    assert isinstance(model, LiteLLMEmbedding)
    units = [TextUnit(id=str(i), short_id=None, text=str(i)) for i in range(5)]
    result = await TextEmbedder(model).embed_batch(units, batch_size=batch_size)
    assert [unit.text_embedding for unit in result] == [
        [float(i)] * 3 for i in range(5)
    ]
    assert len(requests) == (5 + batch_size - 1) // batch_size
    assert [text for request in requests for text in request["input"]] == [
        str(i) for i in range(5)
    ]
    for request in requests:
        assert request["model"] == "github_copilot/text-embedding-3-small"
        assert request["dimensions"] == 3
        assert_provider_credentials_omitted(request)
    assert model.metrics_store.get_metrics()["prompt_tokens"] >= 5
    assert model.tokenizer.decode(model.tokenizer.encode("hello")) == "hello"
