# Copyright (c) 2025 Microsoft Corporation.
"""Provider routing and authentication regression tests."""

from enum import StrEnum
from unittest.mock import Mock, patch

import pytest
from graphrag_llm.config.types import AuthMethod
from pydantic import SecretStr, ValidationError

from benchmark_qed.config.llm_config import (
    AuthType,
    CustomLLMProvider,
    LLMConfig,
    LLMProvider,
    ModelType,
)
from benchmark_qed.llm.factory import ModelFactory


class ProviderAuthMethod(StrEnum):
    """The upstream contract, used only to inspect consumer translation."""

    ApiKey = "api_key"
    AzureManagedIdentity = "azure_managed_identity"
    Provider = "provider"


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai.chat", "openai"),
        ("openai.embedding", "openai"),
        ("azure.openai.chat", "azure"),
        ("azure.openai.embedding", "azure"),
        ("azure.inference.chat", "azure_ai"),
        ("azure.inference.embedding", "azure_ai"),
    ],
)
def test_existing_provider_routing(provider, expected):
    config = LLMConfig(
        llm_provider=provider,
        api_key=SecretStr("test-key"),
        init_args={"azure_endpoint": "https://example.openai.azure.com"}
        if expected == "azure"
        else {},
    )
    translated = config.to_model_config()
    assert translated.type == "litellm"
    assert translated.model_provider == expected
    assert translated.api_key == "test-key"
    assert translated.auth_method == AuthMethod.ApiKey


def test_api_key_default_reads_environment_at_configuration_time(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    assert LLMConfig().api_key.get_secret_value() == "environment-key"
    monkeypatch.setenv("OPENAI_API_KEY", "changed-key")
    assert LLMConfig().api_key.get_secret_value() == "changed-key"


def test_missing_api_key_still_fails(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="API key is required"):
        LLMConfig()


def test_azure_auth_translation(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-sent")
    config = LLMConfig(
        llm_provider=LLMProvider.AzureOpenAIChat,
        auth_type=AuthType.AzureManagedIdentity,
        model="deployment",
        init_args={
            "azure_endpoint": "https://example.openai.azure.com",
            "api_version": "2024-12-01-preview",
        },
    ).to_model_config()
    assert config.api_key is None
    assert config.auth_method == AuthMethod.AzureManagedIdentity
    assert config.api_base == "https://example.openai.azure.com"
    assert config.api_version == "2024-12-01-preview"
    assert config.azure_deployment_name == "deployment"


@pytest.mark.parametrize(
    "provider", ["github.copilot.chat", "github.copilot.embedding"]
)
def test_copilot_does_not_inherit_openai_key(monkeypatch, provider):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-read")
    config = LLMConfig(llm_provider=provider, auth_type=AuthType.Provider)
    assert config.api_key.get_secret_value() == ""
    assert "api_key" not in config.model_dump()
    assert LLMConfig.model_validate_json(config.model_dump_json()) == config


def test_copilot_embeddings_default_to_no_chat_arguments():
    config = LLMConfig(
        llm_provider=LLMProvider.CopilotEmbedding, auth_type=AuthType.Provider
    )
    assert config.call_args == {}


@pytest.mark.parametrize("auth", [AuthType.API, AuthType.AzureManagedIdentity])
@pytest.mark.parametrize(
    "provider", ["github.copilot.chat", "github.copilot.embedding"]
)
def test_copilot_requires_provider_auth(provider, auth):
    with pytest.raises(ValidationError, match="requires auth_type='provider'"):
        LLMConfig(
            llm_provider=provider,
            auth_type=auth,
            api_key=SecretStr("not-a-copilot-key"),
        )


def test_provider_auth_rejects_explicit_api_key():
    with pytest.raises(ValidationError, match="Omit api_key"):
        LLMConfig(
            llm_provider=LLMProvider.CopilotChat,
            auth_type=AuthType.Provider,
            api_key=SecretStr("not-a-copilot-key"),
        )


@pytest.mark.parametrize("field", ["init_args", "call_args"])
@pytest.mark.parametrize(
    "credential",
    [
        "api_key",
        "api_token",
        "azure_ad_token",
        "azure_ad_token_provider",
        "auth_method",
        "type",
        "model_provider",
    ],
)
def test_provider_auth_rejects_credential_overrides(field, credential):
    with pytest.raises(ValidationError, match="cannot include"):
        LLMConfig.model_validate({
            "llm_provider": "github.copilot.chat",
            "auth_type": "provider",
            field: {credential: "override"},
        })


@pytest.mark.parametrize("model_type", [ModelType.Chat, ModelType.Embedding])
def test_declared_custom_provider_keeps_registered_type(model_type):
    provider = f"custom.{model_type}"
    config = LLMConfig(
        llm_provider=provider,
        api_key=SecretStr("test-key"),
        custom_providers=[
            CustomLLMProvider(
                name=provider,
                model_type=model_type,
                module="example",
                model_class="CustomModel",
            )
        ],
    ).to_model_config()
    assert config.type == provider


def test_unknown_provider_is_actionable():
    config = LLMConfig(llm_provider="typo.chat", api_key=SecretStr("test-key"))
    with pytest.raises(ValueError, match=r"Unknown LLM provider.*custom_providers"):
        config.to_model_config()


@pytest.mark.parametrize("purpose", ["chat", "embedding"])
def test_factory_translates_copilot_contract_without_credentials(monkeypatch, purpose):
    """Inspect consumer kwargs without pretending the pinned upstream supports auth."""
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-sent")
    config = LLMConfig(
        llm_provider=f"github.copilot.{purpose}",
        auth_type=AuthType.Provider,
        model="account-model",
        call_args={"dimensions": 3} if purpose == "embedding" else {"temperature": 0.0},
    )
    with (
        patch("benchmark_qed.config.llm_config.AuthMethod", ProviderAuthMethod),
        patch("benchmark_qed.config.llm_config.ModelConfig") as model_config,
        patch(
            f"benchmark_qed.llm.factory.create_{'completion' if purpose == 'chat' else 'embedding'}"
        ) as create,
    ):
        factory = (
            ModelFactory.create_chat_model
            if purpose == "chat"
            else ModelFactory.create_embedding_model
        )
        assert factory(config) is create.return_value
        create.assert_called_once_with(model_config.return_value)
        kwargs = model_config.call_args.kwargs
        assert kwargs.get("type", "litellm") == "litellm"
        assert kwargs["model_provider"] == "github_copilot"
        assert kwargs["model"] == "account-model"
        assert kwargs["auth_method"] == "provider"
        assert "api_key" not in kwargs
        assert kwargs["azure_deployment_name"] is None
        assert kwargs["call_args"] == config.call_args


def test_old_graphrag_reports_release_dependency(monkeypatch):
    class OldAuthMethod(StrEnum):
        ApiKey = "api_key"
        AzureManagedIdentity = "azure_managed_identity"

    monkeypatch.setattr("benchmark_qed.config.llm_config.AuthMethod", OldAuthMethod)
    config = LLMConfig(
        llm_provider=LLMProvider.CopilotChat, auth_type=AuthType.Provider
    )
    with pytest.raises(
        ValueError, match=r"graphrag-llm release.*auth_method='provider'"
    ):
        config.to_model_config()


@pytest.mark.parametrize("purpose", ["chat", "embedding"])
def test_custom_factory_registration(monkeypatch, purpose):
    provider = f"test.custom.{purpose}"
    model_class = Mock()
    module = Mock(CustomModel=model_class)
    config = LLMConfig(
        llm_provider=provider,
        api_key=SecretStr("test-key"),
        custom_providers=[
            CustomLLMProvider(
                name=provider,
                model_type=ModelType(purpose),
                module="example",
                model_class="CustomModel",
            )
        ],
    )
    monkeypatch.setattr(ModelFactory, f"_registered_{purpose}", set())
    with (
        patch(
            f"benchmark_qed.llm.factory.register_{'completion' if purpose == 'chat' else 'embedding'}"
        ) as register,
        patch(
            f"benchmark_qed.llm.factory.create_{'completion' if purpose == 'chat' else 'embedding'}"
        ) as create,
        patch("benchmark_qed.llm.factory.importlib.import_module", return_value=module),
    ):
        factory = (
            ModelFactory.create_chat_model
            if purpose == "chat"
            else ModelFactory.create_embedding_model
        )
        factory(config)
        register.assert_called_once_with(provider, model_class)
        assert create.call_args.args[0].type == provider
