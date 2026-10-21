# LLM Configuration

This document outlines the configuration options for setting up and using a Large Language Model (LLM) in benchmark_qed. It includes details on supported providers, authentication methods, and runtime parameters.

---

### `LLMConfig`
Defines the configuration for the language model used in scoring or generation tasks.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"gpt-4.1"` | The name of the model to use. Must be a valid model identifier. |
| `auth_type` | `AuthType` | `"api_key"` | Authentication method. Options: `"api_key"` or `"azure_managed_identity"`. |
| `api_key` | `SecretStr` | `OPENAI_API_KEY` environment value, otherwise empty | API key for accessing the model. Required and non-empty for API-key authentication. |
| `concurrent_requests` | `int` | `4` | Number of concurrent requests allowed to the model. |
| `llm_provider` | `LLMProvider \| str` | `"openai.chat"` | Specifies the provider and type of model. See `LLMProvider` enum for options. |
| `azure_identity_scopes` | `list[str]` | `["https://cognitiveservices.azure.com/.default"]` | Accepted compatibility field; the current GraphRAG adapter does not forward it to the provider. |
| `init_args` | `dict[str, Any]` | `{}` | Additional arguments passed during model initialization. |
| `call_args` | `dict[str, Any]` | `{"temperature": 0.0, "seed": 42}` | Parameters passed when invoking the model. |
| `retry` | `RetryConfig \| None` | `None` | Optional retry strategy for transient provider errors (for example HTTP 429 rate limits). |
| `custom_providers` | `list[CustomLLMProvider]` | `[]` | List of custom LLM providers to register. |



### `LLMProvider` Enum

Defines the supported LLM providers and model types.

| Value | Description |
|-------|-------------|
| `openai.chat` | OpenAI's chat-based models. |
| `openai.embedding` | OpenAI's embedding models. |
| `azure.openai.chat` | Azure-hosted OpenAI chat models. |
| `azure.openai.embedding` | Azure-hosted OpenAI embedding models. |
| `azure.inference.chat` | Azure Inference Service chat models. |
| `azure.inference.embedding` | Azure Inference Service embedding models. |

---

### `AuthType` Enum

Specifies the authentication method used to access the model.

| Value | Description |
|-------|-------------|
| `api_key` | Use a static API key for authentication. |
| `azure_managed_identity` | Use Azure Managed Identity for authentication. |

---

### `RetryConfig`

Optional retry behavior for transient failures.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | `"exponential_backoff"` | Retry strategy. Supported values: `"exponential_backoff"`, `"immediate"`. |
| `max_retries` | `int \| None` | `None` | Maximum retry attempts. |
| `base_delay` | `float \| None` | `None` | Base delay in seconds for exponential backoff. |
| `max_delay` | `float \| None` | `None` | Maximum delay in seconds between retries. |
| `jitter` | `bool \| None` | `None` | Whether to randomize delay intervals. |

---

### YAML Example for `LLMConfig`

#### OpenAI

```yaml
llm_config:
  model: "gpt-4.1"
  auth_type: "api_key"
  api_key: ${OPENAI_API_KEY}
  concurrent_requests: 4
  llm_provider: "openai.chat"
  init_args: {}
  call_args:
    temperature: 0.0
    seed: 42
  retry:
    type: exponential_backoff
    max_retries: 6
    base_delay: 2.0
    max_delay: 30.0
    jitter: true
```

```
# .env file
OPENAI_API_KEY=your-secret-api-key-here
```

>💡 Note: The api_key field uses an environment variable reference `${OPENAI_API_KEY}`. Make sure to define this variable in a .env file or your environment before running the application.

#### Azure OpenAI

```yaml
llm_config:
  model: "gpt-4.1"
  auth_type: "azure_managed_identity" # or api_key like the example above
  concurrent_requests: 4
  llm_provider: "azure.openai.chat"
  init_args: 
    api_version: 2024-12-01-preview
    azure_endpoint: https://<instance>.openai.azure.com # Replace <instance> with the actual value
  call_args:
    temperature: 0.0
    seed: 42
  retry:
    type: exponential_backoff
    max_retries: 6
    base_delay: 2.0
    max_delay: 30.0
    jitter: true
```

With `auth_type: azure_managed_identity`, authenticate locally with `az login`
or configure an identity in the hosting environment. Built-in Azure completion
uses the Cognitive Services audience by default. Although `azure_identity_scopes`
is accepted in configuration, the current GraphRAG adapter does not forward that
field; do not rely on it to change provider token scopes.
API-key authentication instead requires
a non-empty `api_key`, typically loaded from the environment.

#### Custom Provider

BenchmarkQED uses the GraphRAG LLM interfaces. Custom providers must implement
the corresponding interface, not the former BenchmarkQED `ChatModel`,
`EmbeddingModel`, or `BaseModelResponse` APIs.

| Provider kind | Interface | Required methods and properties |
|---|---|---|
| Chat | `graphrag_llm.completion.LLMCompletion` | Factory-compatible constructor, `completion`, `completion_async`, `tokenizer`, and `metrics_store` |
| Embedding | `graphrag_llm.embedding.LLMEmbedding` | Factory-compatible constructor, `embedding`, `embedding_async`, `tokenizer`, and `metrics_store` |

The factory supplies keyword arguments including `model_id`, `model_config`,
`tokenizer`, `metrics_store`, and `cache_key_creator`, plus optional metrics,
rate-limiting, retry, and cache components. The configuration object passed to
the provider is a GraphRAG `ModelConfig`, not BenchmarkQED's `LLMConfig`.

For chat providers, return GraphRAG completion responses and support structured
response formats: evaluation workflows depend on `formatted_response`.
Embedding providers return GraphRAG embedding responses.

You can implement the interface for a new backend or subclass an existing
implementation when extending its behavior. For example, the following minimal
provider reuses the built-in LiteLLM implementation under a custom registration
name. It does not introduce a new backend.

Place it in an importable module named `your_package.providers.custom_chat`:

```python
from graphrag_llm.completion.lite_llm_completion import LiteLLMCompletion


class CustomChatModel(LiteLLMCompletion):
    """Reuse LiteLLM completion behavior under a custom provider name."""
```

Register that class through configuration:

```yaml
llm_config:
  model: "gpt-4.1"
  auth_type: "api_key"
  api_key: ${YOUR_API_KEY}
  concurrent_requests: 4
  llm_provider: "custom.chat"
  init_args:
    model_provider: openai
  retry:
    type: exponential_backoff
    max_retries: 6
    base_delay: 2.0
    max_delay: 30.0
    jitter: true
  custom_providers: # list of custom providers
    - model_type: chat 
      name: custom.chat # This name should match the llm_provider above
      module: your_package.providers.custom_chat
      model_class: CustomChatModel
```

`name` must match `llm_provider`, and `model_type` must match the operation
(`chat` or `embedding`). BenchmarkQED imports the class and registers it with
the corresponding GraphRAG factory. In this LiteLLM example,
`init_args.model_provider: openai` selects the underlying OpenAI backend while
`llm_provider: custom.chat` selects the registered class.

Install your provider package into the same environment as BenchmarkQED before
running the CLI.

```
# .env file
YOUR_API_KEY=your-secret-api-key-here
```

>💡 Note: The api_key field uses an environment variable reference `${YOUR_API_KEY}`. Make sure to define this variable in a .env file or your environment before running the application.