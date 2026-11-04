# LLM Configuration

This document outlines the configuration options for setting up and using a Large Language Model (LLM) in benchmark_qed. It includes details on supported providers, authentication methods, and runtime parameters.

---

### `LLMConfig`
Defines the configuration for the language model used in scoring or generation tasks.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"gpt-4.1"` | The name of the model to use. Must be a valid model identifier. |
| `auth_type` | `AuthType` | `"api_key"` | Authentication method: `"api_key"`, `"azure_managed_identity"`, or `"provider"`. |
| `api_key` | `SecretStr` | `"$OPENAI_API_KEY"` | API key for accessing the model. Omit for provider-managed authentication; that mode does not read `OPENAI_API_KEY`. |
| `concurrent_requests` | `int` | `4` | Number of concurrent requests allowed to the model. |
| `llm_provider` | `LLMProvider \| str` | `"openai.chat"` | Specifies the provider and type of model. See `LLMProvider` enum for options. |
| `init_args` | `dict[str, Any]` | `{}` | Additional arguments passed during model initialization. |
| `call_args` | `dict[str, Any]` | `{"temperature": 0.0, "seed": 42}` | Parameters passed when invoking the model. Defaults to `{}` for Copilot embeddings. |
| `retry` | `RetryConfig \| None` | `None` | Optional retry strategy for transient provider errors (for example HTTP 429 rate limits). |
| `custom_providers` | `list[CustomLLMProvider]` | `list` | List of custom LLM providers to register. |



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
| `github.copilot.chat` | GitHub Copilot Chat API through GraphRAG's existing LiteLLM completion provider. |
| `github.copilot.embedding` | GitHub Copilot Embedding API through GraphRAG's existing LiteLLM embedding provider. |

Unknown provider names must be explicitly declared in `custom_providers`; they are not automatically treated as LiteLLM provider names.

---

### `AuthType` Enum

Specifies the authentication method used to access the model.

| Value | Description |
|-------|-------------|
| `api_key` | Use a static API key for authentication. |
| `azure_managed_identity` | Use Azure Managed Identity for authentication. |
| `provider` | Let the provider manage authentication. Required for the GitHub Copilot options; do not supply `api_key` or credential overrides in `init_args`/`call_args`. |

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

> 💡 Note: If you use azure_manager_identity make sure to be authenticated with `az login` on a terminal, if you use api_key make sure to include the api reference and the .env file.

#### GitHub Copilot chat and embeddings

**Release dependency:** this configuration requires upstream `graphrag-llm` support for `auth_method="provider"`. The currently pinned `graphrag-llm==3.3.0` does not support it. Until that support is published and the dependency is updated, selecting provider-managed authentication raises an actionable error. These options are not yet runnable with the committed dependency lock.

The two options map to GraphRAG's existing `type="litellm"` implementations with `model_provider="github_copilot"`. There is no Copilot SDK/CLI adapter, local client, BenchmarkQED token manager, placeholder API key, or Azure authentication workaround.

For AutoQ, configure the models independently:

```yaml
chat_model:
  llm_provider: github.copilot.chat
  model: gpt-4.1
  auth_type: provider
  call_args:
    temperature: 0.0

embedding_model:
  llm_provider: github.copilot.embedding
  model: text-embedding-3-small
  auth_type: provider
  call_args: {}
```

For AutoE, use the chat configuration under `llm_config` instead of `chat_model`. You can also mix providers, such as Copilot chat with OpenAI or Azure embeddings, or OpenAI chat with Copilot embeddings. Only API-key-authenticated sections need an API key.

Model names above are examples, not guarantees of availability. Chat, structured outputs, embedding models, and optional arguments such as embedding dimensions depend on the selected model and your account's entitlements. Choose a model your account can access; provider errors are not replaced with dummy responses or a different provider. Use `call_args: {}` for embeddings unless supplying supported embedding parameters.

LiteLLM performs its own one-time device-flow login and manages its credential cache and refresh. Chat and embeddings share that LiteLLM cache. This login is separate from Copilot CLI authentication; signing into the CLI does not authenticate LiteLLM. Complete the provider login interactively before an unattended run, and follow LiteLLM's device-flow instructions. Do not copy tokens into settings or `.env`.

The interactive wizard lists Copilot separately for chat and embeddings, selects `auth_type: provider`, and omits the API-key field. Copilot-only wizard configurations do not create an API-key placeholder `.env`. The non-interactive scaffold retains OpenAI defaults and documents how to switch either section to Copilot.

#### Custom Provider

BenchmarkQED allows for a custom implementation for a LLM, to do so first it's needed to implement the corresponding Protocol (ChatModel or EmbeddingModel)

```python
class ChatModel(Protocol):
    def get_usage(self) -> dict[str, Any]: ...

    async def chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> BaseModelResponse: ...


class EmbeddingModel(Protocol):
    def get_usage(self) -> dict[str, Any]: ...

    async def embed(self, text_list: list[str], **kwargs: Any) -> list[list[float]]: ...
```


For example, create a python project and copy this file under `your_package.providers.custom_chat.py`

```python
class CustomChatModel:
    def __init__(self, llm_config: LLMConfig) -> None:
        pass

    def get_usage(self) -> dict[str, Any]:
        return self._usage.model_dump()

    async def chat(
        self, messages: list[dict[str, str]], **kwargs: dict[str, Any]
    ) -> BaseModelResponse:
        return BaseModelResponse(
            output=BaseModelOutput(
                content="Hello from custom provider!",
            ),
            history=[],
            usage={},
        )
```

For more information about these protocols and subclasses check [base.py](https://github.com/microsoft/benchmark-qed/blob/main/benchmark_qed/llm/type/base.py)

After implementing your own provider you can use it in a config file as follows:

```yaml
llm_config:
  model: "your_model_name"
  auth_type: "api_key"
  api_key: ${YOUR_API_KEY}
  concurrent_requests: 4
  llm_provider: "custom.chat"
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

```
# .env file
YOUR_API_KEY=your-secret-api-key-here
```

>💡 Note: The api_key field uses an environment variable reference `${YOUR_API_KEY}`. Make sure to define this variable in a .env file or your environment before running the application.