# LLM Configuration

This document outlines the configuration options for setting up and using a Large Language Model (LLM) in benchmark_qed. It includes details on supported providers, authentication methods, and runtime parameters.

---

### `LLMConfig`
Defines the configuration for the language model used in scoring or generation tasks.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"gpt-4.1"` | The name of the model to use. Must be a valid model identifier. |
| `auth_type` | `AuthType` | `"api_key"` | Authentication method. Options: `"api_key"` or `"azure_managed_identity"`. |
| `api_key` | `SecretStr` | `"$OPENAI_API_KEY"` | API key for accessing the model. Should be provided via environment variable. |
| `concurrent_requests` | `int` | `4` | Number of concurrent requests allowed to the model. |
| `llm_provider` | `LLMProvider \| str` | `"openai.chat"` | Specifies the provider and type of model. See `LLMProvider` enum for options. |
| `init_args` | `dict[str, Any]` | `{}` | Additional arguments passed during model initialization. |
| `call_args` | `dict[str, Any]` | `{"temperature": 0.0, "seed": 42}` | Parameters passed when invoking the model. |

---

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
```

💡 Note: The api_key field uses an environment variable reference `${OPENAI_API_KEY}`. Make sure to define this variable in a .env file or your environment before running the application.

```
# .env file
OPENAI_API_KEY=your-secret-api-key-here
```

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
```

💡 Note: If you use azure_manager_identity make sure to be authenticated with `az login` on a terminal, if you use api_key make sure to include the api reference and the .env file.