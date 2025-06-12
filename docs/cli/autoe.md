## Pairwise Scoring Configuration

This section describes the configuration schema for performing relative comparisons of RAG methods using the LLM-as-a-Judge approach. It includes definitions for conditions, evaluation criteria, and model configuration. For more information about how to configure the LLM, please refer to: [LLM Configuration](llm_config.md)

To create a template configuration file, run:

```sh
benchmark-qed config init autoe_pairwise local/pairwise_test/settings.yaml
```

To perform pairwise scoring with your configuration file, use:

```sh
benchmark-qed autoe pairwise-scores local/pairwise_test/settings.yaml local/pairwise_test/output
```

For information about the `config init` command, refer to: [Config Init CLI](config_init.md)

---

### Classes and Fields

#### `Condition`
Represents a condition to be evaluated.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Name of the condition. |
| `answer_base_path` | `Path` | Path to the JSON file containing the answers for this condition. |

---

#### `Criteria`
Defines a scoring criterion used to evaluate conditions.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Name of the criterion. |
| `description` | `str` | Detailed explanation of what the criterion means and how to apply it. |

---

#### `PairwiseConfig`
Top-level configuration for scoring a set of conditions.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base` | `Condition \| None` | `None` | The base condition to compare others against. |
| `others` | `list[Condition]` | `[]` | List of other conditions to compare. |
| `question_sets` | `list[str]` | `[]` | List of question sets to use for scoring. |
| `criteria` | `list[Criteria]` | `pairwise_scores_criteria()` | List of criteria to use for scoring. |
| `llm_config` | `LLMConfig` | `LLMConfig()` | Configuration for the LLM used in scoring. |
| `trials` | `int` | `4` | Number of trials to run for each condition. |

---

### YAML Example

Below is an example showing how this configuration might be represented in a YAML file. The API key is referenced using an environment variable.

```yaml
base:
  name: vector_rag
  answer_base_path: input/vector_rag

others:
  - name: lazygraphrag
    answer_base_path: input/lazygraphrag
  - name: graphrag_global
    answer_base_path: input/graphrag_global

question_sets:
  - activity_global
  - activity_local

# Optional: Custom Evaluation Criteria
# You may define your own list of evaluation criteria here. If this section is omitted, the default criteria will be used.
# criteria: 
#   - name: "criteria name"
#     description: "criteria description"

trials: 4

llm_config:
  auth_type: api_key
  model: gpt-4.1
  api_key: ${OPENAI_API_KEY}
  llm_provider: openai.chat
  concurrent_requests: 20
```

```
# .env file
OPENAI_API_KEY=your-secret-api-key-here
```

>💡 Note: The api_key field uses an environment variable reference `${OPENAI_API_KEY}`. Make sure to define this variable in a .env file or your environment before running the application.

---

## Providing Prompts: File or Text

Prompts for pairwise and reference-based scoring can be provided in two ways, as defined by the `PromptConfig` class:

- **As a file path**: Specify the path to a `.txt` file containing the prompt (recommended for most use cases).
- **As direct text**: Provide the prompt text directly in the configuration.

Only one of these options should be set for each prompt. If both are set, or neither is set, an error will be raised.

### Example (File Path)
```yaml
prompt_config:
  user_prompt:
    prompt: prompts/pairwise_user_prompt.txt
  system_prompt:
    prompt: prompts/pairwise_system_prompt.txt
```

### Example (Direct Text)
```yaml
prompt_config:
  user_prompt:
    prompt_text: |
      Please compare the following answers and select the better one.
  system_prompt:
    prompt_text: |
      You are an expert judge for answer quality.
```

This applies to both `PairwiseConfig` and `ReferenceConfig`.

See the [PromptConfig](../../benchmark_qed/config/prompt_config.py) class for details.

---

## Reference-Based Scoring Configuration

This section explains how to configure reference-based scoring, where generated answers are evaluated against a reference set using the LLM-as-a-Judge approach. It covers the definitions for reference and generated conditions, scoring criteria, and model configuration. For details on LLM configuration, see: [LLM Configuration](llm_config.md)

To create a template configuration file, run:

```sh
benchmark-qed config init autoe_reference local/reference_test/settings.yaml
```

To perform reference-based scoring with your configuration file, run:

```sh
benchmark-qed autoe reference-scores local/reference_test/settings.yaml local/reference_test/output
```

For information about the `config init` command, see: [Config Init CLI](config_init.md)

---

### Classes and Fields

#### `Condition`
Represents a condition to be evaluated.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Name of the condition. |
| `answer_base_path` | `Path` | Path to the JSON file containing the answers for this condition. |

---

#### `Criteria`
Defines a scoring criterion used to evaluate conditions.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Name of the criterion. |
| `description` | `str` | Detailed explanation of what the criterion means and how to apply it. |

---

#### `ReferenceConfig`
Top-level configuration for scoring generated answers against a reference.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reference` | `Condition` | _required_ | The condition containing the reference answers. |
| `generated` | `list[Condition]` | `[]` | List of conditions with generated answers to be scored. |
| `criteria` | `list[Criteria]` | `reference_scores_criteria()` | List of criteria to use for scoring. |
| `score_min` | `int` | `1` | Minimum score for each criterion. |
| `score_max` | `int` | `10` | Maximum score for each criterion. |
| `llm_config` | `LLMConfig` | `LLMConfig()` | Configuration for the LLM used in scoring. |
| `trials` | `int` | `4` | Number of trials to run for each condition. |

---

### YAML Example

Below is an example of how this configuration might be represented in a YAML file. The API key is referenced using an environment variable.

```yaml
reference:
  name: lazygraphrag
  answer_base_path: input/lazygraphrag/activity_global.json

generated:
  - name: vector_rag
    answer_base_path: input/vector_rag/activity_global.json

# Scoring scale
score_min: 1
score_max: 10

# Optional: Custom Evaluation Criteria
# You may define your own list of evaluation criteria here. If this section is omitted, the default criteria will be used.
# criteria: 
#   - name: "criteria name"
#     description: "criteria description"

trials: 4

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

```
# .env file
OPENAI_API_KEY=your-secret-api-key-here
```

>💡 Note: The api_key field uses an environment variable reference `${OPENAI_API_KEY}`. Make sure to define this variable in a .env file or your environment before running the application.

---

## CLI Reference

This section documents the command-line interface of the BenchmarkQED's AutoE package.

::: mkdocs-typer2
    :module: benchmark_qed.autoe.cli
    :name: autoe