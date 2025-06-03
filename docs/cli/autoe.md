## Pairwise Scoring Configuration

This document describes the configuration schema for scoring a set of conditions using a language model. It includes definitions for conditions, evaluation criteria, and model configuration. For more information about how to configure the LLM check: [LLM Configuration](llm_config.md)

To generate a template configuration file you can run:

```sh
benchmark_qed config init autoe_pairwise local/autoe_pairwise/settings.yaml
```

See more about the config init command: [Config Init CLI](config_init.md)


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

Below is an example of how this configuration might be represented in a YAML file. The API key is referenced using an environment variable.

Save the following yaml file as autoe_pairwise_settings.yaml and use with the command:

```sh
benchmark_qed autoe pairwise-scores autoe_pairwise_settings.yaml local/output_test
```

To run autoe with our [generated answers](https://github.com/microsoft/benchmark-qed/docs/example_notebooks/example_answers). See the CLI Reference section for more options.


```yaml
base:
  name: vector_rag
  answer_base_path: example_answers/vector_rag
others:
  - name: lazygraphrag
    answer_base_path: example_answers/lazygraphrag
  - name: graphrag_global
    answer_base_path: example_answers/graphrag_global
question_sets:
  - activity_global
  - activity_local
trials: 4
llm_config:
  auth_type: api_key
  model: gpt-4.1
  api_key: ${OPENAI_API_KEY}
  llm_provider: openai.chat
  concurrent_requests: 20
```

💡 Note: The api_key field uses an environment variable reference `${OPENAI_API_KEY}`. Make sure to define this variable in a .env file or your environment before running the application.

```
# .env file
OPENAI_API_KEY=your-secret-api-key-here
```


## Reference-Based Scoring Configuration

This document describes the configuration schema for evaluating generated answers against a reference set using a language model. It includes definitions for reference and generated conditions, scoring criteria, and model configuration. For more information about how to configure the LLM check: [LLM Configuration](llm_config.md)

To generate a template configuration file you can run:

```sh
benchmark_qed config init autoe_reference local/autoe_reference/settings.yaml
```

See more about the config init command: [Config Init CLI](config_init.md)


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

Save the following yaml file as autoe_reference_settings.yaml and use with the command:

```sh
benchmark_qed autoe reference-scores autoe_reference_settings.yaml local/output_test
```

To run autoe with our [generated answers](https://github.com/microsoft/benchmark-qed/docs/example_notebooks/example_answers). See the CLI Reference section for more options.


```yaml
reference:
  name: lazygraphrag
  answer_base_path: example_answers/lazygraphrag/activity_global.json

generated:
  - name: vector_rag
    answer_base_path: example_answers/vector_rag/activity_global.json

score_min: 1
score_max: 10

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
💡 Note: The api_key field uses an environment variable reference `${OPENAI_API_KEY}`. Make sure to define this variable in a .env file or your environment before running the application.

```
# .env file
OPENAI_API_KEY=your-secret-api-key-here
```


## CLI Reference

This page documents the command-line interface of the benchmark-qed autoe package.

::: mkdocs-typer2
    :module: benchmark_qed.autoe.cli
    :name: autoe