## Question Generation Configuration

This section provides an overview of the configuration schema for the question generation process, covering input data, sampling, encoding, and model settings. For details on configuring the LLM, see: [LLM Configuration](llm_config.md).

To create a template configuration file, run:

```sh
benchmark-qed config init autoq local/autoq_test/settings.yaml
```

To generate synthetic queries using your configuration file, run:

```sh
benchmark-qed autoq local/autoq_test/settings.yaml local/autoq_test/output
```

For more information about the `config init` command, see: [Config Init CLI](config_init.md)

---

### Classes and Fields

#### `InputConfig`
Configuration for the input data used in question generation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_path` | `Path` | _required_ | Path to the input dataset file. |
| `input_type` | `InputDataType` | `CSV` | The type of the input data (e.g., CSV, JSON). |
| `text_column` | `str` | `"text"` | The column containing the text data. |
| `metadata_columns` | `list[str] \| None` | `None` | Optional list of columns containing metadata. |
| `file_encoding` | `str` | `"utf-8"` | Encoding of the input file. |

---

#### `QuestionConfig`
Configuration for generating standard questions.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_questions` | `int` | `20` | Number of questions to generate per class. |
| `oversample_factor` | `float` | `2.0` | Factor to overgenerate questions before filtering. |

---

#### `ActivityQuestionConfig`
Extends `QuestionConfig` with additional fields for persona-based question generation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_personas` | `int` | `5` | Number of personas to generate questions for. |
| `num_tasks_per_persona` | `int` | `5` | Number of tasks per persona. |
| `num_entities_per_task` | `int` | `10` | Number of entities per task. |

---

#### `EncodingModelConfig`
Configuration for the encoding model used to chunk documents.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"o200k_base"` | Name of the encoding model. |
| `chunk_size` | `int` | `600` | Size of each text chunk. |
| `chunk_overlap` | `int` | `100` | Overlap between consecutive chunks. |

---

#### `SamplingConfig`
Configuration for sampling data from clusters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_clusters` | `int` | `50` | Number of clusters to sample from. |
| `num_samples_per_cluster` | `int` | `10` | Number of samples per cluster. |
| `random_seed` | `int` | `42` | Seed for reproducibility. |

---

#### `AssertionConfig`
Configuration for assertion generation with separate settings for local and global questions.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `local` | `LocalAssertionConfig` | _(see below)_ | Configuration for local assertion generation. |
| `global` | `GlobalAssertionConfig` | _(see below)_ | Configuration for global assertion generation. |

---

#### `LocalAssertionConfig`
Configuration for local assertion generation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_assertions` | `int \| None` | `20` | Maximum assertions per question. Set to `0` to disable, or `None` for unlimited. |
| `enable_validation` | `bool` | `True` | Whether to validate assertions against source data. |
| `min_validation_score` | `int` | `3` | Minimum score (1-5) for grounding, relevance, and verifiability. |
| `concurrent_llm_calls` | `int` | `8` | Concurrent LLM calls for validation. |
| `max_concurrent_questions` | `int \| None` | `8` | Questions to process in parallel. Set to `1` for sequential. |

---

#### `GlobalAssertionConfig`
Configuration for global assertion generation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_assertions` | `int \| None` | `20` | Maximum assertions per question. Set to `0` to disable, or `None` for unlimited. |
| `enable_validation` | `bool` | `True` | Whether to validate assertions against source data. |
| `min_validation_score` | `int` | `3` | Minimum score (1-5) for grounding, relevance, and verifiability. |
| `batch_size` | `int` | `50` | Batch size for map-reduce claim processing. |
| `max_data_tokens` | `int` | `32000` | Maximum input tokens for the reduce step. |
| `concurrent_llm_calls` | `int` | `8` | Concurrent LLM calls for batch processing and validation. |
| `max_concurrent_questions` | `int \| None` | `2` | Questions to process in parallel. Set to `1` for sequential. |

---

#### `AssertionPromptConfig`
Configuration for assertion generation prompts. Each prompt can be specified as a file path or direct text.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `local_assertion_gen_prompt` | `PromptConfig` | _(default file)_ | Prompt for generating assertions from local claims. |
| `global_assertion_map_prompt` | `PromptConfig` | _(default file)_ | Prompt for the map step in global assertion generation. |
| `global_assertion_reduce_prompt` | `PromptConfig` | _(default file)_ | Prompt for the reduce step in global assertion generation. |
| `local_validation_prompt` | `PromptConfig` | _(default file)_ | Prompt for validating local assertions (fact-focused) against source data. |
| `global_validation_prompt` | `PromptConfig` | _(default file)_ | Prompt for validating global assertions (theme-focused) against source data. |

---

#### `QuestionGenerationConfig`
Top-level configuration for the entire question generation process.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | `InputConfig` | _required_ | Input data configuration. |
| `data_local` | `QuestionConfig` | `QuestionConfig()` | Local data question generation settings. |
| `data_global` | `QuestionConfig` | `QuestionConfig()` | Global data question generation settings. |
| `activity_local` | `ActivityQuestionConfig` | `ActivityQuestionConfig()` | Local activity question generation. |
| `activity_global` | `ActivityQuestionConfig` | `ActivityQuestionConfig()` | Global activity question generation. |
| `concurrent_requests` | `int` | `8` | Number of concurrent model requests. |
| `encoding` | `EncodingModelConfig` | `EncodingModelConfig()` | Encoding model configuration. |
| `sampling` | `SamplingConfig` | `SamplingConfig()` | Sampling configuration. |
| `chat_model` | `LLMConfig` | `LLMConfig()` | LLM configuration for chat. |
| `embedding_model` | `LLMConfig` | `LLMConfig()` | LLM configuration for embeddings. |
| `assertions` | `AssertionConfig` | `AssertionConfig()` | Assertion generation configuration. |
| `assertion_prompts` | `AssertionPromptConfig` | `AssertionPromptConfig()` | Assertion prompt configuration. |

---

### YAML Example

Here is an example of how this configuration might look in a YAML file.

```yaml
## Input Configuration
input:
  dataset_path: ./input
  input_type: json
  text_column: body_nitf # The column in the dataset that contains the text to be processed. Modify this for your dataset
  metadata_columns: [headline, firstcreated] # Additional metadata columns to include in the input. Modify this for your dataset
  file_encoding: utf-8-sig

## Encoder configuration
encoding:
  model_name: o200k_base
  chunk_size: 600
  chunk_overlap: 100

## Sampling Configuration
sampling:
  num_clusters: 20
  num_samples_per_cluster: 10
  random_seed: 42

## LLM Configuration
chat_model:
  auth_type: api_key
  model: gpt-4.1
  api_key: ${OPENAI_API_KEY}
  llm_provider: openai.chat
embedding_model:
  auth_type: api_key
  model: text-embedding-3-large
  api_key: ${OPENAI_API_KEY}
  llm_provider: openai.embedding

## Question Generation Sample Configuration
data_local:
  num_questions: 10
  oversample_factor: 2.0
data_global:
  num_questions: 10
  oversample_factor: 2.0
activity_local:
  num_questions: 10
  oversample_factor: 2.0
  num_personas: 5
  num_tasks_per_persona: 2
  num_entities_per_task: 5
activity_global:
  num_questions: 10
  oversample_factor: 2.0
  num_personas: 5
  num_tasks_per_persona: 2
  num_entities_per_task: 5

## Assertion Generation Configuration
assertions:
  local:
    max_assertions: 20  # Set to 0 to disable, or null/None for unlimited
    enable_validation: true  # Enable to filter low-quality assertions
    min_validation_score: 3  # Minimum score (1-5) to pass validation
    concurrent_llm_calls: 8  # Concurrent LLM calls for validation
    max_concurrent_questions: 8  # Parallel questions for assertion generation. Set to 1 for sequential.
  global:
    max_assertions: 20
    enable_validation: true
    min_validation_score: 3
    batch_size: 50  # Batch size for map-reduce processing
    max_data_tokens: 32000  # Max tokens for reduce step
    concurrent_llm_calls: 8  # Concurrent LLM calls for batch processing/validation
    max_concurrent_questions: 2  # Parallel questions for assertion generation. Set to 1 for sequential.

assertion_prompts:
  local_assertion_gen_prompt:
    prompt: prompts/data_questions/assertions/local_claim_assertion_gen_prompt.txt
  global_assertion_map_prompt:
    prompt: prompts/data_questions/assertions/global_claim_assertion_map_prompt.txt
  global_assertion_reduce_prompt:
    prompt: prompts/data_questions/assertions/global_claim_assertion_reduce_prompt.txt
  local_validation_prompt:
    prompt: prompts/data_questions/assertions/local_validation_prompt.txt
  global_validation_prompt:
    prompt: prompts/data_questions/assertions/global_validation_prompt.txt
```

```markdown
# .env file
OPENAI_API_KEY=your-secret-api-key-here
```

>💡 Note: The api_key field uses an environment variable reference `${OPENAI_API_KEY}`. Make sure to define this variable in a .env file or your environment before running the application.

---

## Assertion Generation

Assertions are testable factual statements derived from extracted claims that can be used as "unit tests" to evaluate the accuracy of RAG system answers. Each question can have multiple assertions that verify specific facts the answer should contain.

### How Assertions Work

1. **Claim Extraction**: During question generation, claims (factual statements) are extracted from the source text.
2. **Assertion Generation**: Claims are transformed into testable assertions with clear pass/fail criteria.
3. **Optional Validation**: Assertions can be validated against source data to filter out low-quality assertions.

### Assertion Types

- **Local Assertions**: Generated for `data_local` questions from claims extracted from individual text chunks.
- **Global Assertions**: Generated for `data_global` questions using a map-reduce approach across multiple source documents.

### Validation

When `enable_validation` is set to `true`, each assertion is scored on three criteria (1-5 scale):

| Criterion | Description |
|-----------|-------------|
| **Grounding** | Is the assertion factually supported by the source data? |
| **Relevance** | Is the assertion relevant to the question being asked? |
| **Verifiability** | Can the assertion be objectively verified from an answer? |

Assertions must meet the `min_validation_score` threshold on all three criteria to be included.

### Controlling Assertion Limits

To disable assertion generation entirely, set `max_assertions: 0` for both local and global:

```yaml
assertions:
  local:
    max_assertions: 0
  global:
    max_assertions: 0
```

To generate unlimited assertions (no cap), set `max_assertions: null`:

```yaml
assertions:
  local:
    max_assertions: null  # or omit to use default of 20
  global:
    max_assertions: null
```

---

## Providing Prompts: File or Text

Prompts for question generation can be provided in two ways, as defined by the `PromptConfig` class:

- **As a file path**: Specify the path to a `.txt` file containing the prompt (recommended for most use cases).
- **As direct text**: Provide the prompt text directly in the configuration.

Only one of these options should be set for each prompt. If both are set, or neither is set, an error will be raised.

### Example (File Path)
```yaml
activity_questions_prompt_config:
  activity_local_gen_system_prompt:
    prompt: prompts/activity_questions/local/activity_local_gen_system_prompt.txt
```

### Example (Direct Text)
```yaml
activity_questions_prompt_config:
  activity_local_gen_system_prompt:
    prompt_text: |
      Generate a question about the following activity:
```

This applies to all prompt fields in [`QuestionGenerationConfig`](https://github.com/microsoft/benchmark-qed/tree/main/benchmark_qed/autoq/config.py#L289-L302) (including [map/reduce](https://github.com/microsoft/benchmark-qed/tree/main/benchmark_qed/autoq/config.py#L106-L130), [activity question generation](https://github.com/microsoft/benchmark-qed/tree/main/benchmark_qed/autoq/config.py#L133-L192), and [data question generation](https://github.com/microsoft/benchmark-qed/tree/main/benchmark_qed/autoq/config.py#L195-L233) prompt configs).


See the [PromptConfig](https://github.com/microsoft/benchmark-qed/tree/main/benchmark_qed/config/prompt_config.py) class for details.

---

## CLI Reference

This section documents the command-line interface of the BenchmarkQED's AutoQ package.

::: mkdocs-typer2
    :module: benchmark_qed.autoq.cli
    :name: autoq
