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

See the [PromptConfig](https://github.com/microsoft/benchmark-qed/tree/main/benchmark_qed/config/prompt_config.py) class for details.

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

## Assertion-Based Scoring Configuration

This section describes the configuration schema for evaluating generated answers against predefined assertions using the LLM-as-a-Judge approach. The `assertion-scores` command **auto-detects** the config format and supports both single-RAG and multi-RAG evaluation.

For more information about how to configure the LLM, please refer to: [LLM Configuration](llm_config.md)

### CLI Usage

```sh
# Single-RAG mode (output directory required)
benchmark-qed autoe assertion-scores config.yaml output_dir

# Multi-RAG mode (output in config, includes significance testing)
benchmark-qed autoe assertion-scores config.yaml
```

The command auto-detects the config format:
- **Single-RAG**: Config has `generated` key → requires output argument
- **Multi-RAG**: Config has `rag_methods` key → output in config, includes significance testing

To create a template configuration file, run:

```sh
benchmark-qed config init autoe_assertion local/assertion_test/settings.yaml
```

For information about the `config init` command, refer to: [Config Init CLI](config_init.md)

---

### Single-RAG Configuration (Legacy)

#### Classes and Fields

#### `Condition`
Represents a condition to be evaluated.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Name of the condition. |
| `answer_base_path` | `Path` | Path to the JSON file containing the answers for this condition. |

---

#### `Assertions`
Defines the assertions to be evaluated.

| Field | Type | Description |
|-------|------|-------------|
| `assertions_path` | `Path` | Path to the JSON file containing the assertions to evaluate. |

---

#### `AssertionConfig`
Top-level configuration for scoring generated answers against assertions.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `generated` | `Condition` | _required_ | The condition containing the generated answers to be evaluated. |
| `assertions` | `Assertions` | _required_ | The assertions to use for evaluation. |
| `pass_threshold` | `float` | `0.5` | Threshold for passing the assertion score. |
| `llm_config` | `LLMConfig` | `LLMConfig()` | Configuration for the LLM used in scoring. |
| `trials` | `int` | `4` | Number of trials to run for each assertion. |

---

### YAML Example

Below is an example showing how this configuration might be represented in a YAML file. The API key is referenced using an environment variable.

```yaml
generated:
  name: vector_rag
  answer_base_path: input/vector_rag/activity_global.json

assertions:
  assertions_path: input/assertions.json

# Pass threshold for assertions
pass_threshold: 0.5

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

> 📋 Assertions json example:
```json
[
  {
    "question_id": "abc123",
    "question_text": "What is the capital of France?",
    "assertions": [
        "The response should align with the following ground truth text: Paris is the capital of France.",
        "The response should be concise and directly answer the question and do not add any additional information."
    ]
  }
]
```

---

### Multi-RAG Configuration

This format evaluates multiple RAG methods against the same assertions in a single command, with optional statistical significance testing at the end.

#### `MultiRAGAssertionConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_dir` | `Path` | _required_ | Base directory containing RAG method answer files. |
| `output_dir` | `Path` | _required_ | Directory to save evaluation results. |
| `rag_methods` | `list[str]` | _required_ | List of RAG method names to evaluate. |
| `question_sets` | `list[str]` | _required_ | List of question set names to evaluate. |
| `assertions_filename_template` | `str` | `"{question_set}_assertions.json"` | Template for assertion filenames. |
| `answers_path_template` | `str` | `"{input_dir}/{rag_method}/{question_set}.json"` | Template for answer file paths. |
| `pass_threshold` | `float` | `0.5` | Threshold for passing the assertion score. |
| `top_k_assertions` | `int \| None` | `None` | Number of top-ranked assertions to evaluate (None = all). |
| `run_significance_test` | `bool` | `True` | Whether to run significance tests after scoring. |
| `significance_alpha` | `float` | `0.05` | Alpha level for significance tests. |
| `significance_correction` | `str` | `"holm"` | P-value correction method. |
| `trials` | `int` | `4` | Number of evaluation trials. |
| `llm_config` | `LLMConfig` | `LLMConfig()` | LLM configuration. |

### YAML Example

```yaml
# Multi-RAG assertion scoring configuration
input_dir: ./input
output_dir: ./output/assertion_scoring

rag_methods:
  - graphrag_global
  - vectorrag
  - lazygraphrag

question_sets:
  - data_global_questions
  - data_local_questions

# Assertion file template: {input_dir}/{question_set}_assertions.json
assertions_filename_template: "{question_set}_assertions.json"

# Answer file template: {input_dir}/{rag_method}/{question_set}.json
answers_path_template: "{input_dir}/{rag_method}/{question_set}.json"

pass_threshold: 0.5
top_k_assertions: null  # Use all assertions, or set to e.g. 5 for top-5

# Significance testing (runs automatically after scoring)
run_significance_test: true
significance_alpha: 0.05
significance_correction: holm

trials: 4

llm_config:
  auth_type: api_key
  model: gpt-4.1
  api_key: ${OPENAI_API_KEY}
  llm_provider: openai.chat
  concurrent_requests: 20
```

### Expected Directory Structure

```
input_dir/
  data_global_questions_assertions.json
  data_local_questions_assertions.json
  graphrag_global/
    data_global_questions.json
    data_local_questions.json
  vectorrag/
    data_global_questions.json
    data_local_questions.json
  lazygraphrag/
    data_global_questions.json
    data_local_questions.json
```

### Output Structure

```
output_dir/
  data_global_questions/
    graphrag_global_assertion_scores.csv
    graphrag_global_summary_by_question.csv
    graphrag_global_summary_by_assertion.csv
    vectorrag_assertion_scores.csv
    ...
    significance_per_question.csv
  data_local_questions/
    ...
  assertion_scores_summary.csv
  assertion_scores_pivot_summary.csv
  model_usage.json
```

---

## Hierarchical Assertion Scoring Configuration

This section describes the configuration schema for evaluating generated answers against hierarchical assertions (global assertions with supporting assertions) using the LLM-as-a-Judge approach. This is useful when you have multi-level assertions where a global claim is supported by more specific sub-claims.

The `hierarchical-assertion-scores` command **auto-detects** the config format and supports both single-RAG and multi-RAG evaluation.

### CLI Usage

```sh
# Single-RAG mode (output directory required)
benchmark-qed autoe hierarchical-assertion-scores config.yaml output_dir

# Multi-RAG mode (output in config, includes significance testing)
benchmark-qed autoe hierarchical-assertion-scores config.yaml
```

The command auto-detects the config format:
- **Single-RAG**: Config has `generated` key → requires output argument
- **Multi-RAG**: Config has `rag_methods` key → output in config, includes significance testing

---

### Single-RAG Configuration (Legacy)

#### `HierarchicalAssertionConfig`
Top-level configuration for scoring generated answers against hierarchical assertions.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `generated` | `Condition` | _required_ | The condition containing the generated answers to be evaluated. |
| `assertions` | `Assertions` | _required_ | The assertions with supporting_assertions field for hierarchical scoring. |
| `mode` | `str` | `"staged"` | Evaluation mode: `"staged"` (default) evaluates global assertions first, then supporting assertions only for passed globals. `"joint"` evaluates both together (cheaper but may have anchoring bias). |
| `pass_threshold` | `float` | `0.5` | Threshold for passing the assertion score. |
| `detect_discovery` | `bool` | `true` | Whether to detect information in answers beyond supporting assertions. |
| `llm_config` | `LLMConfig` | `LLMConfig()` | Configuration for the LLM used in scoring. |
| `trials` | `int` | `4` | Number of trials to run for each assertion. |

---

### Evaluation Modes

- **`staged`** (default): Evaluates global assertions first using standard scoring, then evaluates supporting assertions only for globals that passed. This ensures the global pass rate matches what you'd get from standard assertion scoring. More expensive but avoids anchoring bias.

- **`joint`**: Evaluates global and supporting assertions together in a single LLM call. Cheaper but may have anchoring bias where supporting assertion results influence the global evaluation.

---

### YAML Example (Single-RAG)

```yaml
generated:
  name: graphrag_global
  answer_base_path: input/graphrag_global/activity_global.json

assertions:
  assertions_path: input/hierarchical_assertions.json

# Evaluation mode: "staged" (default) or "joint"
mode: staged

# Pass threshold for assertions
pass_threshold: 0.5

# Whether to detect discovery (info beyond supporting assertions)
detect_discovery: true

trials: 4

llm_config:
  auth_type: api_key
  model: gpt-4.1
  api_key: ${OPENAI_API_KEY}
  llm_provider: openai.chat
  concurrent_requests: 20
```

---

### Multi-RAG Configuration

This format evaluates multiple RAG methods against the same hierarchical assertions in a single command, with optional statistical significance testing at the end.

#### `MultiRAGHierarchicalAssertionConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_dir` | `Path` | _required_ | Base directory containing RAG method answer files and assertions. |
| `output_dir` | `Path` | _required_ | Directory to save evaluation results. |
| `rag_methods` | `list[str]` | _required_ | List of RAG method names to evaluate. |
| `assertions_file` | `str` | _required_ | Path to hierarchical assertions file (relative to input_dir or absolute). |
| `answers_path_template` | `str` | `"{input_dir}/{rag_method}/data_global.json"` | Template for answer file paths. |
| `question_id_key` | `str` | `"question_id"` | Column name for question ID. |
| `question_text_key` | `str` | `"question_text"` | Column name for question text. |
| `answer_text_key` | `str` | `"answer"` | Column name for answer text. |
| `supporting_assertions_key` | `str` | `"supporting_assertions"` | Column name for supporting assertions. |
| `pass_threshold` | `float` | `0.5` | Threshold for passing the assertion score. |
| `mode` | `str` | `"staged"` | Evaluation mode: `"staged"` or `"joint"`. |
| `run_significance_test` | `bool` | `True` | Whether to run significance tests after scoring. |
| `significance_alpha` | `float` | `0.05` | Alpha level for significance tests. |
| `significance_correction` | `str` | `"holm"` | P-value correction method. |
| `run_clustered_permutation` | `bool` | `False` | Whether to run assertion-level clustered permutation tests as secondary analysis. |
| `n_permutations` | `int` | `10000` | Number of permutations for clustered permutation tests. |
| `permutation_seed` | `int \| None` | `None` | Random seed for reproducibility of permutation tests. |
| `trials` | `int` | `4` | Number of evaluation trials. |
| `llm_config` | `LLMConfig` | `LLMConfig()` | LLM configuration. |

### YAML Example (Multi-RAG)

```yaml
# Multi-RAG hierarchical assertion scoring configuration
input_dir: ./input
output_dir: ./output/hierarchical_assertion_scoring

rag_methods:
  - graphrag_global
  - vectorrag
  - lazygraphrag

# Path to hierarchical assertions file (relative to input_dir)
assertions_file: data_global_assertions.json

# Answer file template
answers_path_template: "{input_dir}/{rag_method}/data_global.json"

# Column names in the data files
question_id_key: question_id
question_text_key: question  # Use "question" if that's the column name in answers
answer_text_key: answer
supporting_assertions_key: supporting_assertions

# Evaluation mode: "staged" (default) or "joint"
mode: staged

pass_threshold: 0.5

# Significance testing (runs automatically after scoring)
run_significance_test: true
significance_alpha: 0.05
significance_correction: holm

# Clustered permutation test (optional secondary analysis)
# Runs assertion-level tests that account for within-question correlation
# by permuting RAG method labels at the question level (Gail et al., 1996).
run_clustered_permutation: false  # Set to true to enable
n_permutations: 10000
permutation_seed: null  # Set for reproducibility

trials: 4

llm_config:
  auth_type: api_key
  model: gpt-4.1
  api_key: ${OPENAI_API_KEY}
  llm_provider: openai.chat
  concurrent_requests: 20
```

### Expected Directory Structure (Multi-RAG)

```
input_dir/
  data_global_assertions.json
  graphrag_global/
    data_global.json
  vectorrag/
    data_global.json
  lazygraphrag/
    data_global.json
```

### Output Structure (Multi-RAG)

```
output_dir/
  graphrag_global/
    hierarchical_scores_raw.csv
    hierarchical_scores_aggregated.csv
  vectorrag/
    hierarchical_scores_raw.csv
    hierarchical_scores_aggregated.csv
  lazygraphrag/
    hierarchical_scores_raw.csv
    hierarchical_scores_aggregated.csv
  all_hierarchical_scores.csv
  hierarchical_comparison_summary.csv
  significance_summary.csv                                  # If run_significance_test=true
  significance_global_pass_rate_group_stats.csv              #   Per-metric detailed files
  significance_global_pass_rate_omnibus.csv
  significance_global_pass_rate_pairwise.csv
  significance_support_level_group_stats.csv
  significance_support_level_omnibus.csv
  significance_support_level_pairwise.csv
  significance_supporting_pass_rate_group_stats.csv
  significance_supporting_pass_rate_omnibus.csv
  significance_supporting_pass_rate_pairwise.csv
  significance_discovery_rate_group_stats.csv
  significance_discovery_rate_omnibus.csv
  significance_discovery_rate_pairwise.csv
  significance_global_pass_rate_clustered_group_stats.csv    # If run_clustered_permutation=true
  significance_global_pass_rate_clustered_omnibus.csv
  significance_global_pass_rate_clustered_pairwise.csv
  significance_support_level_clustered_group_stats.csv
  significance_support_level_clustered_omnibus.csv
  significance_support_level_clustered_pairwise.csv
  significance_supporting_pass_rate_clustered_group_stats.csv
  significance_supporting_pass_rate_clustered_omnibus.csv
  significance_supporting_pass_rate_clustered_pairwise.csv
  significance_discovery_rate_clustered_group_stats.csv
  significance_discovery_rate_clustered_omnibus.csv
  significance_discovery_rate_clustered_pairwise.csv
  model_usage.json
```

---

> 📋 Hierarchical assertions json example:
```json
[
  {
    "question_id": "abc123",
    "question_text": "How did the company's strategy evolve over time?",
    "assertions": [
      {
        "statement": "The response describes the company's strategic evolution across multiple phases.",
        "supporting_assertions": [
          {"id": "sa1", "statement": "The response mentions the initial focus on product development."},
          {"id": "sa2", "statement": "The response describes the expansion into new markets."},
          {"id": "sa3", "statement": "The response covers the pivot to subscription-based revenue."}
        ]
      }
    ]
  }
]
```

### Output Metrics

The hierarchical scoring produces additional metrics beyond standard assertion scoring:

| Metric | Description |
|--------|-------------|
| `global_score` | Binary (0/1) indicating if the global assertion passed. |
| `global_score_overridden` | True if the global score was forced to 0 due to having no supporting evidence and no discovery. |
| `support_level` | Ratio of supporting assertions that passed (0.0 to 1.0). |
| `n_supporting` | Total number of supporting assertions. |
| `n_supporting_passed` | Number of supporting assertions that passed. |
| `has_discovery` | Whether the answer contains information beyond the supporting assertions. |

> ⚠️ **Note**: If a global assertion passes but has `support_level=0` and `has_discovery=False`, the `global_score` is automatically overridden to 0 (fail). This catches suspicious passes where the LLM said the global claim is satisfied but provided no evidence. The `global_score_overridden` field tracks when this occurs.

---

## Assertion Significance Testing

After running assertion scoring for multiple RAG methods, you can perform statistical significance testing to determine if the observed differences in scores are statistically significant.

> ⚠️ **Prerequisites**: The significance testing commands assume that assertion scoring has **already been run** and that score files exist in the output directory. Run `assertion-scores` or `hierarchical-assertion-scores` first to generate the required input files.

### Standard Assertion Significance Testing

To run significance tests on standard assertion scores, use the CLI command:

```sh
benchmark-qed autoe assertion-significance config.yaml
```

#### Configuration Schema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | `Path` | _required_ | Directory containing assertion scoring results. |
| `rag_methods` | `list[str]` | _required_ | List of RAG method names to compare (must match subdirectory names). |
| `question_sets` | `list[str]` | _required_ | List of question set names to analyze. |
| `alpha` | `float` | `0.05` | Significance level for hypothesis tests. |
| `correction_method` | `str` | `"holm"` | P-value correction method: `"holm"`, `"bonferroni"`, or `"fdr_bh"`. |

#### YAML Example

```yaml
# Standard assertion significance test configuration
output_dir: ./output/assertion_scoring

rag_methods:
  - graphrag_global
  - vectorrag
  - lazygraphrag

question_sets:
  - data_global_questions
  - data_local_questions

alpha: 0.05
correction_method: holm
```

#### Expected Directory Structure

The command expects assertion scoring results organized as:

```
output_dir/
  data_global_questions/
    graphrag_global_summary_by_question.csv
    vectorrag_summary_by_question.csv
    lazygraphrag_summary_by_question.csv
  data_local_questions/
    graphrag_global_summary_by_question.csv
    vectorrag_summary_by_question.csv
    lazygraphrag_summary_by_question.csv
```

Each summary file should contain `success` and `fail` columns for per-question accuracy calculation.

---

### Hierarchical Assertion Significance Testing

> 💡 **Note**: If you use multi-RAG mode with `run_significance_test: true`, significance tests are run automatically. This standalone command is only needed when you want to run significance tests on previously generated scores, or when you ran single-RAG scoring for each method separately.

To run significance tests on hierarchical assertion scores, use the CLI command:

```sh
benchmark-qed autoe hierarchical-assertion-significance config.yaml
```

This tests four metrics:

| Metric | Description |
|--------|-------------|
| `global_pass_rate` | Per-question global assertion pass rate |
| `support_level` | Per-question average support level |
| `supporting_pass_rate` | Per-question supporting assertion pass rate |
| `discovery_rate` | Per-question discovery rate |

#### Configuration Schema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scores_dir` | `Path` | _required_ | Directory containing aggregated hierarchical scores CSVs. |
| `rag_methods` | `list[str]` | _required_ | List of RAG method names to compare. |
| `scores_filename_template` | `str` | `"{rag_method}_hierarchical_scores_aggregated.csv"` | Filename template for aggregated scores. Use `{rag_method}` placeholder. |
| `alpha` | `float` | `0.05` | Significance level for hypothesis tests. |
| `correction_method` | `str` | `"holm"` | P-value correction method: `"holm"`, `"bonferroni"`, or `"fdr_bh"`. |
| `output_dir` | `Path \| None` | `None` | Optional directory to save significance test results as CSV files. |
| `run_clustered_permutation` | `bool` | `False` | Whether to run assertion-level clustered permutation tests as secondary analysis. |
| `n_permutations` | `int` | `10000` | Number of permutations for the clustered permutation test. |
| `permutation_seed` | `int \| None` | `None` | Random seed for reproducibility of permutation tests. |

#### YAML Example

```yaml
# Hierarchical assertion significance test configuration
scores_dir: ./output/hierarchical_scoring

rag_methods:
  - graphrag_global
  - vectorrag
  - lazygraphrag

# Template for finding aggregated score files
scores_filename_template: "{rag_method}_hierarchical_scores_aggregated.csv"

alpha: 0.05
correction_method: holm

# Optional: save detailed results to CSV
output_dir: ./output/significance

# Clustered permutation test (optional secondary analysis)
# Runs assertion-level tests that account for within-question correlation.
run_clustered_permutation: false  # Set to true to enable
n_permutations: 10000
permutation_seed: null  # Set for reproducibility
```

#### Output Files

When `output_dir` is provided, for each of the four metrics (`global_pass_rate`, `support_level`, `supporting_pass_rate`, `discovery_rate`) the following files are saved:

- `significance_{metric}_group_stats.csv` - Per-method descriptive statistics (n, mean, std, median, min, max)
- `significance_{metric}_omnibus.csv` - Omnibus test result (test name, statistic, p-value)
- `significance_{metric}_pairwise.csv` - Pairwise post-hoc comparisons with corrected p-values

If `run_clustered_permutation` is enabled, the same three files are also produced for the assertion-level clustered permutation tests, using the key `{metric}_clustered` (e.g., `significance_global_pass_rate_clustered_omnibus.csv`).

---

### Statistical Tests Used

The significance testing uses the following statistical tests:

| Test Type | Test Used | When Applied |
|-----------|-----------|--------------|
| Omnibus (paired) | Friedman test | When comparing same questions across methods |
| Omnibus (unpaired) | Kruskal-Wallis test | When questions differ across methods |
| Post-hoc (paired) | Wilcoxon signed-rank | Pairwise comparisons with paired data |
| Post-hoc (unpaired) | Mann-Whitney U | Pairwise comparisons with unpaired data |
| Assertion-level (clustered) | Clustered permutation test | When testing at assertion level with within-question correlation (optional secondary analysis) |

P-value correction is applied to post-hoc tests to control for multiple comparisons.

> 💡 **Clustered Permutation Test**: This optional secondary analysis tests significance at the assertion level while accounting for the fact that assertions within the same question are correlated (they share the same RAG answer). It permutes RAG method labels at the question (cluster) level, preserving the within-question correlation structure. This is a well-established approach from cluster-randomized trial methodology (Gail et al., 1996; Ernst, 2004). Enable with `run_clustered_permutation: true`.

---

## CLI Reference

This section documents the command-line interface of the BenchmarkQED's AutoE package.

::: mkdocs-typer2
    :module: benchmark_qed.autoe.cli
    :name: autoe