# Retrieval Metrics CLI Examples

The retrieval metrics CLI commands evaluate RAG (Retrieval-Augmented Generation) systems on retrieval quality. There are two main commands:

1. **`generate-retrieval-reference`** - Generate ground truth relevance data for text chunks
2. **`retrieval-scores`** - Evaluate RAG methods against the reference data

## Overview

```bash
# Step 1: Generate reference data (one-time)
python -m benchmark_qed autoe generate-retrieval-reference <config_file> [options]

# Step 2: Score RAG methods against reference
python -m benchmark_qed autoe retrieval-scores <config_file> [options]
```

---

## Command 1: generate-retrieval-reference

This command creates ground truth relevance assessments by:
1. Clustering text units into semantic groups
2. For each question, assessing which chunks are relevant using LLM-based relevance scoring
3. Saving the reference data for later evaluation

### Configuration File (`retrieval_reference_config.yaml`)

```yaml
# LLM configuration for relevance assessment
llm_config:
  model: gpt-4o
  auth_type: api_key
  api_key: ${OPENAI_API_KEY}
  llm_provider: openai.chat
  call_args:
    temperature: 0.0
    seed: 42

# Embedding model (for clustering if needed)
embedding_config:
  model: text-embedding-3-large
  auth_type: api_key
  api_key: ${OPENAI_API_KEY}
  llm_provider: openai.embedding

# Question sets to process (supports multiple)
question_sets:
  - name: "global_questions"
    questions_path: "./output/global_questions/selected_questions.json"
  - name: "local_questions"
    questions_path: "./output/local_questions/selected_questions.json"

# Path to text units (parquet, CSV, or JSON with embeddings)
text_units_path: "./output/text_units.parquet"

# Output directory for reference data
output_dir: "./output/retrieval_reference"

# Number of clusters to create (single value or list for multiple runs)
num_clusters:
  - 30
  - 20
  - 10

# Save clustering results separately for debugging
save_clusters: true

# Reference generation settings
semantic_neighbors: 50    # Chunks to test per cluster based on semantic similarity
centroid_neighbors: 10    # Chunks to test per cluster based on centroid distance
relevance_threshold: 2    # Min score to consider relevant (0-3 scale)

# Relevance assessor type:
#   "bing" - UMBRELA DNA prompt (faster, binary relevance)
#   "rationale" - Structured JSON with reasoning (more detailed)
assessor_type: bing

# Concurrency for LLM requests
concurrent_requests: 32

# Optional: limit questions for testing
max_questions: null

# Cache directory (enables resume on failure)
cache_dir: "./output/retrieval_reference/cache"

# Text unit column mappings
text_unit_fields:
  id_col: "id"
  text_col: "text"
  embedding_col: "text_embedding"
  short_id_col: "short_id"
```

### Questions File Format

```json
[
  {
    "question_id": "q1",
    "question": "What were the key economic impacts of the 2024 policy changes?",
    "type": "global"
  },
  {
    "question_id": "q2", 
    "question": "How did Company X respond to the market downturn?",
    "type": "local"
  }
]
```

### CLI Examples

```bash
# Basic usage
python -m benchmark_qed autoe generate-retrieval-reference \
    retrieval_reference_config.yaml

# With model usage statistics
python -m benchmark_qed autoe generate-retrieval-reference \
    retrieval_reference_config.yaml \
    --print-model-usage
```

### Output Structure

```
output/retrieval_reference/
├── clusters/
│   ├── clusters_10.json
│   ├── clusters_20.json
│   └── clusters_40.json
├── global_questions/
│   ├── clusters_10/
│   │   ├── reference.json
│   │   └── model_usage.json
│   ├── clusters_20/
│   │   └── reference.json
│   └── clusters_40/
│       └── reference.json
├── local_questions/
│   └── ... (same structure)
└── cache/
    └── ... (cached LLM responses)
```

---

## Command 2: retrieval-scores

This command evaluates RAG methods by computing precision, recall, and fidelity metrics against the reference data.

### Configuration File (`retrieval_scores_config.yaml`)

```yaml
# LLM configuration for relevance assessment
llm_config:
  model: gpt-4o
  auth_type: api_key
  api_key: ${OPENAI_API_KEY}
  llm_provider: openai.chat

# RAG methods to evaluate
rag_methods:
  - name: "graphrag"
    retrieval_results_path: "./output/graphrag/retrieval_results.json"
  - name: "naive_rag"
    retrieval_results_path: "./output/naive_rag/retrieval_results.json"
  - name: "hyde_rag"
    retrieval_results_path: "./output/hyde_rag/retrieval_results.json"

# Question sets to evaluate (must match reference data)
question_sets:
  - "global_questions"
  - "local_questions"

# Paths to reference data and clusters
reference_dir: "./output/retrieval_reference"
clusters_path: "./output/retrieval_reference/clusters/clusters_40.json"
text_units_path: "./output/text_units.json"

# Output directory
output_dir: "./output/retrieval_scores"

# Evaluation settings
relevance_threshold: 2
context_id_key: "chunk_id"     # Key in retrieval results for chunk ID
context_text_key: "text"       # Key in retrieval results for chunk text

# Statistical significance testing
run_significance_test: true
significance_alpha: 0.05
significance_correction: "holm"  # P-value correction method

# Fidelity metric: "js" (Jensen-Shannon) or "tvd" (Total Variation Distance)
fidelity_metric: "js"

# Cache for relevance assessments
cache_dir: "./output/retrieval_scores/cache"
```

### Retrieval Results File Format

Each RAG method should produce a JSON file with retrieval results:

```json
[
  {
    "question_id": "q1",
    "question": "What were the key economic impacts?",
    "retrieved_chunks": [
      {
        "chunk_id": "chunk_123",
        "text": "The economic impacts included...",
        "score": 0.95
      },
      {
        "chunk_id": "chunk_456",
        "text": "Policy changes led to...",
        "score": 0.87
      }
    ]
  }
]
```

### CLI Examples

```bash
# Basic usage
python -m benchmark_qed autoe retrieval-scores \
    retrieval_scores_config.yaml

# With higher concurrency
python -m benchmark_qed autoe retrieval-scores \
    retrieval_scores_config.yaml \
    --max-concurrent 16

# With model usage statistics
python -m benchmark_qed autoe retrieval-scores \
    retrieval_scores_config.yaml \
    --print-model-usage
```

### Output Files

```
output/retrieval_scores/
├── metrics_summary.csv           # Aggregated metrics per method
├── detailed_scores.csv           # Per-question scores
├── significance_tests.csv        # Statistical test results
├── model_usage.json
└── {question_set}/
    └── {rag_method}/
        └── scores.json
```

---

## Complete Workflow Example

### Step 1: Prepare Data

```bash
# Ensure you have:
# - text_units.parquet (with id, text, text_embedding columns)
# - selected_questions.json (with question_id, question fields)
```

### Step 2: Generate Reference Data

```bash
python -m benchmark_qed autoe generate-retrieval-reference \
    retrieval_reference_config.yaml \
    --print-model-usage
```

### Step 3: Run Your RAG Methods

```bash
# Run each RAG method and save retrieval results
# (This is your own RAG implementation)
python my_graphrag.py --output output/graphrag/retrieval_results.json
python my_naive_rag.py --output output/naive_rag/retrieval_results.json
```

### Step 4: Evaluate Retrieval Quality

```bash
python -m benchmark_qed autoe retrieval-scores \
    retrieval_scores_config.yaml \
    --print-model-usage
```

---

## Configuration Tips

### Assessor Types

| Type | Description | Speed | Detail |
|------|-------------|-------|--------|
| `bing` | UMBRELA DNA prompt | Fast | Binary relevance (0-3) |
| `rationale` | Structured JSON | Slower | Includes reasoning |

### Cluster Counts

- **More clusters** (e.g., 40-50): Finer granularity, better for large corpora
- **Fewer clusters** (e.g., 10-20): Coarser grouping, faster processing
- **Use multiple**: Test different granularities by providing a list

### Caching

Always enable `cache_dir` to:
- Resume interrupted runs
- Avoid re-assessing the same query-chunk pairs
- Share cache across multiple runs

### Concurrency

- Start with `concurrent_requests: 16-32` for OpenAI
- Reduce if hitting rate limits
- Increase for faster processing with higher limits
