# Evaluation Caches

BenchmarkQED can cache individual LLM evaluations so interrupted or repeated
runs do not need to make the same model calls again. Standard and differential
pairwise scoring, reference scoring, standard single-RAG assertion scoring,
chunk-assertion scoring, and retrieval relevance assessment use the shared
GraphRAG cache configuration format.

## Cache backends

Set `cache_config.type` to one of:

| Type | Persistence | Intended use |
|---|---|---|
| `sqlite` | Persistent, local file | Recommended for normal runs, resumability, and concurrent processes. |
| `json` | Persistent GraphRAG storage | Useful when JSON entries are preferred over a SQLite database. |
| `memory` | Current process only | Deduplicates work within one process but cannot resume a later run. |
| `none` | None | Forces every required LLM evaluation to run. |

A SQLite configuration has this shape:

```yaml
cache_config:
  type: sqlite
  storage:
    type: file
    base_dir: .benchmark_qed_cache/pairwise
  database_name: pairwise.sqlite3
```

`base_dir` is a local filesystem path. It is independent of `input_storage` and
`output_storage`; configuring answers or outputs in Azure Blob Storage does not
move a SQLite cache to blob storage.

A JSON file cache can be configured as:

```yaml
cache_config:
  type: json
  storage:
    type: file
    base_dir: .benchmark_qed_cache/pairwise-json
```

Use `memory` or `none` without a storage block:

```yaml
cache_config:
  type: memory
  storage: null
```

```yaml
cache_config:
  type: none
  storage: null
```

## Pairwise cache behavior

Pairwise scoring stores one entry for each question, criterion, and trial. It
does not reuse one stochastic judgment for all trials. The cache identity
includes:

- The question, both answer labels, and both answer texts
- The trial number, which also preserves counterbalanced answer ordering
- The criterion name and description
- The system and user prompt templates
- The model, provider, initialization arguments, and call arguments
- Whether the score ID is included in the prompt

Changing any of these values creates a new cache entry. API keys and other
credentials are redacted and do not affect the cache identity.

Successful judgments are published as soon as they complete. If a run is
interrupted, the next run can reuse completed judgments and evaluate only
missing entries. Expiring work leases prevent concurrent processes sharing the
same persistent cache from evaluating the same entry more than once.

`--include-score-id-in-prompt` controls whether each new provider request
contains a unique score ID. It can avoid provider-side prompt caching, but it
does **not** bypass BenchmarkQED's `cache_config`. Use `cache_config.type: none`
when you need BenchmarkQED to issue fresh LLM requests.

## Differential pairwise cache behavior

Differential pairwise scoring uses two independent namespaces in one configured
cache:

- `differential_pairwise/extraction` stores the common and unique content
  extracted from each answer pair.
- `differential_pairwise/verdict` stores the judgment over the extracted unique
  content.

This stage separation provides two useful forms of reuse:

- If a judge call fails after extraction succeeds, the next run reuses the
  extraction and retries only the verdict.
- If criteria or judge prompts change, extraction remains reusable while the
  verdict is recomputed.

Each trial remains independent. Extraction identity includes the question,
answers and labels, trial/order, extraction prompts, model/provider, call
arguments, and score-ID mode. Verdict identity additionally includes the
extracted content, criteria, and judge prompts.

The generated differential configuration uses:

```yaml
cache_config:
  type: sqlite
  storage:
    type: file
    base_dir: .benchmark_qed_cache/differential_pairwise
  database_name: differential_pairwise.sqlite3
```

Inspect it with:

```sh
benchmark-qed cache inspect \
  .benchmark_qed_cache/differential_pairwise/differential_pairwise.sqlite3
```

The output lists extraction and verdict namespaces separately, making it
possible to confirm that both stages are being persisted.

## Reference score cache behavior

Reference scoring stores one entry for each question, criterion, and trial. Its
identity includes the question, reference and generated answers, criterion,
trial/order, score range, prompts, model/provider configuration, call
arguments, and score-ID mode. Changing any of those values produces a cache
miss.

The generated reference configuration uses:

```yaml
cache_config:
  type: sqlite
  storage:
    type: file
    base_dir: .benchmark_qed_cache/reference
  database_name: reference.sqlite3
```

Inspect it with:

```sh
benchmark-qed cache inspect \
  .benchmark_qed_cache/reference/reference.sqlite3
```

Reference scoring rewrites its output CSV on every invocation, but cache hits
avoid repeating the underlying LLM calls. Set `type: none` with `storage: null`
when fresh judgments are required.

## Standard assertion score cache behavior

Standard single-RAG assertion scoring stores one entry for each question,
assertion, and trial. Its identity includes the question, generated answer,
assertion text, trial, prompt templates, model/provider configuration, call
arguments, and score-ID mode. `top_k` is not part of an entry's identity, so
overlapping assertions remain reusable when that selection limit changes.

The generated `autoe_assertion` configuration uses:

```yaml
cache_config:
  type: sqlite
  storage:
    type: file
    base_dir: .benchmark_qed_cache/assertion
  database_name: assertion.sqlite3
```

Inspect it with:

```sh
benchmark-qed cache inspect \
  .benchmark_qed_cache/assertion/assertion.sqlite3
```

Successful judgments are persisted immediately, so rerunning after an
interruption evaluates only missing entries. Set `type: none` with
`storage: null` for fresh judgments.

This cache applies only to the standard single-RAG `AssertionConfig` path.
Multi-RAG and hierarchical assertion scoring do not use this cache. The
chunk-assertion workflow has its own cache at assertion/chunk granularity under
`.benchmark_qed_cache/chunk_assertions`.

## Output files versus cached judgments

Pairwise scoring has two separate reuse mechanisms:

1. If an output file such as
   `activity_global_vector_rag--lazygraphrag.csv` already exists, the CLI reads
   that completed output and skips generation for that comparison.
2. If the output file does not exist, pairwise scoring checks `cache_config`
   entry by entry and calls the LLM only for cache misses.

To recompute a comparison from scratch, use a new output directory (or remove
the specific completed output CSV) **and** either set `cache_config.type: none`,
use a new cache location, or remove the old cache.

## Inspecting a SQLite cache

Run:

```sh
benchmark-qed cache inspect .benchmark_qed_cache/pairwise/pairwise.sqlite3
```

This reports the cache backend, namespace entry counts, and active evaluation
leases. Use JSON output for scripts:

```sh
benchmark-qed cache inspect .benchmark_qed_cache/pairwise/pairwise.sqlite3 --json
```

The inspection command accepts SQLite cache files only. It does not inspect
`json`, `memory`, or `none` backends.

## Clearing or bypassing a cache

- **One fresh run:** set `type: none`.
- **Keep old results but start a new cache:** change `base_dir` or
  `database_name`.
- **Permanently clear a SQLite cache:** stop active evaluation processes, then
  remove the configured database and its adjacent
  `.benchmark_qed_cache_leases` directory.

Do not delete a cache while evaluations are actively using it.

## Reproducibility guidance

- Keep the cache with the configuration and outputs used to produce a benchmark.
- Use stable model versions and deterministic call arguments where supported.
- A cache hit reproduces the previously stored judgment; it does not ask the
  provider to generate a new sample.
- Use `type: none` when the purpose of a run is to collect new stochastic
  judgments rather than reproduce or resume previous work.
