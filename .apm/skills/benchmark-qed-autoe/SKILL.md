---
name: benchmark-qed-autoe
description: >
  Evaluate RAG system outputs using benchmark-qed scoring methods. Use when:
  running pairwise comparisons, reference-based scoring, assertion-based
  evaluation (flat or hierarchical), retrieval metrics, or statistical
  significance tests on RAG outputs. Also use when the user wants to score,
  compare, or evaluate RAG methods, measure retrieval quality, or run
  significance tests on benchmark results — even if they don't say "autoe"
  explicitly.
---

# Benchmark-QED Evaluation (autoe)

Evaluate and compare RAG system outputs using LLM-judged scoring, assertion-based evaluation, and retrieval metrics — all with built-in statistical significance testing.

## Prerequisites

- Generated questions/assertions from the autoq pipeline (or your own)
- RAG method answer files (JSON, one per method per question set)
- A valid `settings.yaml` for the evaluation type
- A configured workspace with valid `settings.yaml` (use the `benchmark-qed-setup` skill to initialize and configure)
- LLM API key configured

Run all commands with:
```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed <command>
```

## Evaluation Methods Overview

| Method | Command | Best for |
|--------|---------|----------|
| Pairwise comparison | `autoe pairwise-scores` | Comparing two RAG methods head-to-head |
| Reference scoring | `autoe reference-scores` | Scoring against gold-standard answers |
| Assertion scoring | `autoe assertion-scores` | Evaluating with ground-truth assertions (single or multi-RAG) |
| Hierarchical assertions | `autoe hierarchical-assertion-scores` | Global + local assertion hierarchies |
| Retrieval metrics | `autoe retrieval-scores` | Precision, recall, fidelity of retrieval |
| Significance tests | `autoe assertion-significance` | Post-hoc significance on existing scores |

## Commands

### 1. Pairwise Scores

Compare RAG methods using LLM-judged pairwise comparisons.

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoe pairwise-scores <config.yaml> <output_dir> [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--alpha` | `0.05` | P-value threshold for significance |
| `--exclude-criteria` | `[]` | Criteria to exclude (repeatable) |
| `--print-model-usage` | `false` | Print LLM token usage |

**Config requires**: `base` (reference method), `others` (methods to compare), `question_sets`, `criteria`, `trials` (must be even), `llm_config`, `prompt_config`

Default criteria: `comprehensiveness`, `diversity`, `empowerment`, `relevance`

**Output**: `{question_set}_{base}--{other}.csv`, `win_rates.csv`, `winrates_sig_tests.csv`

### 2. Reference Scores

Score generated answers against reference (gold-standard) answers.

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoe reference-scores <config.yaml> <output_dir> [OPTIONS]
```

**Config requires**: `reference`, `generated` (list), `criteria`, `score_min`/`score_max`, `trials`, `llm_config`

Default criteria: `correctness`, `completeness`. Default score range: 1–10.

**Output**: `reference_scores-{name}.csv`, `model_usage.json`

### 3. Assertion Scores

Evaluate RAG methods using assertion-based scoring. Auto-detects single-RAG vs multi-RAG config.

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoe assertion-scores <config.yaml> <output_dir> [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--alpha` | `0.05` | Significance threshold (multi-RAG) |
| `--print-model-usage` | `false` | Print LLM token usage |

**Auto-detection**: If the YAML contains a `rag_methods` key, it runs in multi-RAG mode with automated significance testing. Otherwise, single-RAG mode.

**Single-RAG output**: `assertion_scores.csv`, `assertion_summary_by_question.csv`, `eval_summary.json`

**Multi-RAG output**: Per-method scores + significance tests in structured `output_dir/`

### 4. Hierarchical Assertion Scores

Score hierarchical assertions (global assertions with supporting local assertions).

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoe hierarchical-assertion-scores <config.yaml> <output_dir> [OPTIONS]
```

**Modes**: `staged` (default — evaluate local first, then global) or `joint` (evaluate together)

**Extra field**: `detect_discovery: true` enables detection of novel findings not covered by assertions.

Also auto-detects single vs multi-RAG config (same as assertion-scores).

### 5. Assertion Significance

Run statistical significance tests on existing assertion scores (no LLM calls).

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoe assertion-significance <config.yaml>
```

**Config requires**: `output_dir`, `rag_methods`, `question_sets`, `alpha`, `correction_method`

**Correction methods**: `holm` (default, recommended), `bonferroni`, `fdr_bh`

### 6. Hierarchical Assertion Significance

Significance tests on hierarchical assertion scores.

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoe hierarchical-assertion-significance <config.yaml>
```

**Config requires**: `scores_dir`, `rag_methods`, `scores_filename_template`, `alpha`, `correction_method`, `output_dir`

### 7. Generate Retrieval Reference

Generate cluster relevance reference data for retrieval evaluation (one-off prep step).

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoe generate-retrieval-reference <config.yaml>
```

**Config requires**: `llm_config`, `embedding_config`, question source (`questions_path` or `question_sets`), `text_units_path`

**Key settings**: `num_clusters`, `assessor_type` (`rationale` or `bing`), `semantic_neighbors`, `centroid_neighbors`

### 8. Retrieval Scores

Evaluate retrieval precision, recall, and fidelity for RAG methods.

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoe retrieval-scores <config.yaml>
```

**Config requires**: `rag_methods`, `question_sets`, `reference_dir`, `text_units_path`, `output_dir`

**Fidelity metrics**: `js` (Jensen-Shannon divergence) or `tvd` (total variation distance)

## Workflow

### Quick Evaluation (Assertion-Based)

- [ ] Step 1: Verify questions and answers exist — list the workspace and confirm a `settings.yaml` (or `config.yaml`), question JSON files (typically under `output/`), and your RAG method answer JSONs are present.
- [ ] Step 2: Initialize eval config — use the `benchmark-qed-setup` skill to create and configure an assertion evaluation workspace.
- [ ] Step 3: Configure settings.yaml with answer paths and assertion paths
- [ ] Step 4: Run evaluation — `uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoe assertion-scores ./eval_workspace/settings.yaml ./eval_output`
- [ ] Step 5: Summarize results — read the CSVs in `<output_dir>` (e.g. `assertion_scores.csv`, `assertion_summary_by_question.csv`) and `eval_summary.json` directly.

### Multi-RAG Comparison

For comparing multiple RAG methods, use multi-RAG config format (include `rag_methods` key in YAML). This gives you automated pairwise significance testing.

## Gotchas

- **Config auto-detection**: `assertion-scores` and `hierarchical-assertion-scores` detect single vs multi-RAG based on the `rag_methods` key in YAML. Ensure your config matches your intent.
- **Trials must be even**: For pairwise scores, `trials` must be even (for counterbalancing). Use 4 as default.
- **Stale outputs**: Several commands skip existing output files. Use a fresh output directory or delete specific files to force re-evaluation.
- **Output is in files**: All scores are written to CSV/JSON files. Parse output files, not CLI stdout.
- **Long-running**: Evaluation with many questions and trials can take hours. Use background execution.
- **No `config init` for hierarchical/retrieval**: The `benchmark-qed-setup` skill only supports `autoe_assertion`, `autoe_pairwise`, and `autoe_reference`. For hierarchical, multi-RAG, and retrieval configs, create YAML manually.
- **Advanced config types**: Use the `benchmark-qed-setup` skill for configuration guidance on advanced config types.
