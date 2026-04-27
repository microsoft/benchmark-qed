---
name: benchmark-qed-autoq
description: >
  Generate benchmark questions and assertions from input data using
  benchmark-qed. Use when: generating local, global, linked, or activity
  questions for RAG benchmarking, creating assertions for existing questions,
  computing assertion statistics, or running the autoq question generation
  pipeline. Also use when the user wants to create a benchmark question set,
  build evaluation questions from a dataset, or generate ground-truth
  assertions — even if they don't say "autoq" explicitly.
---

# Benchmark-QED Question Generation (autoq)

Generate benchmark questions and assertions from input data for RAG evaluation.

## Prerequisites

- A configured workspace with valid `settings.yaml` (use the `/benchmark-qed-setup` skill first)
- A configured workspace with valid `settings.yaml` (use the `benchmark-qed-setup` skill to initialize and configure)
- Input data (CSV or JSON) in the workspace `input/` directory
- Valid LLM API key in `.env`

Run all commands with:
```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed <command>
```

## Commands

### 1. Generate Questions (`autoq`)

The main question generation pipeline. Generates benchmark questions from input data.

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoq <settings.yaml> <output_dir> [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--generation-types` | Specific types to generate (repeatable). CLI default: all except `data_linked`, but this skill always includes `data_linked` |
| `--print-model-usage` | Print LLM token usage stats |

**Generation types and dependencies:**

```
data_local          ← runs first (no dependencies)
  ├── data_global   ← requires data_local candidates
  └── data_linked   ← requires data_local candidates (not in CLI default, but this skill always includes it)

activity_local      ← auto-generates activity_context first
  └── activity_global ← requires activity_local
```

> **Important**: `data_linked` is NOT included in the CLI's default generation types, but this skill always generates it by passing all types explicitly. If running the CLI manually, you must add `--generation-types data_linked`.

> **Gotcha**: `data_global` and `data_linked` silently return empty results if `data_local` hasn't been run first. Always run `data_local` before these types.

**Examples:**
```bash
# Run from the workspace directory (paths resolve relative to settings.yaml location)
cd ./workspace

# Generate all types including data_linked (skill default)
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoq settings.yaml ./output \
  --generation-types data_local --generation-types data_global --generation-types data_linked \
  --generation-types activity_local --generation-types activity_global

# Generate only local questions
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoq settings.yaml ./output --generation-types data_local

# Generate local + linked questions
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoq settings.yaml ./output \
  --generation-types data_local --generation-types data_linked
```

**Output structure:**
```
output_dir/
├── sample_texts.parquet              # Intermediate: clustered text samples
├── data_local_questions/
│   ├── selected_questions.json       # Final curated questions
│   ├── selected_questions_text.json  # Human-readable version
│   └── candidate_questions.json      # All generated candidates
├── data_global_questions/            # Same structure
├── data_linked_questions/            # Same structure + question_stats.json
├── activity_local_questions/         # Same structure
├── activity_global_questions/        # Same structure
├── context/
│   └── activity_context_full.json    # Generated activity context
└── model_usage.json                  # LLM token/cost tracking
```

### 2. Generate Assertions (`generate-assertions`)

Generate ground-truth assertions for existing questions (decoupled from question generation). This is a **top-level** command, not a subcommand of `autoq`.

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed generate-assertions <settings.yaml> <questions.json> <output_dir> [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--type` / `-t` | Assertion type: `local`, `global`, or `linked` (default: `local`) |
| `--print-model-usage` | Print LLM token usage stats |

**Examples:**
```bash
# Run from the workspace directory (paths resolve relative to settings.yaml location)
cd ./workspace

uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed generate-assertions \
  settings.yaml \
  ./output/data_local_questions/candidate_questions.json \
  ./output/data_local_questions/ \
  --type local

uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed generate-assertions \
  settings.yaml \
  ./output/data_global_questions/candidate_questions.json \
  ./output/data_global_questions/ \
  --type global
```

### 3. Assertion Statistics (`assertion-stats`)

Compute quality statistics for assertion files. This is a **top-level** command, not a subcommand of `autoq`.

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed assertion-stats <assertions_path> [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--output` / `-o` | Output path for stats JSON (auto-generated if omitted) |
| `--type` / `-t` | `global`, `map`, or `local` (auto-inferred if omitted) |
| `--quiet` / `-q` | Suppress console output |

**Examples:**
```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed assertion-stats ./output/assertions.json
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed assertion-stats ./output/data_global_questions/ -q
```

## Workflow

### Standard Question Generation Flow

- [ ] Step 1: Initialize workspace if needed — use the `benchmark-qed-setup` skill to create and configure the workspace. Verify `settings.yaml`, `.env`, and `input/` exist.
- [ ] Step 2: `cd <workspace_dir>` then run question generation — `uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoq settings.yaml ./output --generation-types data_local --generation-types data_global --generation-types data_linked --generation-types activity_local --generation-types activity_global`
- [ ] Step 3: Verify output artifacts — list `<output_dir>` and confirm the per-type `selected_questions.json` files (see "Output structure" above) plus `model_usage.json` exist.
- [ ] Step 4: (Optional) Generate additional assertions — use `generate-assertions`
- [ ] Step 5: (Optional) Check assertion quality — use `assertion-stats`

## Gotchas

- **Path resolution**: The `autoq` and `generate-assertions` commands resolve `output_dir` (and other relative paths) **relative to the settings.yaml file's directory**, not the current working directory. Always `cd` into the workspace directory first, or use absolute paths. For example, running `benchmark-qed autoq workspace/settings.yaml workspace/output` from the repo root creates output at `workspace/workspace/output/` (not `workspace/output/`).
- **Stale outputs**: The pipeline skips steps if output files already exist (`sample_texts.parquet`, `activity_context_full.json`). Use a fresh output directory for clean runs, or delete specific files to re-run a step.
- **Long-running**: Question generation with large datasets can take hours. Use background execution and monitor via `model_usage.json` presence.
- **Output is in files, not stdout**: All results are written to JSON/CSV/Parquet files. Parse the output files, not CLI stdout.
- **Generation ordering**: `data_global` and `data_linked` depend on `data_local`. `activity_global` depends on `activity_local`. Running dependent types without their prerequisites produces silent empty results.
- **`data_linked` CLI opt-in**: The CLI excludes `data_linked` by default, but this skill always includes it. If running the CLI manually outside this skill, add `--generation-types data_linked`.
