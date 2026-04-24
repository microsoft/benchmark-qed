---
name: benchmark-qed-setup
description: >
  Initialize and configure benchmark-qed workspaces for RAG benchmarking.
  Use when: setting up a new benchmarking project, initializing config files
  for question generation or evaluation, downloading sample datasets,
  or modifying benchmark-qed settings.yaml configuration. Also use when
  the user mentions "benchmark-qed config", workspace setup, or needs to
  prepare a benchmarking environment — even if they don't say "setup" explicitly.
---

# Benchmark-QED Workspace Setup

Initialize workspaces, generate configuration files, download datasets, and manage settings for the benchmark-qed RAG benchmarking tool.

## Prerequisites

benchmark-qed requires Python 3.11+ and uv. Run commands with `uvx` to avoid installing globally:

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed <command>
```

Pin a specific version for reproducibility:
```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed@v1.2.3" benchmark-qed <command>
```

If `uvx` is unavailable, install uv first:
```bash
pip install uv && uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed <command>
```

## Procedure

### Step 1 — Initialize a Workspace

**Option A (Recommended): Interactive wizard**

The interactive wizard guides you through configuration with sensible defaults:

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed init <root_directory>
```

This walks through:
- Config type selection (autoq, autoe_pairwise, autoe_reference, autoe_assertion)
- LLM provider selection with Azure-specific prompts (endpoint, API version)
- Section-by-section customization (press Enter to accept defaults)
- Automatic YAML validation before writing

**Option B: Non-interactive (template-based)**

Generate a static template and edit manually:

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed config init <config_type> <root_directory>
```

**Config types** (pick one):
| Type | Purpose |
|------|---------|
| `autoq` | Question generation (includes all prompt templates) |
| `autoe_pairwise` | Pairwise comparison evaluation |
| `autoe_reference` | Reference-based scoring |
| `autoe_assertion` | Assertion-based scoring |

Example:
```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed config init autoq ./my_workspace
```

This creates:
```
root/
├── .env              # API key placeholder
├── input/            # Place your data here
├── settings.yaml     # Main configuration file
└── prompts/          # LLM prompt templates
```

### Step 2 — Download Sample Data (Optional)

Download sample datasets for testing. This command has an interactive confirmation prompt with no `--yes` flag — use one of these approaches to avoid hanging:

**Bash/Linux/macOS:**
```bash
echo y | uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed data download <dataset> <output_dir>
```

**PowerShell:**
```powershell
"y" | uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed data download <dataset> <output_dir>
```

**Available datasets**: `AP_news`, `podcast`, `example_answers`

### Step 3 — Gather Configuration Choices from the User

Before writing any values into `settings.yaml`, **prompt the user with `ask_user`** to collect the LLM / auth / endpoint settings. Do not guess — these decisions are environment-specific and getting them wrong wastes downstream LLM calls. Use enum/boolean fields whenever possible so the user picks from a known set rather than typing free-form text.

Ask in **a single `ask_user` form** (split into two if the workflow is autoq, since autoq also needs an embedding model). Tailor the follow-up fields based on the provider/auth choice — if the first answer reveals an Azure provider, ask the Azure-only fields in a second form.

#### LLM (chat) fields to collect

| Field | Type | Options / examples | Notes |
|-------|------|--------------------|-------|
| `llm_provider` | enum | `openai.chat`, `azure.openai.chat`, `azure.inference.chat` | See provider table in [references/config-reference.md](references/config-reference.md). |
| `model` | string | `gpt-4.1`, `gpt-4o`, `o3-mini`, an Azure deployment name | For Azure providers this is the **deployment name**, not the base model id. |
| `auth_type` | enum | `api_key` (default), `azure_managed_identity` | `azure_managed_identity` is only valid for `azure.*` providers. |
| `api_key_env_var` | string | `OPENAI_API_KEY` (default), `AZURE_OPENAI_API_KEY`, … | Only ask when `auth_type=api_key`. The skill writes `${VAR_NAME}` into YAML and adds the variable to `.env`. |
| `azure_endpoint` | string (uri) | e.g. `https://my-resource.openai.azure.com/` | Only ask for `azure.*` providers. |
| `api_version` | string | e.g. `2024-06-01` | Only ask for `azure.openai.*` providers. |
| `concurrent_requests` | integer | default `4` | Optional; offer the default. |

#### Embedding fields to collect (autoq only)

Ask the same shape of questions for the embedding model:

| Field | Type | Notes |
|-------|------|-------|
| `embedding_provider` | enum (`openai.embedding`, `azure.openai.embedding`, `azure.inference.embedding`) | Must be an *embedding* provider. |
| `embedding_model` | string | e.g. `text-embedding-3-large`, or an Azure deployment name. |
| Reuse `auth_type` / `api_key_env_var` / `azure_endpoint` / `api_version` from the chat answers unless the user wants different values — ask a yes/no `reuse_chat_auth` boolean first. |

#### Input data fields (autoq only)

| Field | Type | Notes |
|-------|------|-------|
| `dataset_path` | string | Path to CSV/JSON dataset, e.g. `./input/data.csv`. |
| `input_type` | enum (`csv`, `json`) | |
| `text_column` | string | Column/key containing the text content. |

#### Eval-config-specific fields (autoe_*)

Only ask the questions relevant to the chosen `config_type`:
- `autoe_pairwise`: `base.name` + `base.answer_base_path`, plus a list of `others` (each with `name` and `answer_base_path`), and `question_sets`.
- `autoe_reference`: `reference.name` + `reference.answer_base_path`, list of `generated`, and `question_sets`.
- `autoe_assertion`: in single-RAG mode, `generated.name` + `generated.answer_base_path` and `assertions.assertions_path`. In multi-RAG mode (`rag_methods` provided), ask for `input_dir`, `output_dir`, `rag_methods` list, and `question_sets`.

If the user declines a field, fall back to the documented default and call out the assumption in your response.

### Step 4 — Apply the Answers

Use the answers from Step 3 to edit `settings.yaml` and `.env` directly:

```yaml
# LLM configuration (template — substitute values from ask_user answers)
chat_model:
  model: <model from ask_user>
  llm_provider: <llm_provider from ask_user>
  auth_type: <auth_type from ask_user>
  api_key: ${<api_key_env_var>}        # only when auth_type=api_key
  concurrent_requests: <concurrent_requests>
  init_args:                            # only for azure.* providers
    azure_endpoint: <azure_endpoint>
    api_version: "<api_version>"        # azure.openai.* only

# Input data (autoq only)
input:
  dataset_path: <dataset_path>
  input_type: <input_type>
  text_column: <text_column>
```

**Rules when writing the YAML:**
- Omit `api_key` entirely when `auth_type=azure_managed_identity` — do not leave `${OPENAI_API_KEY}` in place.
- Omit `init_args` for non-Azure providers.
- Quote `api_version` (it would otherwise be parsed as a date).
- For `azure_managed_identity`, do **not** add anything to `.env` for that key.
- For `api_key` auth, append `<api_key_env_var>=<placeholder>` to `.env` if the variable is missing, and tell the user to replace the placeholder with their real key before running any command.

For the full set of optional fields, read [references/config-reference.md](references/config-reference.md).

### Step 5 — Review Settings with the User

After writing `settings.yaml`, **show the user the generated configuration** and ask if they want to customize anything. This is critical — the generated config uses sensible defaults, but users often need to tune dataset-specific or environment-specific values.

1. Read the generated `settings.yaml` and display its contents to the user (use `show_file`).
2. Use `ask_user` with a boolean field: *"Would you like to customize any settings before proceeding?"*
3. If the user wants changes, use `ask_user` with a **free-text string field**: *"Describe what you'd like to change"* — let them say it in their own words (e.g., "increase num_questions to 50 for all types", "change the model to gpt-4o", "set trials to 6 and add a custom criterion"). Then apply the requested changes to `settings.yaml`.
4. After applying changes, show the updated file and ask again: *"Any other changes?"* (boolean). Repeat until the user says no.

Do **not** limit the user to predefined sections — they should be able to modify any field in `settings.yaml` by describing what they want.

**Sections the user is most likely to customize** (call these out):
- **autoq**: `num_questions` per type, `num_clusters`, `chunk_size`, assertion settings, `concurrent_requests`
- **autoe_pairwise**: `trials`, `criteria`, `question_sets`
- **autoe_reference**: `score_min`/`score_max`, `trials`
- **autoe_assertion**: `pass_threshold`, `trials`

For the full set of optional fields and best practices, read [references/config-reference.md](references/config-reference.md).

### Step 6 — Validate Configuration

The benchmark-qed CLI validates `settings.yaml` via pydantic at startup, so any missing or malformed fields are reported when you run a command. After applying the answers, run the actual target command (e.g. `benchmark-qed autoq …`) — config errors surface immediately, before any LLM calls.

## Best Practices

See [references/config-reference.md](references/config-reference.md) for detailed best practices covering LLM configuration, prompts, question generation, assertion generation, evaluation, and retrieval.

Key highlights:
- Use `${OPENAI_API_KEY}` env var substitution — never hardcode secrets
- Use `benchmark-qed init` (interactive wizard) to avoid manual YAML errors
- Pin a specific version of benchmark-qed for reproducibility in CI/CD

## Gotchas

- The `data download` command blocks on `typer.confirm()`. Always use `echo y | uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed data download ...` to prevent hanging.
- Environment variables in YAML use `${VAR_NAME}` syntax (resolved at load time via python-dotenv).
- The `.env` file must be in the workspace root directory, not the project root.
- Config types `autoe_pairwise`, `autoe_reference`, and `autoe_assertion` generate different settings.yaml templates — use the correct type for your evaluation method.
- Prompts are copied as `.txt` files using Python `string.Template` syntax (`$variable` or `${variable}`).
- **`prompts_config` vs `prompt_config`**: The non-interactive `config init` for some autoe types generates `prompts_config`, but the runtime expects `prompt_config`. The interactive `benchmark-qed init` wizard avoids this issue. If using `config init`, rename the key if you get validation errors.
