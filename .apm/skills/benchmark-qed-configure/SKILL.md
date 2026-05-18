---
name: benchmark-qed-configure
description: >
  Configure an existing benchmark-qed workspace whose dataset was already
  loaded by the benchmark-UI ("Load Dataset" action). Use when: a dataset
  has just been downloaded/extracted into a workspace directory by the UI
  and the user needs to scaffold or finish `settings.yaml` + `.env` for
  that workspace — without re-downloading the dataset. Also use when the
  user mentions "configure benchmark-qed", "set up settings.yaml for this
  dataset", or wants to pick an LLM provider / config type for a workspace
  that already contains input data.
---

# Benchmark-QED Workspace Configuration (Post-Dataset-Load)

Initialize configuration files, gather LLM / auth / endpoint choices,
and write `settings.yaml` + `.env` for a workspace whose dataset has
**already been loaded by the UI**. This skill is intentionally scoped
to configuration only — dataset downloading is handled by the UI's
"Load Dataset" action and **must not** be re-invoked here.

> **Difference vs. `benchmark-qed-setup`:** this skill assumes the
> workspace root and its `input/` directory already exist and contain
> the dataset. It skips the `benchmark-qed data download` step entirely.

## Prerequisites

benchmark-qed requires Python 3.11+ and uv. Run commands with `uvx`:

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed <command>
```

Pin a specific version for reproducibility:
```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed@v1.2.3" benchmark-qed <command>
```

If `uvx` is unavailable, install uv first:
```bash
pip install uv
```

## Inputs from the UI

When the UI triggers this skill it provides:

| Input | Description |
|-------|-------------|
| `workspace_root` | Absolute path to the workspace directory containing the loaded dataset. |
| `dataset_name` | The dataset that was loaded (e.g. `AP_news`, `podcast`, `example_answers`) — informational only. |
| `dataset_path` (optional) | Absolute or workspace-relative path to the main data file inside `input/` if the UI detected one. |

If any of these are missing, ask the user with `ask_user` before continuing.

## Procedure

### Step 1 — Choose Config Type and Scaffold

Pick the appropriate config type for the user's workflow and scaffold
`settings.yaml` and prompt templates into the existing workspace.

**Config types** (pick one with `ask_user`):

| Type | Purpose |
|------|---------|
| `autoq` | Question generation (includes all prompt templates) |
| `autoe_pairwise` | Pairwise comparison evaluation |
| `autoe_reference` | Reference-based scoring |
| `autoe_assertion` | Assertion-based scoring |

Run the non-interactive template generator against the already-existing
workspace root:

```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" \
  benchmark-qed config init <config_type> <workspace_root>
```

This adds (or overwrites — confirm with the user first if files exist):

```
<workspace_root>/
├── .env              # API key placeholder
├── input/            # Already populated by the UI — leave alone
├── settings.yaml     # Main configuration file
└── prompts/          # LLM prompt templates
```

> Do **not** pass `--storage-type blob` unless the user explicitly asks
> for cloud storage in Step 2. The UI only loads datasets to local
> directories.

### Step 2 — Gather Configuration Choices from the User

Before writing values into `settings.yaml`, **prompt with `ask_user`**
to collect LLM / auth / endpoint settings. Use enum/boolean fields
whenever possible.

Ask in **a single `ask_user` form** (split into two if the workflow is
`autoq`, since `autoq` also needs an embedding model). Tailor follow-up
fields based on the provider/auth choice — if the first answer reveals
an Azure provider, ask the Azure-only fields in a second form.

#### LLM (chat) fields to collect

| Field | Type | Options / examples | Notes |
|-------|------|--------------------|-------|
| `llm_provider` | enum | `openai.chat`, `azure.openai.chat`, `azure.inference.chat` | See provider table in [references/config-reference.md](../benchmark-qed-setup/references/config-reference.md). |
| `model` | string | `gpt-4.1`, `gpt-4o`, `o3-mini`, an Azure deployment name | For Azure providers this is the **deployment name**, not the base model id. |
| `auth_type` | enum | `api_key` (default), `azure_managed_identity` | `azure_managed_identity` is only valid for `azure.*` providers. |
| `api_key_env_var` | string | `OPENAI_API_KEY` (default), `AZURE_OPENAI_API_KEY`, … | Only ask when `auth_type=api_key`. The skill writes `${VAR_NAME}` into YAML and adds the variable to `.env`. |
| `azure_endpoint` | string (uri) | e.g. `https://my-resource.openai.azure.com/` | Only ask for `azure.*` providers. |
| `api_version` | string | e.g. `2024-06-01` | Only ask for `azure.openai.*` providers. |
| `concurrent_requests` | integer | default `4` | Optional; offer the default. |

#### Embedding fields to collect (autoq only)

| Field | Type | Notes |
|-------|------|-------|
| `embedding_provider` | enum (`openai.embedding`, `azure.openai.embedding`, `azure.inference.embedding`) | Must be an *embedding* provider. |
| `embedding_model` | string | e.g. `text-embedding-3-large`, or an Azure deployment name. |
| `reuse_chat_auth` | boolean | Ask first. If yes, reuse `auth_type` / `api_key_env_var` / `azure_endpoint` / `api_version` from the chat answers. Otherwise ask them again. |

#### Input data fields (autoq only)

Use the dataset already loaded by the UI as the default — don't ask the
user to type the path from scratch.

| Field | Type | Notes |
|-------|------|-------|
| `dataset_path` | string | Default to the UI-provided `dataset_path` (or auto-detect the first `.csv` / `.json` under `<workspace_root>/input/`). |
| `input_type` | enum (`csv`, `json`) | Infer from the file extension and confirm. |
| `text_column` | string | Column/key containing the text content. For `AP_news` the conventional value is `body_nitf`; for `podcast` it is `text`. Confirm with the user. |

#### Eval-config-specific fields (autoe_*)

Only ask the questions relevant to the chosen `config_type`:

- `autoe_pairwise`: `base.name` + `base.answer_base_path`, plus a list
  of `others` (each with `name` and `answer_base_path`), and
  `question_sets`.
- `autoe_reference`: `reference.name` + `reference.answer_base_path`,
  list of `generated`, and `question_sets`.
- `autoe_assertion`: in single-RAG mode, `generated.name` +
  `generated.answer_base_path` and `assertions.assertions_path`. In
  multi-RAG mode (`rag_methods` provided), ask for `input_dir`,
  `output_dir`, `rag_methods` list, and `question_sets`.

For all `autoe_*` types, default any answer/output paths to live
**under `<workspace_root>`** so the workspace stays self-contained.

#### Storage fields (all config types, optional)

The UI loads datasets locally, so default `use_blob_storage` to **false**
and only ask the follow-up fields if the user explicitly opts in.

| Field | Type | Notes |
|-------|------|-------|
| `use_blob_storage` | boolean | Default false. |
| `storage_container_name` | string | Azure Blob container name. |
| `storage_auth_method` | enum (`connection_string`, `managed_identity`) | |
| `storage_connection_string_env_var` | string | Default `AZURE_STORAGE_CONNECTION_STRING`. Only when `storage_auth_method=connection_string`. |
| `storage_account_url` | string (uri) | Only when `storage_auth_method=managed_identity`. |
| `storage_base_dir` | string | Optional prefix path within the container. |
| `separate_output_container` | boolean | Whether output uses a different container than input. |

If storage is enabled, write the appropriate `input.storage`,
`input_storage`, and/or `output_storage` blocks into `settings.yaml`.

If the user declines a field, fall back to the documented default and
call out the assumption in your response.

### Step 3 — Apply the Answers

Edit `<workspace_root>/settings.yaml` and `<workspace_root>/.env`
directly using the answers from Step 2.

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

# Input data (autoq only) — point at the UI-loaded dataset
input:
  dataset_path: <dataset_path>
  input_type: <input_type>
  text_column: <text_column>
```

**Rules when writing the YAML:**
- Omit `api_key` entirely when `auth_type=azure_managed_identity` — do
  not leave `${OPENAI_API_KEY}` in place.
- Omit `init_args` for non-Azure providers.
- Quote `api_version` (it would otherwise be parsed as a date).
- For `azure_managed_identity`, do **not** add anything to `.env` for
  that key.
- For `api_key` auth, append `<api_key_env_var>=<placeholder>` to
  `<workspace_root>/.env` if the variable is missing, and tell the user
  to replace the placeholder with their real key before running any
  command.
- Never modify files under `<workspace_root>/input/` — that's the UI's
  responsibility.

For the full set of optional fields, read
[references/config-reference.md](../benchmark-qed-setup/references/config-reference.md).

### Step 4 — Review Settings with the User

After writing `settings.yaml`, **show the user the generated
configuration** and ask if they want to customize anything.

1. Read the generated `<workspace_root>/settings.yaml` and display its
   contents to the user (use `show_file`).
2. Use `ask_user` with a boolean field: *"Would you like to customize
   any settings before proceeding?"*
3. If the user wants changes, use `ask_user` with a **free-text string
   field**: *"Describe what you'd like to change"* — let them say it in
   their own words. Then apply the requested changes to `settings.yaml`.
4. After applying changes, show the updated file and ask again: *"Any
   other changes?"* (boolean). Repeat until the user says no.

Do **not** limit the user to predefined sections — they should be able
to modify any field in `settings.yaml` by describing what they want.

**Sections the user is most likely to customize** (call these out):
- **autoq**: `num_questions` per type, `num_clusters`, `chunk_size`,
  assertion settings, `concurrent_requests`
- **autoe_pairwise**: `trials`, `criteria`, `question_sets`
- **autoe_reference**: `score_min`/`score_max`, `trials`
- **autoe_assertion**: `pass_threshold`, `trials`

### Step 5 — Validate Configuration

The benchmark-qed CLI validates `settings.yaml` via pydantic at
startup, so any missing or malformed fields are reported when you run
a command. Suggest the user kick off the actual workflow from the UI
(Run button) or run the matching CLI command (e.g. `benchmark-qed
autoq <workspace_root>/settings.yaml <workspace_root>/output`) —
config errors surface immediately, before any LLM calls.

## What This Skill Must NOT Do

- **Do not run `benchmark-qed data download`.** The UI's "Load Dataset"
  action already handled this. Re-downloading would overwrite the
  user's data.
- **Do not modify files under `<workspace_root>/input/`.**
- **Do not initialize a workspace at a different root.** Always operate
  on the `workspace_root` provided by the UI.
- **Do not invoke the interactive `benchmark-qed init` wizard.** It
  re-creates the workspace from scratch and conflicts with the UI's
  loaded dataset. Always use the non-interactive `benchmark-qed config
  init <config_type> <workspace_root>` instead.

## Gotchas

- Environment variables in YAML use `${VAR_NAME}` syntax (resolved at
  load time via python-dotenv).
- The `.env` file must be in the workspace root directory, not the
  project root.
- Config types `autoe_pairwise`, `autoe_reference`, and
  `autoe_assertion` generate different `settings.yaml` templates — use
  the correct type for your evaluation method.
- Prompts are copied as `.txt` files using Python `string.Template`
  syntax (`$variable` or `${variable}`).
- **`prompt_config` key**: The runtime expects `prompt_config`
  (singular) for all autoe config types. Both `benchmark-qed init`
  and `config init` now generate the correct key. If you hand-edit
  YAML, ensure you use `prompt_config`, not `prompts_config`.
- **Storage config in YAML**: AutoQ uses `input.storage` (nested under
  `input`) and `output_storage` (top-level). AutoE uses `input_storage`
  and `output_storage` (both top-level). When storage is omitted, local
  filesystem is used.
- If `<workspace_root>/settings.yaml` already exists, ask the user
  whether to overwrite, merge, or abort before running `config init`.

## Best Practices

See
[references/config-reference.md](../benchmark-qed-setup/references/config-reference.md)
for detailed best practices covering LLM configuration, prompts,
question generation, assertion generation, evaluation, and retrieval.

Key highlights:
- Use `${OPENAI_API_KEY}` env var substitution — never hardcode secrets
- Default any new paths under `<workspace_root>` so the workspace stays
  portable
- Pin a specific version of benchmark-qed for reproducibility in CI/CD
