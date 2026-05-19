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

## Execution Environment

This skill can be invoked from different hosts (a plain terminal/agent, the
benchmark-qed VS Code extension, or the benchmark-qed UI). The caller is
expected to declare which environment it is running in via a marker in the
**initial prompt** of the form:

```
Execution context: <cli | vscode | ui>
```

If no marker is present, assume `cli`.

Use the marker to decide what to say at the end of the flow:

- **`cli`** (default): you may end with the exact CLI command the user
  should run next (e.g. `uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed autoq settings.yaml ./output …`).
- **`vscode`**: instruct the user to use the extension's command palette
  entry or the **Run** code-lens above their config file. Do not paste
  long shell commands.
- **`ui`**: the host has an integrated pipeline runner. Do **not** print
  CLI commands or "now run `benchmark-qed autoq …`" instructions. Your
  closing message should only:
  1. Summarize what was created (workspace path, config type, dataset location).
  2. Always include the note: **"⚠️ Update `.env` with your actual API
     key — unless you are using managed identity (`auth_type:
     azure_managed_identity`), in which case no key is required."**
  3. Stop.

During the rest of the flow the steps are the same regardless of the host
— only the closing guidance changes.

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

**Storage options** for `config init`:
| Option | Description |
|--------|-------------|
| `--storage-type` / `-s` | `local` (default) or `blob`. When `blob`, storage config sections are scaffolded as active YAML (not commented out). |
| `--container-name` | Pre-fill the blob container name in generated storage config. |
| `--account-url` | Pre-fill the account URL (managed-identity auth) in generated storage config. |
| `--connection-string` | Pre-fill the connection string in generated storage config. |
| `--base-dir` | Pre-fill a base prefix path within the container. |

When `--storage-type blob` is combined with `--account-url` or `--connection-string`, the generated config and prompt files are also uploaded directly to the blob container.

Example (blob):
```bash
uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed config init autoq ./my_workspace \
  --storage-type blob \
  --container-name my-datasets \
  --account-url https://myaccount.blob.core.windows.net \
  --base-dir experiments/run1
```

This creates:
```
root/
├── .env              # API key placeholder
├── input/            # Place your data here
├── settings.yaml     # Main configuration file
└── prompts/          # LLM prompt templates
```

### Step 2 — Provide Data (Sample or User-Supplied)

The workspace needs data under `<workspace>/input/` before any benchmark run can succeed. Two paths:

**Option A — Download a sample dataset.** Use this when the user picks `AP_news`, `podcast`, or `example_answers`. The command has an interactive confirmation prompt with no `--yes` flag — use one of these approaches to avoid hanging:

**Bash/Linux/macOS:**
```bash
echo y | uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed data download <dataset> <workspace>/input
```

**PowerShell:**
```powershell
"y" | uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed data download <dataset> <workspace>/input
```

**Available datasets**: `AP_news`, `podcast`, `example_answers`

**Option B — User-supplied data.** When the user picks the "use my own data" / "skip sample data" choice and selects a folder via the file picker (the UI returns an absolute path like `/Users/alice/Desktop/raw_data`), **copy the contents of that folder into `<workspace>/input/`** — do **not** register the source folder as the workspace and do **not** modify the source in place.

Use `cp` (macOS/Linux) or `Copy-Item` (PowerShell). Copy the *contents* (children) of the picked folder, not the folder itself, so the input layout matches what `dataset_path: ./input/...` expects:

**Bash/Linux/macOS:**
```bash
mkdir -p "<workspace>/input"
cp -R "<picked_folder>"/. "<workspace>/input/"
```

**PowerShell:**
```powershell
New-Item -ItemType Directory -Force -Path "<workspace>/input" | Out-Null
Copy-Item -Path "<picked_folder>/*" -Destination "<workspace>/input/" -Recurse -Force
```

After copying, list the new contents of `<workspace>/input/` back to the user so they can confirm the files arrived where expected. From this point on, `dataset_path` in `settings.yaml` must reference paths under `./input/`, not the original source folder.

**Storage options** for `data download`:
| Option | Description |
|--------|-------------|
| `--storage-type` | Set to `blob` to upload the dataset to Azure Blob Storage instead of extracting locally. |
| `--container-name` | The blob container name. |
| `--account-url` | Azure storage account URL (managed-identity auth). |
| `--connection-string` | Azure storage connection string (alternative to `--account-url`). |
| `--base-dir` | Base prefix in blob storage. Files are stored as `{base_dir}/{output_dir}/`. |

Example (download to blob):
```bash
echo y | uvx --from "git+https://github.com/microsoft/benchmark-qed" benchmark-qed data download AP_news datasets \
  --storage-type blob \
  --container-name my-datasets \
  --account-url https://myaccount.blob.core.windows.net
```

### Step 3 — Gather Configuration Choices from the User

Before writing any values into `settings.yaml`, **first ask the user how they want to configure the LLM settings**. Many users prefer to skip the interactive Q&A and just edit `settings.yaml` themselves in the editor.

Use a single `ask_user` with a closed enum:

| Field | Type | Options |
|-------|------|---------|
| `llm_setup_mode` | enum | `customize` (walk me through the LLM/auth/endpoint settings now) · `scaffold_only` (just create `settings.yaml` with defaults and I'll edit it in the editor) |

**Branching:**
- If `llm_setup_mode = scaffold_only`: skip the rest of Step 3 and Step 4. The file already exists from Step 1 with placeholder values (e.g. `${OPENAI_API_KEY}`, default provider). Tell the user which file to open (`<workspace>/settings.yaml`) and which env vars to fill in `<workspace>/.env`, then jump straight to **Step 5 — Review Settings with the User** (or end the flow if they decline a review).
- If `llm_setup_mode = customize`: continue with the prompts below.

Do not guess these decisions — they are environment-specific and getting them wrong wastes downstream LLM calls. Use enum/boolean fields whenever possible so the user picks from a known set rather than typing free-form text.

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
| `embedding_azure_deployment` | string | **Required when `embedding_provider=azure.openai.embedding`.** The Azure deployment name for the embedding model (often the same as `embedding_model`). Written to `embedding_model.init_args.azure_deployment`. |
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

#### Storage fields (all config types, optional)

Ask the user if they want to use Azure Blob Storage for input/output. If yes, collect:

| Field | Type | Notes |
|-------|------|-------|
| `use_blob_storage` | boolean | Whether to configure cloud storage. |
| `storage_container_name` | string | Azure Blob container name (e.g. `my-datasets`). |
| `storage_auth_method` | enum (`connection_string`, `managed_identity`) | How to authenticate to Azure. |
| `storage_connection_string_env_var` | string | Env var name for connection string (default: `AZURE_STORAGE_CONNECTION_STRING`). Only when `storage_auth_method=connection_string`. |
| `storage_account_url` | string (uri) | Storage account URL. Only when `storage_auth_method=managed_identity`. |
| `storage_base_dir` | string | Optional prefix path within the container. |
| `separate_output_container` | boolean | Whether output uses a different container than input. |

If storage is enabled, write the appropriate `input.storage`, `input_storage`, and/or `output_storage` blocks into `settings.yaml`.

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
    azure_deployment: <azure_deployment> # azure.openai.embedding only — REQUIRED

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
- **`azure.openai.embedding` requires `azure_deployment` inside `init_args`** — the runtime calls `init_args.pop("azure_deployment")` and will raise `KeyError: 'azure_deployment'` if it is missing. Always ask the user for it (default to the embedding model name if they don't know) and write it under `embedding_model.init_args.azure_deployment`. Chat models (`azure.openai.chat`) do **not** need this field.
- For `azure_managed_identity`, do **not** add anything to `.env` for that key, and do **not** include any "update .env with your API key" warning in the closing message — managed identity uses the ambient Azure credential, there is no key to fill in.
- For `api_key` auth, append `<api_key_env_var>=<placeholder>` to `.env` if the variable is missing, and tell the user to replace the placeholder with their real key before running any command.
- If the workspace mixes auth types (e.g. `chat_model` uses `api_key` but `embedding_model` uses `azure_managed_identity`), only warn about the keys that actually need to be set — name the specific env vars, do not blanket-say "update your API key".

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

The benchmark-qed CLI validates `settings.yaml` via pydantic at startup, so any missing or malformed fields are reported when the pipeline starts. After applying the answers, close the flow according to the **Execution Environment** rules at the top of this skill (CLI → show the next command; VS Code → point at the extension's Run action; UI → just confirm the workspace is ready).

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
- **`prompt_config` key**: The runtime expects `prompt_config` (singular) for all autoe config types. Both `benchmark-qed init` and `config init` now generate the correct key. If you hand-edit YAML, ensure you use `prompt_config`, not `prompts_config`.
- **`config init --storage-type blob`**: When combined with `--account-url` or `--connection-string`, the command uploads the generated `settings.yaml` and prompt files directly to blob storage. Without those auth options, it only scaffolds the storage YAML sections locally.
- **Blob URI format**: CLI commands accept `blob://<container>/<key>` for config paths. The CLI downloads the config and all sibling files (prompt templates) to a temp directory so relative paths resolve correctly. Credentials can be passed via `--account-url`/`--connection-string` or the environment variables `AZURE_STORAGE_ACCOUNT_URL`/`AZURE_STORAGE_CONNECTION_STRING`.
- **Storage config in YAML**: AutoQ uses `input.storage` (nested under `input`) and `output_storage` (top-level). AutoE uses `input_storage` and `output_storage` (both top-level). When storage is omitted, local filesystem is used.
