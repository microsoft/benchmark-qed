## CLI Reference

This page documents the command-line interface of the benchmark-qed configuration package.

### Init Command

The `config init` command generates a starter `settings.yaml` file, prompt templates, and an `.env` file for the specified workflow.

#### Arguments

| Argument | Description |
|---|---|
| `config_type` | The type of configuration to generate. One of: `autoq`, `autoe_pairwise`, `autoe_differential_pairwise`, `autoe_reference`, `autoe_assertion`, `autoe_chunk_assertion`. |
| `root` | The path to the root directory where the configuration will be created. |

#### Options

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `--storage-type` | `-s` | `str` | `local` | Storage setup mode. Use `blob` to scaffold active Azure Blob Storage sections in the generated settings file. Default (`local`) keeps storage config commented out as examples. |
| `--container-name` | | `str` | `None` | The blob container name to pre-fill in the generated storage config. |
| `--account-url` | | `str` | `None` | The storage account URL to pre-fill (uses managed identity for auth). |
| `--connection-string` | | `str` | `None` | The storage connection string to pre-fill (alternative to `--account-url`). |
| `--base-dir` | | `str` | `None` | Base prefix path within the container to pre-fill in storage config. |

#### Local Filesystem (Default)

```bash
benchmark-qed config init autoq ./my_project
```

This creates `./my_project/settings.yaml` with blob storage sections **commented out** as documentation examples. Input is read from local files.

#### Azure Blob Storage

!!! note "Supported cloud backends"
    Only **Azure Blob Storage** (`--storage-type blob`) is currently supported.
    Azure Cosmos DB and other backends are **not supported**.

```bash
benchmark-qed config init autoq ./my_project --storage-type blob --base-dir experiments/run1
```

This creates `./my_project/settings.yaml` with **active** (uncommented) blob storage sections for both input and output. You then fill in your container name and credentials.

You can also pre-fill your storage credentials directly. When credentials (`--account-url` or `--connection-string`) are supplied, the generated `settings.yaml`, `.env`, and `prompts/` are uploaded **directly to your blob container** — no local files are created:

```bash
benchmark-qed config init autoq ./my_project \
  --storage-type blob \
  --container-name my-datasets \
  --account-url https://myaccount.blob.core.windows.net \
  --base-dir experiments/run1
```

Or using a connection string:

```bash
benchmark-qed config init autoq ./my_project \
  --storage-type blob \
  --container-name my-datasets \
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
  --base-dir experiments/run1
```

> **Note:** When `--storage-type blob` is used **without** `--account-url` or `--connection-string`, files are written locally with active blob sections in `settings.yaml` for you to fill in manually.

#### Examples

```bash
# Generate AutoQ config with local filesystem (default)
benchmark-qed config init autoq ./autoq_project

# Generate AutoQ config with active blob storage scaffolding
benchmark-qed config init autoq ./autoq_project -s blob --base-dir experiments/run1

# Generate AutoE pairwise config with blob storage
benchmark-qed config init autoe_pairwise ./pairwise_project --storage-type blob --base-dir experiments/run1

# Generate differential pairwise config with stage-aware caching
benchmark-qed config init autoe_differential_pairwise ./differential_project

# Generate AutoE reference config with blob storage
benchmark-qed config init autoe_reference ./reference_project -s blob --base-dir experiments/run1

# Generate AutoE assertion config with blob storage
benchmark-qed config init autoe_assertion ./assertion_project -s blob --base-dir experiments/run1
```

#### Generated Files

The command creates the following structure in the `root` directory (when writing locally):

```
root/
├── settings.yaml       # Main configuration file
├── .env                # Environment variables (e.g., OPENAI_API_KEY)
├── input/              # Placeholder for input data
└── prompts/            # Prompt template files (copied from package defaults)
```

When `--storage-type blob` is used together with `--account-url` or `--connection-string`, the same `settings.yaml`, `.env`, and `prompts/` files are written **directly to the configured blob container** (under `--base-dir` if provided) instead of locally.

#### Running CLI commands against a blob-stored config

Once the `settings.yaml` and `prompts/` tree live in blob storage, the `autoq` and `autoe` CLI commands accept a `blob://` URI in place of a local path. Pass your blob credentials inline via `--account-url` (managed identity) or `--connection-string`:

```bash
benchmark-qed autoq blob://my-container/experiments/run1/settings.yaml output \
  --account-url https://myaccount.blob.core.windows.net
```

Or with a connection string:

```bash
benchmark-qed autoq blob://my-container/experiments/run1/settings.yaml output \
  --connection-string "<your-connection-string>"
```

The CLI streams `settings.yaml` and **every sibling file under the same prefix** (e.g. `prompts/**/*.txt`, `.env`) into a temporary local directory before loading the config. If neither option is supplied, auth falls back to the `AZURE_STORAGE_CONNECTION_STRING` or `AZURE_STORAGE_ACCOUNT_URL` environment variables. Input data and output destinations still flow through the `input:` / `output_storage:` blocks declared inside `settings.yaml`.

Every command that accepts a config path also accepts `--account-url` and `--connection-string`: `autoq`, `autoq generate-assertions`, `autoe pairwise-scores`, `autoe reference-scores`, `autoe assertion-scores`, `autoe hierarchical-assertion-scores`, `autoe assertion-significance`, `autoe hierarchical-assertion-significance`, `autoe generate-retrieval-reference`, and `autoe retrieval-scores`.

#### Storage Configuration in Generated Settings

When `--storage-type blob` is used:

**For AutoQ (`autoq`):**

- An **input storage** block is added under the `input:` section with `type: blob`, `container_name`, and auth options.
- An **output storage** block is added at the top level (`output_storage:`) for writing results to blob.

**For AutoE (`autoe_pairwise`, `autoe_reference`, `autoe_assertion`):**

- An **input_storage** block is added for reading answers/assertions from blob.
- An **output_storage** block is added for writing scores to blob.

Both blocks include placeholders for `container_name`, `connection_string`, and `account_url` (managed identity).

#### Evaluation Cache Configuration

Generated `autoe_pairwise` settings enable a persistent SQLite cache:

```yaml
cache_config:
  type: sqlite
  storage:
    type: file
    base_dir: .benchmark_qed_cache/pairwise
  database_name: pairwise.sqlite3
```

This cache stores completed question/criterion/trial judgments and allows
interrupted or repeated runs to avoid duplicate LLM calls. Set `type: none` and
`storage: null` to disable it.

The SQLite cache is always local, including when `--storage-type blob` is used
for inputs and outputs. See [Evaluation Caches](cache.md) for supported
backends, cache identity, inspection, and clearing instructions.

Generated `autoe_differential_pairwise` settings use a separate SQLite database
under `.benchmark_qed_cache/differential_pairwise`. Extraction and verdict
stages are cached independently so judge failures can resume from extraction
and criteria changes need only recompute verdicts.

Generated `autoe_reference` settings enable a persistent SQLite cache under
`.benchmark_qed_cache/reference`. Each question/criterion/trial judgment is
stored independently. Set `type: none` with `storage: null` to request fresh
judgments.

Generated `autoe_assertion` settings enable a persistent SQLite cache under
`.benchmark_qed_cache/assertion` for standard single-RAG answer-level scoring.
Each question/assertion/trial judgment is stored independently. This cache is
not used by multi-RAG or hierarchical assertion scoring, and chunk-assertion
scoring uses its own assertion/chunk cache.

::: mkdocs-typer2
    :module: benchmark_qed.cli.init_config
    :name: config