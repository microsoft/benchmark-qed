## CLI Reference

This page documents the command-line interface of the benchmark-qed data download package.

### Download Command

The `data download` command downloads datasets from GitHub and optionally uploads them to Azure Blob Storage.

!!! note "Supported cloud backends"
    Only **Azure Blob Storage** (`--storage-type blob`) is currently supported.
    Azure Cosmos DB and other backends are **not supported**.

#### Arguments

| Argument | Description |
|---|---|
| `dataset` | The dataset to download. One of: `AP_news`, `podcast`, `example_answers`. |
| `output_dir` | The directory (local) or path prefix (blob) to save the downloaded dataset. |

#### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--storage-type` | `str` | `None` | Storage backend: `blob` for Azure Blob Storage. Omit for local filesystem. |
| `--container-name` | `str` | `None` | The blob container name. Required when `--storage-type` is set. |
| `--account-url` | `str` | `None` | The storage account URL. Uses managed identity (DefaultAzureCredential) for authentication. |
| `--connection-string` | `str` | `None` | The storage connection string. Alternative to `--account-url` for authentication. |
| `--base-dir` | `str` | `None` | Base prefix in blob storage. Files are stored under `{base_dir}/{output_dir}/`. If omitted, files are stored under `{output_dir}/` only. |

#### Local Filesystem

```bash
benchmark-qed data download AP_news ./input
```

The dataset is extracted to the specified `output_dir` (e.g., `./input`).

#### Azure Blob Storage

```bash
benchmark-qed data download AP_news input \
  --storage-type blob \
  --container-name my-container \
  --account-url https://myaccount.blob.core.windows.net \
  --base-dir my-project
```

Or with a connection string:

```bash
benchmark-qed data download AP_news input \
  --storage-type blob \
  --container-name my-container \
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
  --base-dir my-project
```

#### Path Structure

- **With `--base-dir`**: Files are stored as `{base_dir}/{output_dir}/{file_path}`
  - Example: `my-project/input/2023/11/22/file.json`
- **Without `--base-dir`**: Files are stored as `{output_dir}/{file_path}`
  - Example: `input/2023/11/22/file.json`

This ensures blob storage mirrors your local directory structure.

::: mkdocs-typer2
    :module: benchmark_qed.data.cli
    :name: config