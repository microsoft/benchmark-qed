## CLI Reference

This page documents the command-line interface of the benchmark-qed data download package.

### Download Command

The `data download` command downloads datasets from GitHub and optionally uploads them to cloud storage.

#### Arguments

| Argument | Description |
|---|---|
| `dataset` | The dataset to download. One of: `AP_news`, `podcast`, `example_answers`. |
| `output_dir` | The directory (local) or path prefix (cloud) to save the downloaded dataset. |

#### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--storage-type` | `str` | `None` | Storage backend: `blob` for Azure Blob Storage, `cosmosdb` for Azure Cosmos DB. Omit for local filesystem. |
| `--container-name` | `str` | `None` | The blob container or Cosmos DB container name. Required when `--storage-type` is set. |
| `--account-url` | `str` | `None` | The storage account URL. Uses managed identity (DefaultAzureCredential) for authentication. |
| `--connection-string` | `str` | `None` | The storage connection string. Alternative to `--account-url` for authentication. |
| `--database-name` | `str` | `None` | The Cosmos DB database name. Required when `--storage-type` is `cosmosdb`. |
| `--base-dir` | `str` | `None` | Base prefix in cloud storage. Files are stored under `{base_dir}/{output_dir}/`. If omitted, files are stored under `{output_dir}/` only. |

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
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING"
```

#### Azure Cosmos DB

```bash
benchmark-qed data download AP_news input \
  --storage-type cosmosdb \
  --container-name my-container \
  --database-name my-database \
  --account-url https://myaccount.documents.azure.com
```

#### Path Structure

- **With `--base-dir`**: Files are stored as `{base_dir}/{output_dir}/{file_path}`
  - Example: `my-project/input/2023/11/22/file.json`
- **Without `--base-dir`**: Files are stored as `{output_dir}/{file_path}`
  - Example: `input/2023/11/22/file.json`

This ensures cloud storage mirrors your local directory structure.

::: mkdocs-typer2
    :module: benchmark_qed.data.cli
    :name: config