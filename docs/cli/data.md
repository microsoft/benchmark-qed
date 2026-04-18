## CLI Reference

This page documents the command-line interface of the benchmark-qed data download package.

### Download Command

The `data download` command downloads datasets from GitHub and optionally uploads them to cloud storage.

#### Local Filesystem

```bash
benchmark-qed data download AP_news ./input
```

The dataset is extracted to the specified `output_dir` (e.g., `./input`).

#### Cloud Storage (Azure Blob Storage or Cosmos DB)

```bash
benchmark-qed data download AP_news input \
  --storage-type blob \
  --container-name my-container \
  --account-url https://myaccount.blob.core.windows.net \
  --base-dir my-project
```

**Path Structure:**
- **With `--base-dir`**: Files are stored as `{base_dir}/{output_dir}/{file_path}`
  - Example: `my-project/input/2023/11/22/file.json`
- **Without `--base-dir`**: Files are stored as `{output_dir}/{file_path}`
  - Example: `input/2023/11/22/file.json`

This ensures cloud storage mirrors your local directory structure.

::: mkdocs-typer2
    :module: benchmark_qed.data.cli
    :name: config