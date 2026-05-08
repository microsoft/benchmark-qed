# Datasets

BenchmarkQED offers two datasets to facilitate the development and evaluation of Retrieval-Augmented Generation (RAG) systems:

- **Podcast Transcripts:** Contains transcripts from 70 episodes of the [Behind the Tech](https://www.microsoft.com/en-us/behind-the-tech) podcast series. This is an updated version of the dataset featured in the [GraphRAG](https://arxiv.org/abs/2404.16130) paper.
- **AP News:** Includes 1,397 health-related news articles from the Associated Press.

## Downloading to Local Filesystem

To download these datasets programmatically, use the following commands:

- **Podcast Transcripts:**
    ```sh
    benchmark-qed data download podcast OUTPUT_DIR
    ```
- **AP News:**
    ```sh
    benchmark-qed data download AP_news OUTPUT_DIR
    ```

Replace `OUTPUT_DIR` with the path to the directory where you want the dataset to be saved.

## Downloading to Azure Blob Storage

You can download datasets directly into Azure Blob Storage by providing storage options:

- **Using managed identity:**
    ```sh
    benchmark-qed data download AP_news input \
      --storage-type blob \
      --container-name my-datasets \
      --account-url https://<account>.blob.core.windows.net
    ```

- **Using a connection string:**
    ```sh
    benchmark-qed data download AP_news input \
      --storage-type blob \
      --container-name my-datasets \
      --connection-string "$AZURE_STORAGE_CONNECTION_STRING"
    ```

The `OUTPUT_DIR` argument (e.g., `input`) becomes the prefix path within the blob container. The dataset zip is downloaded from GitHub, extracted in memory, and each file is uploaded directly to the storage backend.

## Storage Options Reference

!!! note "Supported cloud backends"
    Only **Azure Blob Storage** (`--storage-type blob`) is currently supported.
    Azure Cosmos DB and other backends are **not supported**.

| Option | Description |
|---|---|
| `--storage-type` | Storage backend: `blob` for Azure Blob Storage. Omit for local filesystem. |
| `--container-name` | The blob container name. |
| `--account-url` | The storage account URL (uses managed identity for authentication). |
| `--connection-string` | The storage connection string (alternative to `--account-url`). |

You can also find these datasets in the [datasets directory](https://github.com/microsoft/benchmark-qed/tree/main/datasets).