# Development Guide

## Requirements

| Name                | Installation                                                 | Purpose                                                                             |
| ------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| Python 3.11+    | [Download](https://www.python.org/downloads/)                | The library is Python-based.                                                        |
| uv              | [Instructions](https://docs.astral.sh/uv/getting-started/installation/) | uv is used for package management and virtualenv management in Python codebases |

## Installing dependencies

```sh
# Install Python dependencies
uv sync
```

## Generating synthetic queries

Follow these steps to generate synthetic queries using AutoQ:

1. **Set up your project directory:**
    ```sh
    mkdir -p ./local/autoq_test
    cd ./local/autoq_test
    ```

2. **Create an `input` folder and add your input data:**
    ```sh
    mkdir ./input
    ```
    Place your input files inside the `./input` directory. To get started, you can use the AP News dataset provided in the [datasets folder](https://github.com/microsoft/benchmark-qed/tree/main/datasets/AP_news/raw_data). To download this example dataset directly into your `input` folder, run:
    ```sh
    uv run benchmark-qed data download AP_news input
    ```
    You can also download directly to Azure Blob Storage. See the [Datasets documentation](datasets.md) for storage options.

3. **Initialize the configuration:**
    ```sh
    uv run benchmark-qed config init autoq .
    ```
    This is the local-filesystem variant.

    Alternative blob variant (choose this instead of the local command above; do not run both):
    ```sh
    uv run benchmark-qed config init autoq . \
        --storage-type blob \
        --container-name my-container \
        --account-url https://<account>.blob.core.windows.net \
        --base-dir autoq_test
    ```
    This command creates two files in the `./autoq_test` directory:
    - `.env`: Stores environment variables for the AutoQ pipeline. Open this file and replace `<API_KEY>` with your OpenAI or Azure API key.
    - `settings.yaml`: Contains pipeline settings. Edit this file as needed for your use case.

    The generated `settings.yaml` includes commented-out sections for configuring Azure Blob Storage as input and output backends. Uncomment and fill in the `storage` section under `input` to read data from blob storage, or the `output_storage` section to write results to blob storage instead of the local filesystem.

4. **Generate synthetic queries:**
    ```sh
    uv run benchmark-qed autoq settings.yaml output
    ```
    This is the local-filesystem variant.

    Alternative blob-stored config variant (choose this instead of the local command above; do not run both):

    ```sh
    uv run benchmark-qed autoq blob://my-container/autoq_test/settings.yaml output \
        --account-url https://<account>.blob.core.windows.net
    ```
    This will process your input data and save the generated queries in the `output` directory.

    By default, AutoQ also generates **assertions** for data-driven queries. Assertions are testable factual statements that can be used to evaluate answer accuracy. You can configure assertion generation in `settings.yaml`:
    ```yaml
    assertions:
      max_assertions: 20  # Set to 0 to disable, or null for unlimited
      enable_validation: true  # Enable to filter low-quality assertions (can be slow)
    ```

## Comparing RAG answer pairs

Follow these steps to compare RAG answer pairs using the pairwise scoring pipeline:

1. **Set up your project directory:**
    ```sh
    mkdir -p ./local/pairwise_test
    cd ./local/pairwise_test
    ```

2. **Create an `input` folder and add your question-answer data:**
    ```sh
    mkdir ./input
    ```
    Copy your RAG answer files into the `./input` directory. To get started, you can use the example RAG answers available in the [example data folder](https://github.com/microsoft/benchmark-qed/tree/main/docs/notebooks/example_answers). To download this example dataset directly into your `input` folder, run:
    ```sh
    uv run benchmark-qed data download example_answers input
    ```
    You can also download directly to Azure Blob Storage. See the [Datasets documentation](datasets.md) for storage options.

3. **Create a configuration file for pairwise comparison:**
    ```sh
    uv run benchmark-qed config init autoe_pairwise .
    ```
    This is the local-filesystem variant.

    Alternative blob variant (choose this instead of the local command above; do not run both):
    ```sh
    uv run benchmark-qed config init autoe_pairwise . \
        --storage-type blob \
        --container-name my-container \
        --account-url https://<account>.blob.core.windows.net \
        --base-dir pairwise_test
    ```
    This command creates two files in the `./pairwise_test` directory:
    - `.env`: Contains environment variables for the pairwise comparison tests. Open this file and replace `<API_KEY>` with your OpenAI or Azure API key.
    - `settings.yaml`: Contains pipeline settings, which you can modify as needed.

    The generated `settings.yaml` includes commented-out `input_storage` and `output_storage` sections for configuring Azure Blob Storage backends.

4. **Run the pairwise comparison:**
    ```sh
    uv run benchmark-qed autoe pairwise-scores settings.yaml output
    ```
    This is the local-filesystem variant.

    Alternative blob-stored config variant (choose this instead of the local command above; do not run both):
    ```sh
    uv run benchmark-qed autoe pairwise-scores blob://my-container/pairwise_test/settings.yaml output \
        --account-url https://<account>.blob.core.windows.net
    ```
    The results will be saved in the `output` directory.

## Scoring RAG answers against reference answers
Follow these steps to score RAG answers against reference answers using example data from the AP news dataset:

1. **Set up your project directory:**
    ```sh
    mkdir -p ./local/reference_test
    cd ./local/reference_test
    ```

2. **Create an `input` folder and add your data:**
    ```sh
    mkdir ./input
    ```
    Copy your RAG answers and reference answers into the `input` directory. To get started, you can use the example RAG answers available in the [example data folder](https://github.com/microsoft/benchmark-qed/tree/main/docs/notebooks/example_answers). To download this example dataset directly into your `input` folder, run:
    ```sh
    uv run benchmark-qed data download example_answers input
    ```
    You can also download directly to Azure Blob Storage. See the [Datasets documentation](datasets.md) for storage options.

3. **Create a configuration file for reference scoring:**
    ```sh
    uv run benchmark-qed config init autoe_reference .
    ```
    This is the local-filesystem variant.

    Alternative blob variant (choose this instead of the local command above; do not run both):
    ```sh
    uv run benchmark-qed config init autoe_reference . \
        --storage-type blob \
        --container-name my-container \
        --account-url https://<account>.blob.core.windows.net \
        --base-dir reference_test
    ```
    This creates two files in the `./reference_test` directory:
    - `.env`: Contains environment variables for the reference scoring pipeline. Open this file and replace `<API_KEY>` with your OpenAI or Azure API key.
    - `settings.yaml`: Contains pipeline settings, which you can modify as needed.

    The generated `settings.yaml` includes commented-out `input_storage` and `output_storage` sections for configuring Azure Blob Storage backends.

4. **Run the reference scoring:**
    ```sh
    uv run benchmark-qed autoe reference-scores settings.yaml output
    ```
    This is the local-filesystem variant.

    Alternative blob-stored config variant (choose this instead of the local command above; do not run both):
    ```sh
    uv run benchmark-qed autoe reference-scores blob://my-container/reference_test/settings.yaml output \
        --account-url https://<account>.blob.core.windows.net
    ```
    The results will be saved in the `output` directory.

## Scoring RAG answers against assertions

Follow these steps to evaluate RAG answers against per-question assertions using answer-level (LLM-as-a-judge) evaluation:

1. **Set up your project directory:**
    ```sh
    mkdir -p ./local/assertion_test
    cd ./local/assertion_test
    ```

2. **Create an `input` folder and add your data:**
    ```sh
    mkdir ./input
    ```
    Copy your RAG answers and assertion files into the `input` directory. To get started, you can use the example RAG answers and generated assertions from the AP news dataset:
    ```sh
    uv run benchmark-qed data download example_answers input
    uv run benchmark-qed data download AP_news input
    ```
    This downloads example answers and assertion files. You can also use your own assertions JSON files or download directly from Azure Blob Storage. See the [Datasets documentation](datasets.md) for storage options.

3. **Create a configuration file for assertion evaluation:**
    ```sh
    uv run benchmark-qed config init autoe_assertion .
    ```
    This is the local-filesystem variant.

    Alternative blob variant (choose this instead of the local command above; do not run both):
    ```sh
    uv run benchmark-qed config init autoe_assertion . \
        --storage-type blob \
        --container-name my-container \
        --account-url https://<account>.blob.core.windows.net \
        --base-dir assertion_test
    ```
    This command creates two files in the `./assertion_test` directory:
    - `.env`: Contains environment variables for the assertion evaluation pipeline. Open this file and replace `<API_KEY>` with your OpenAI or Azure API key.
    - `settings.yaml`: Contains pipeline settings, which you can modify as needed.

    The generated `settings.yaml` includes commented-out `input_storage` and `output_storage` sections for configuring Azure Blob Storage backends.

4. **Run the assertion evaluation:**
    ```sh
    uv run benchmark-qed autoe assertion-scores settings.yaml output
    ```
    This is the local-filesystem variant.

    Alternative blob-stored config variant (choose this instead of the local command above; do not run both):
    ```sh
    uv run benchmark-qed autoe assertion-scores blob://my-container/assertion_test/settings.yaml output \
        --account-url https://<account>.blob.core.windows.net
    ```
    The results will be saved in the `output` directory, including per-assertion scores and per-question summaries.

## Evaluating retrieved chunks against assertions

Chunk-level evaluation scores how well retrieved passages (chunks) support assertions, without waiting for answer synthesis. This is useful for:
- **Fast retrieval evaluation**: Assess retrieval quality before generating complete answers
- **Efficient iteration**: Fix retriever settings without re-running expensive answer generation
- **Persistent caching**: Avoid re-evaluating same (assertion, chunk) pairs across multiple runs

### Prerequisites: Where do chunks come from?

**Chunks** are the retrieved passages from your RAG system. You obtain them by running your retriever (e.g., vector search, BM25, hybrid) on each question. Chunks should include:
- The passage text
- The chunk index (chunk_id)
- The retrieval rank

You'll provide this data to benchmark-qed in one of two formats (see Step 2 below).

### Step-by-step guide

1. **Set up your project directory:**
    ```sh
    mkdir -p ./local/chunk_assertion_test
    cd ./local/chunk_assertion_test
    ```

2. **Prepare your input data:**
    ```sh
    mkdir ./input
    ```
    
    You need:
    - **Assertion file**: Download example data and place it in your input directory:
      ```sh
      uv run benchmark-qed data download example_answers input
      uv run benchmark-qed data download AP_news input
      ```
      Update `assertions_path` in `settings.yaml` to point to your assertions file.
    - **Chunks file**: Created by YOUR retrieval system. The tool always evaluates the
      retrieved **chunks** (not the synthesized answer) against your assertions. Provide
      these chunks in **one of two mutually exclusive formats** — pick whichever matches
      the data you already have, and set the corresponding key in `settings.yaml`:
      - **Format A** → a dedicated chunks file, set via `chunks_path`.
      - **Format B** → an existing answers file with chunks embedded under `retrieval_context`, set via `answer_base_path`.

      Set only ONE of `chunks_path` / `answer_base_path`; leave the other commented out.

    ### Input Format Option A: Separate chunks JSON file

    Use this (`chunks_path`) when you can export retrieval results on their own. Create a
    JSON array (e.g., `input/chunks.json`, `input/vector_rag/retrieved_chunks.json`, or any
    path you prefer) with the retrieved passages per question:
    ```json
    [
      {
        "question_id": "q1",
        "question_text": "What is photosynthesis?",
        "chunks": [
          {
            "text": "Photosynthesis is the process by which plants convert sunlight into chemical energy...",
            "rank": 0,
            "chunk_id": 0
          },
          {
            "text": "In photosynthesis, light energy is captured by chlorophyll molecules...",
            "rank": 1,
            "chunk_id": 5
          }
        ]
      }
    ]
    ```
    
    **Convert your RAG output to this format** using the conversion notebook: [notebooks/chunk_format_conversion/convert_to_chunks_format_a.ipynb](../notebooks/chunk_format_conversion/convert_to_chunks_format_a.ipynb)

    ### Input Format Option B: Embedded in answers JSON

    Use this (`answer_base_path`) when you **already have an answers file** that carries its
    retrieval context. The top-level `answer` text is ignored — only the chunks under
    `retrieval_context[].regions[]` are scored. Note that in this format `chunk_id` must be
    an integer:
    ```json
    [
      {
        "question_id": "q1",
        "question_text": "What is photosynthesis?",
        "answer": "Photosynthesis is a process where plants convert sunlight...",
        "retrieval_context": [
          {
            "regions": [
              {
                "text": "Photosynthesis is the process by which plants...",
                "chunk_id": 0
              }
            ]
          },
          {
            "regions": [
              {
                "text": "In photosynthesis, light energy is captured...",
                "chunk_id": 5
              }
            ]
          }
        ]
      }
    ]
    ```
    
    **Convert your RAG output to this format** using the conversion notebook: [notebooks/chunk_format_conversion/convert_to_chunks_format_b.ipynb](../notebooks/chunk_format_conversion/convert_to_chunks_format_b.ipynb)

3. **Create a configuration file for chunk-level evaluation:**
    ```sh
    uv run benchmark-qed config init autoe_chunk_assertion .
    ```
    This is the local-filesystem variant.

    Alternative blob variant (choose this instead of the local command above; do not run both):
    ```sh
    uv run benchmark-qed config init autoe_chunk_assertion . \
        --storage-type blob \
        --container-name my-container \
        --account-url https://<account>.blob.core.windows.net \
        --base-dir chunk_assertion_test
    ```
    This command creates two files in the `./chunk_assertion_test` directory:
    - `.env`: Contains environment variables. Open this file and replace `<API_KEY>` with your OpenAI or Azure API key.
    - `settings.yaml`: Contains pipeline settings, including:
      - `chunks_path`: Path to your retrieved chunks JSON (for Format A above)
      - `answer_base_path`: Path to answers with embedded retrieval_context (for Format B above)
      - `k_list`: K values to report coverage metrics (e.g., [5, 10, 20, 50])
      - `cache_dir`: Directory for persistent (assertion, chunk) cache

    The generated `settings.yaml` includes commented-out `input_storage` and `output_storage` sections for configuring Azure Blob Storage backends.

4. **Update settings.yaml for your data format:**

    The generated template includes placeholder paths. Update them to match your actual file locations:

    **For Format A (separate chunks file):**
    ```yaml
    generated:
      name: my_retriever
      chunks_path: input/chunks.json  # Update to your chunks file path
      # answer_base_path: (leave commented out for Format A)
    assertions:
      assertions_path: input/activity_global_assertions.json  # Update to your assertions file path
    ```

    **For Format B (embedded in answers):**
    ```yaml
    generated:
      name: my_retriever
      answer_base_path: input/answers.json  # Update to your answers file path
      # chunks_path: (leave commented out for Format B)
    assertions:
      assertions_path: input/activity_global_assertions.json  # Update to your assertions file path
    ```

5. **Run the chunk-level assertion evaluation:**
    ```sh
    uv run benchmark-qed autoe chunk-assertion-scores settings.yaml output
    ```
    This is the local-filesystem variant.

    Alternative blob-stored config variant (choose this instead of the local command above; do not run both):
    ```sh
    uv run benchmark-qed autoe chunk-assertion-scores blob://my-container/chunk_assertion_test/settings.yaml output \
        --account-url https://<account>.blob.core.windows.net
    ```
    The results will be saved in the `output` directory, including:
    - `chunk_assertion_results.json`: Coverage metrics at each k
    - `per_query_metrics_*.json`: Per-question metrics for paired significance testing
    - `debug/`: Detailed per-question evaluation records

**Chunk-level evaluation benefits:**
- **Efficient caching**: Results cached at (assertion, chunk) granularity using SHA256 content-addressing
- **Multi-k reporting**: Coverage, Strict Coverage, and Coverage Strength metrics at each k value
- **Reusable cache**: Re-run with different k values or retriever configs with zero LLM cost on overlapping chunks
- **Coverage metrics**:
  - Coverage: % of assertions with full or partial support in top-k chunks
  - Strict Coverage: % of assertions with full support only
  - Coverage Strength: Average score across all assertions

For detailed instructions on configuring and running AutoE subcommands, please refer to the [AutoE CLI Documentation](cli/autoe.md).

To learn how to use AutoE programmatically, please see the [AutoE Notebook Example](notebooks/autoe.ipynb).

## Diving Deeper
To explore the query synthesis workflow in detail, please see the [AutoQ CLI Documentation](cli/autoq.md) for command-line usage and the [AutoQ Notebook Example](notebooks/autoq.ipynb) for a step-by-step programmatic guide.

For a deeper understanding of AutoE evaluation pipelines, please refer to the [AutoE CLI Documentation](cli/autoe.md) for available commands and the [AutoE Notebook Example](notebooks/autoe.ipynb) for hands-on examples.


