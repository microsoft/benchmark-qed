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

3. **Initialize the configuration:**
    ```sh
    uv run benchmark-qed config init autoq .
    ```
    This command creates two files in the `./autoq_test` directory:
    - `.env`: Stores environment variables for the AutoQ pipeline. Open this file and replace `<API_KEY>` with your OpenAI or Azure API key.
    - `settings.yaml`: Contains pipeline settings. Edit this file as needed for your use case.

4. **Generate synthetic queries:**
    ```sh
    uv run benchmark-qed autoq settings.yaml output
    ```
    This will process your input data and save the generated queries in the `output` directory.

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

3. **Create a configuration file for pairwise comparison:**
    ```sh
    uv run benchmark-qed config init autoe_pairwise .
    ```
    This command creates two files in the `./pairwise_test` directory:
    - `.env`: Contains environment variables for the pairwise comparison tests. Open this file and replace `<API_KEY>` with your OpenAI or Azure API key.
    - `settings.yaml`: Contains pipeline settings, which you can modify as needed.

4. **Run the pairwise comparison:**
    ```sh
    uv run benchmark-qed autoe pairwise-scores settings.yaml output
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

3. **Create a configuration file for reference scoring:**
    ```sh
    uv run benchmark-qed config init autoe_reference .
    ```
    This creates two files in the `./reference_test` directory:
    - `.env`: Contains environment variables for the reference scoring pipeline. Open this file and replace `<API_KEY>` with your OpenAI or Azure API key.
    - `settings.yaml`: Contains pipeline settings, which you can modify as needed.

4. **Run the reference scoring:**
    ```sh
    uv run benchmark-qed autoe reference-scores settings.yaml output
    ```
    The results will be saved in the `output` directory.

For detailed instructions on configuring and running AutoE subcommands, please refer to the [AutoE CLI Documentation](cli/autoe.md).

To learn how to use AutoE programmatically, please see the [AutoE Notebook Example](notebooks/autoe.ipynb).

## Diving Deeper
To explore the query synthesis workflow in detail, please see the [AutoQ CLI Documentation](cli/autoq.md) for command-line usage and the [AutoQ Notebook Example](notebooks/autoq.ipynb) for a step-by-step programmatic guide.

For a deeper understanding of AutoE evaluation pipelines, please refer to the [AutoE CLI Documentation](cli/autoe.md) for available commands and the [AutoE Notebook Example](notebooks/autoe.ipynb) for hands-on examples.


