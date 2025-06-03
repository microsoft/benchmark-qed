# Datasets

BenchmarkQED offers two datasets to facilitate the development and evaluation of Retrieval-Augmented Generation (RAG) systems:

- **Podcast Transcripts:** Contains transcripts from 70 episodes of the [Behind the Tech](https://www.microsoft.com/en-us/behind-the-tech) podcast series. This is an updated version of the dataset featured in the [GraphRAG](https://arxiv.org/abs/2404.16130) paper.
- **AP News:** Includes 1,397 health-related news articles from the Associated Press.

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

You can also find these datasets in the [datasets directory](https://github.com/microsoft/benchmark-qed/tree/main/datasets).