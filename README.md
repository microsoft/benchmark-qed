# BenchmarkQED

<!-- >
TODO: 
 - Add missing badges
 - Add blog post final url
<-->

👉 [Microsoft Research Blog Post](https://www.microsoft.com/en-us/research/blog/)<br/>

<div align="left">
  <a href="https://github.com/microsoft/benchmark-qed/issues">
    <img alt="GitHub Issues" src="https://img.shields.io/github/issues/microsoft/benchmark-qed">
  </a>
  <a href="https://github.com/microsoft/benchmark-qed/discussions">
    <img alt="GitHub Discussions" src="https://img.shields.io/github/discussions/microsoft/benchmark-qed">
  </a>
</div>

## Overview

BenchmarkQED is a suite of tools designed for automated benchmarking of retrieval-augmented generation (RAG) systems, particularly in the context of answering questions over private datasets. It provides components for query generation, evaluation, and dataset preparation to facilitate rigorous testing.

### AutoQ

AutoQ generates synthetic queries ranging from local to global, facilitating consistent benchmarking across datasets without user customization. 

### AutoE

AutoE evaluates answers based on comprehensiveness, diversity, empowerment, and relevance, utilizing the LLM-as-a-Judge method for scaling evaluations.

### AutoD
The AutoD component ensures consistent dataset sampling and summarization, which aids in creating comparable AutoQ queries and supporting consistent evaluations in AutoE. 

### Datasets

BenchmarkQED also provides access to datasets, including the Behind the Tech podcast transcripts and AP News articles, to support the development of RAG systems and AI question-answering.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
