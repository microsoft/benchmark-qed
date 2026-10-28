# Evaluation

Evaluation answers two different questions:

- **Answer quality:** Did the system write a good answer?
- **Retrieval quality:** Did the system find useful evidence?

A system can retrieve the right information but write a poor answer, or write a
convincing answer without retrieving enough evidence. Evaluate both parts when
you want to understand where a RAG system needs improvement.

## Choose an evaluation method

| Your question | Method | What you provide |
|---|---|---|
| Which of two systems gives better answers? | [Pairwise Scoring](pairwise.md) | Two sets of answers to the same questions |
| Which system adds more useful information? | [Differential Pairwise Scoring](differential_pairwise.md) | Two sets of answers to the same questions |
| How well does an answer match a trusted answer? | [Reference-Based Scoring](reference.md) | Generated answers and reference answers |
| Does an answer satisfy specific requirements? | [Assertion-Based Scoring](assertion.md) | Generated answers and per-question assertions |
| Does an answer support a broad claim with specific details? | [Hierarchical Assertion Scoring](hierarchical_assertion.md) | Answers, global assertions, and supporting assertions |
| Did the retriever find evidence for the required facts? | [Chunk Assertion Scoring](chunk_assertion.md) | Retrieved chunks and per-question assertions |
| Were retrieved chunks relevant and representative? | [Retrieval Metrics](retrieval.md) | Retrieved chunks and a relevance reference |
| How good is an answer on a fixed, example-based scale? | [Absolute Scoring](../absolute_scoring.md) | A calibration collection, then new answers |
| How does direct grading compare with calibrated grading? | [Direct Absolute Baseline](direct_absolute.md) | Answers and a written five-level scale |

The direct absolute baseline is available through the notebook and Python API;
it is not a separate AutoE CLI command.

## Keep comparisons fair

Use the same questions, criteria, input data, and judge settings when comparing
systems. Review a sample of the judge's reasoning rather than relying only on
the final numbers.

Repeated trials help reveal variation in model judgments. Statistical
significance testing helps assess differences across a question set, but neither
repetition nor significance proves that the judge is correct.

For assertion comparisons, see [Assertion Significance Testing](../cli/autoe.md#assertion-significance-testing).
For controlling reuse of completed judgments, see
[Evaluation Caches](../cli/cache.md).
