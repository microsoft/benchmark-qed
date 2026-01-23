# Copyright (c) 2025 Microsoft Corporation.
"""Score CLI for generating scores and significance tests for different conditions."""

import asyncio
import json
from itertools import combinations, product
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pandas as pd
import typer
from rich import print as rich_print

from benchmark_qed.autoe.assertion_scores import get_assertion_scores
from benchmark_qed.autoe.config import (
    AssertionConfig,
    PairwiseConfig,
    ReferenceConfig,
    RetrievalReferenceConfig,
    RetrievalScoresConfig,
)
from benchmark_qed.autoe.pairwise_scores import analyze_criteria, get_pairwise_scores
from benchmark_qed.autoe.reference_scores import get_reference_scores
from benchmark_qed.cli.utils import print_df
from benchmark_qed.config.utils import load_config
from benchmark_qed.llm.factory import ModelFactory

app: typer.Typer = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="Evaluate Retrieval-Augmented Generation (RAG) methods.",
)


@app.command()
def pairwise_scores(
    comparison_spec: Annotated[
        Path,
        typer.Argument(help="The path to the JSON file containing the conditions."),
    ],
    output: Annotated[
        Path, typer.Argument(help="The path to the output file for the scores.")
    ],
    *,
    alpha: Annotated[
        float, typer.Option(help="The p-value threshold for the significance test.")
    ] = 0.05,
    exclude_criteria: Annotated[
        list[str] | None,
        typer.Option(help="The criteria to exclude from the scoring."),
    ] = None,
    print_model_usage: Annotated[
        bool,
        typer.Option(help="Whether to print the model usage statistics after scoring."),
    ] = False,
    include_score_id_in_prompt: Annotated[
        bool,
        typer.Option(
            help="Whether to include the score ID in the evaluation prompt for the LLM (might be useful to avoid cached scores)."
        ),
    ] = True,
    question_id_key: Annotated[
        str,
        typer.Option(
            help="The key in the JSON file that contains the question ID. This is used to match questions across different conditions."
        ),
    ] = "question_id",
) -> None:
    """Generate scores for the different conditions provided in the JSON file."""
    if exclude_criteria is None:
        exclude_criteria = []
    config = load_config(comparison_spec, PairwiseConfig)

    config.criteria = [
        criterion
        for criterion in config.criteria
        if criterion.name not in exclude_criteria
    ]

    llm_client = ModelFactory.create_chat_model(config.llm_config)
    output.mkdir(parents=True, exist_ok=True)
    all_results = []

    all_combinations = (
        product([config.base], config.others)
        if config.base
        else combinations(config.others, 2)
    )

    for base, other in all_combinations:
        for question_set in config.question_sets:
            rich_print(f"Scoring {base.name} vs {other.name} for {question_set}")
            if (output / f"{question_set}_{base.name}--{other.name}.csv").exists():
                rich_print(
                    f"{base.name} vs {other.name} for {question_set} already exists. Skipping generation.\n"
                    f"[bold yellow]If you want to generate a new comparison, delete {question_set}_{base.name}--{other.name}.csv from {output}.[/bold yellow]"
                )
                result = pd.read_csv(
                    output / f"{question_set}_{base.name}--{other.name}.csv"
                )
            else:
                result = get_pairwise_scores(
                    llm_client=llm_client,
                    llm_config=config.llm_config,
                    base_name=base.name,
                    other_name=other.name,
                    base_answers=pd.read_json(
                        (base.answer_base_path / f"{question_set}.json"),
                        encoding="utf-8",
                    ),
                    other_answers=pd.read_json(
                        (other.answer_base_path / f"{question_set}.json"),
                        encoding="utf-8",
                    ),
                    criteria=config.criteria,
                    assessment_user_prompt=config.prompt_config.user_prompt.template,
                    assessment_system_prompt=config.prompt_config.system_prompt.template,
                    trials=config.trials,
                    question_id_key=question_id_key,
                    include_score_id_in_prompt=include_score_id_in_prompt,
                )

                result.to_csv(
                    output / f"{question_set}_{base.name}--{other.name}.csv",
                    index=False,
                )
            result["question_set"] = question_set
            all_results.append(result)

    all_results = pd.concat(all_results)
    all_results.to_csv(output / "win_rates.csv", index=False)

    all_results_p_value = analyze_criteria(
        all_results,
        alpha=alpha,
    )

    all_results_p_value.to_csv(output / "winrates_sig_tests.csv", index=False)

    print_df(
        cast(
            pd.DataFrame,
            all_results_p_value[
                [
                    "question_set",
                    "criteria",
                    "base_name",
                    "other_name",
                    "base_mean",
                    "other_mean",
                    "formatted_corrected_p_value",
                ]
            ],
        ),
        "Pairwise Scores Summary",
    )

    if print_model_usage:
        rich_print("Model usage statistics:")
        rich_print(llm_client.get_usage())
    usage_file = output / "model_usage.json"
    usage_file.write_text(json.dumps(llm_client.get_usage()), encoding="utf-8")


@app.command()
def reference_scores(
    comparison_spec: Annotated[
        Path,
        typer.Argument(help="The path to the JSON file containing the configuration."),
    ],
    output: Annotated[
        Path, typer.Argument(help="The path to the output file for the scores.")
    ],
    *,
    exclude_criteria: Annotated[
        list[str] | None,
        typer.Option(help="The criteria to exclude from the scoring."),
    ] = None,
    print_model_usage: Annotated[
        bool,
        typer.Option(help="Whether to print the model usage statistics after scoring."),
    ] = False,
    include_score_id_in_prompt: Annotated[
        bool,
        typer.Option(
            help="Whether to include the score ID in the evaluation prompt for the LLM (might be useful to avoid cached scores)."
        ),
    ] = True,
    question_id_key: Annotated[
        str,
        typer.Option(
            help="The key in the JSON file that contains the question ID. This is used to match questions across different conditions."
        ),
    ] = "question_id",
) -> None:
    """Generate scores for the generated answers provided in the JSON file."""
    if exclude_criteria is None:
        exclude_criteria = []
    config = load_config(comparison_spec, ReferenceConfig)

    config.criteria = [
        criterion
        for criterion in config.criteria
        if criterion.name not in exclude_criteria
    ]

    llm_client = ModelFactory.create_chat_model(config.llm_config)

    for generated in config.generated:
        result = get_reference_scores(
            llm_client=llm_client,
            llm_config=config.llm_config,
            generated_answers=pd.read_json(
                generated.answer_base_path, encoding="utf-8"
            ),
            reference_answers=pd.read_json(
                config.reference.answer_base_path, encoding="utf-8"
            ),
            criteria=config.criteria,
            assessment_user_prompt=config.prompt_config.user_prompt.template,
            assessment_system_prompt=config.prompt_config.system_prompt.template,
            score_min=config.score_min,
            score_max=config.score_max,
            trials=config.trials,
            include_score_id_in_prompt=include_score_id_in_prompt,
            question_id_key=question_id_key,
        )
        output.mkdir(parents=True, exist_ok=True)
        result.to_csv(output / f"reference_scores-{generated.name}.csv", index=False)
        summary_df = cast(
            pd.DataFrame,
            result.drop(
                columns=[
                    "question",
                    "reference_answer",
                    "generated_answer",
                    "reasoning",
                    "trial",
                ]
            )
            .groupby("criteria")
            .agg(list),
        )

        summary_df["mean"] = summary_df["score"].apply(np.mean)
        summary_df["std"] = summary_df["score"].apply(np.std)
        summary_df = summary_df.drop(columns=["score"])
        print_df(
            summary_df.drop(columns=["score_id"]).reset_index(),
            f"Reference Scores Summary for {generated.name}",
        )

    if print_model_usage:
        rich_print("Model usage statistics:")
        rich_print(llm_client.get_usage())
    usage_file = output / "model_usage.json"
    usage_file.write_text(json.dumps(llm_client.get_usage()), encoding="utf-8")


@app.command()
def assertion_scores(
    comparison_spec: Annotated[
        Path,
        typer.Argument(help="The path to the JSON file containing the configuration."),
    ],
    output: Annotated[
        Path, typer.Argument(help="The path to the output file for the scores.")
    ],
    *,
    print_model_usage: Annotated[
        bool,
        typer.Option(help="Whether to print the model usage statistics after scoring."),
    ] = False,
    include_score_id_in_prompt: Annotated[
        bool,
        typer.Option(
            help="Whether to include the score ID in the evaluation prompt for the LLM (might be useful to avoid cached scores)."
        ),
    ] = True,
    question_id_key: Annotated[
        str,
        typer.Option(
            help="The key in the JSON file that contains the question ID. This is used to match questions with assertions."
        ),
    ] = "question_id",
    question_text_key: Annotated[
        str,
        typer.Option(help="The key in the JSON file that contains the question text."),
    ] = "question_text",
    answer_text_key: Annotated[
        str,
        typer.Option(help="The key in the JSON file that contains the answer text."),
    ] = "answer",
    assertions_key: Annotated[
        str,
        typer.Option(
            help="The key in the JSON file that contains the assertions. This should be a list of assertions for each question."
        ),
    ] = "assertions",
) -> None:
    """Generate assertion for the generated answers provided in the JSON file."""
    config = load_config(comparison_spec, AssertionConfig)
    output.mkdir(parents=True, exist_ok=True)

    llm_client = ModelFactory.create_chat_model(config.llm_config)
    assertions = pd.read_json(config.assertions.assertions_path, encoding="utf-8")

    if assertions.loc[:, assertions_key].isna().any():  # type: ignore
        msg = f"Some questions in the assertions file do not have assertions. Please check {config.assertions.assertions_path}, these questions will be skipped."
        rich_print(f"[bold red]{msg}[/bold red]")
    assertions = assertions[~assertions.loc[:, assertions_key].isna()]

    if assertions.loc[:, assertions_key].apply(lambda x: len(x) == 0).any():
        msg = f"Some questions in the assertions file have empty assertions. Please check {config.assertions.assertions_path}, these questions will be skipped."
        rich_print(f"[bold red]{msg}[/bold red]")
    assertions = cast(
        pd.DataFrame,
        assertions[assertions.loc[:, assertions_key].apply(lambda x: len(x) > 0)],
    )

    assertions = (
        assertions.explode(assertions_key)
        .rename(columns={assertions_key: "assertion"})
        .reset_index(drop=True)
    )

    assertion_score = get_assertion_scores(
        llm_client=llm_client,
        llm_config=config.llm_config,
        answers=pd.read_json(config.generated.answer_base_path, encoding="utf-8"),
        assertions=assertions,
        assessment_user_prompt=config.prompt_config.user_prompt.template,
        assessment_system_prompt=config.prompt_config.system_prompt.template,
        trials=config.trials,
        include_score_id_in_prompt=include_score_id_in_prompt,
        question_id_key=question_id_key,
        question_text_key=question_text_key,
        answer_text_key=answer_text_key,
    )

    assertion_score.to_csv(output / "assertion_scores.csv", index=False)

    summary_by_assertion = (
        assertion_score.groupby(["question", "assertion"])
        .agg(score=("score", lambda x: int(x.mean() > 0.5)), scores=("score", list))
        .reset_index()
    )

    summary_by_question = (
        summary_by_assertion.groupby(["question"])
        .agg(
            success=("score", lambda x: (x == 1).sum()),
            fail=("score", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )

    summary_by_assertion["score_mean"] = summary_by_assertion["scores"].apply(
        lambda x: np.mean(x) if len(x) > 0 else 0.0
    )
    summary_by_assertion["score_std"] = summary_by_assertion["scores"].apply(
        lambda x: np.std(x) if len(x) > 0 else 0.0
    )
    summary_by_assertion = summary_by_assertion.drop(columns=["scores"])

    print_df(
        summary_by_question,
        "Assertion Scores Summary by Question",
    )

    failed_assertions: pd.DataFrame = cast(
        pd.DataFrame, summary_by_assertion[summary_by_assertion["score"] == 0]
    )

    failed_assertions = failed_assertions.drop(columns=["score"])

    if len(failed_assertions) > 0:
        print_df(
            failed_assertions,
            f"[bold red]{failed_assertions.shape[0]} Failed Assertions[/bold red]",
        )
        rich_print(
            f"[bold red]{failed_assertions.shape[0]} assertions failed. See {output / 'assertion_scores.csv'} for details.[/bold red]"
        )
    else:
        rich_print("[bold green]All assertions passed.[/bold green]")

    if print_model_usage:
        rich_print("Model usage statistics:")
        rich_print(llm_client.get_usage())
    usage_file = output / "model_usage.json"
    usage_file.write_text(json.dumps(llm_client.get_usage()), encoding="utf-8")


@app.command()
def generate_retrieval_reference(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to the retrieval reference configuration JSON file."),
    ],
    *,
    print_model_usage: Annotated[
        bool,
        typer.Option(help="Whether to print the model usage statistics."),
    ] = False,
) -> None:
    """Generate retrieval reference data (cluster relevance) for a question set.

    This is a one-off operation that creates reference data used by retrieval_scores.
    The reference data identifies which clusters are relevant to each question.

    If clusters_path is provided, pre-computed clusters will be loaded.
    Otherwise, text units will be loaded from text_units_path and clustered.
    """
    import pandas as pd

    from benchmark_qed.autod.data_model.text_unit import TextUnit
    from benchmark_qed.autod.data_processor.embedding import TextEmbedder
    from benchmark_qed.autod.io.text_unit import load_text_units
    from benchmark_qed.autod.sampler.clustering.cluster import TextCluster
    from benchmark_qed.autoe.retrieval_metrics.reference_gen.cluster_relevance import (
        ClusterRelevanceRater,
        save_cluster_references_to_json,
    )
    from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.bing_rater import (
        BingRelevanceRater,
    )
    from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.rationale_rater import (
        RationaleRelevanceRater,
    )

    config = load_config(config_path, RetrievalReferenceConfig)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM client
    llm_client = ModelFactory.create_chat_model(config.llm_config)

    # Initialize relevance rater based on config
    if config.assessor_type == "bing":
        relevance_rater = BingRelevanceRater(
            llm_client=llm_client,
            llm_config=config.llm_config,
            cache_dir=config.cache_dir,
            cache_enabled=config.cache_dir is not None,
        )
        rich_print("Using BingRelevanceRater (UMBRELA DNA prompt)")
    else:
        relevance_rater = RationaleRelevanceRater(
            llm_client=llm_client,
            llm_config=config.llm_config,
            cache_dir=config.cache_dir,
            cache_enabled=config.cache_dir is not None,
        )
        rich_print("Using RationaleRelevanceRater (structured JSON response)")

    # Initialize embedding model and text embedder
    embedding_model = ModelFactory.create_embedding_model(config.embedding_config)
    embedder = TextEmbedder(embedding_model)

    # Load corpus data (either pre-computed clusters or text units to cluster)
    corpus: list[TextCluster] | list[TextUnit]

    if config.clusters_path is not None:
        # Load pre-computed clusters
        rich_print(f"Loading pre-computed clusters from {config.clusters_path}...")
        with config.clusters_path.open(encoding="utf-8") as f:
            clusters_data = json.load(f)

        clusters = []
        for cluster_data in clusters_data:
            text_units = [
                TextUnit(
                    id=tu.get("id", ""),
                    short_id=tu.get("short_id", tu.get("id", "")),
                    text=tu.get("text", ""),
                    text_embedding=tu.get("text_embedding"),
                )
                for tu in cluster_data.get("text_units", [])
            ]
            cluster = TextCluster(
                cluster_id=cluster_data["cluster_id"],
                text_units=text_units,
            )
            clusters.append(cluster)

        rich_print(f"Loaded {len(clusters)} pre-computed clusters")
        corpus = clusters
    else:
        # Load text units and let ClusterRelevanceRater perform clustering
        rich_print(f"Loading text units from {config.text_units_path}...")

        suffix = config.text_units_path.suffix.lower()
        if suffix == ".parquet":
            text_df = pd.read_parquet(config.text_units_path)
        elif suffix == ".csv":
            text_df = pd.read_csv(config.text_units_path)
        elif suffix in {".json", ".jsonl"}:
            # lines=True for JSONL (one JSON object per line)
            text_df = pd.read_json(config.text_units_path, lines=(suffix == ".jsonl"))
        else:
            msg = f"Unsupported file format: {suffix}. Supported: .parquet, .csv, .json, .jsonl"
            raise ValueError(msg)

        # Get field mappings from config
        fields = config.text_unit_fields

        # Validate required columns exist
        if fields.id_col not in text_df.columns:
            msg = f"Required column '{fields.id_col}' not found. Available: {list(text_df.columns)}"
            raise ValueError(msg)
        if fields.text_col not in text_df.columns:
            msg = f"Required column '{fields.text_col}' not found. Available: {list(text_df.columns)}"
            raise ValueError(msg)

        # Check optional columns and warn if specified but missing
        if fields.short_id_col and fields.short_id_col not in text_df.columns:
            rich_print(f"[yellow]Column '{fields.short_id_col}' not found, will auto-generate short_id[/yellow]")
        if fields.embedding_col and fields.embedding_col not in text_df.columns:
            rich_print(f"[yellow]Column '{fields.embedding_col}' not found, will generate embeddings[/yellow]")

        text_units = load_text_units(
            text_df,
            id_col=fields.id_col,
            text_col=fields.text_col,
            short_id_col=fields.short_id_col if fields.short_id_col and fields.short_id_col in text_df.columns else None,
            embedding_col=fields.embedding_col if fields.embedding_col and fields.embedding_col in text_df.columns else None,
        )
        rich_print(f"Loaded {len(text_units)} text units")

        # Check if embeddings exist and generate if needed
        units_with_embeddings = sum(1 for tu in text_units if tu.text_embedding is not None)
        if units_with_embeddings < len(text_units):
            units_without = len(text_units) - units_with_embeddings
            rich_print(f"[yellow]{units_without}/{len(text_units)} text units missing embeddings. Generating...[/yellow]")

            async def embed_text_units() -> list[TextUnit]:
                return await embedder.embed_batch(text_units=text_units, batch_size=32)

            text_units = asyncio.run(embed_text_units())
            rich_print(f"[green]Embedded {len(text_units)} text units[/green]")
        else:
            rich_print(f"All {len(text_units)} text units have embeddings")

        corpus = text_units

    # Initialize cluster relevance rater
    # Will cluster text units if needed, or use pre-computed clusters
    cluster_rater = ClusterRelevanceRater(
        text_embedder=embedder,
        relevance_rater=relevance_rater,
        corpus=corpus,
        semantic_neighbors=config.semantic_neighbors,
        centroid_neighbors=config.centroid_neighbors,
        num_clusters=config.num_clusters,
    )

    rich_print(f"Cluster relevance rater initialized with {len(cluster_rater.clusters)} clusters")

    # Load questions
    rich_print(f"Loading questions from {config.questions_path}...")
    with config.questions_path.open(encoding="utf-8") as f:
        questions_data = json.load(f)

    # Limit questions if max_questions is set
    if config.max_questions is not None and config.max_questions < len(questions_data):
        questions_data = questions_data[:config.max_questions]
        rich_print(f"Limited to {len(questions_data)} questions (max_questions={config.max_questions})")
    else:
        rich_print(f"Loaded {len(questions_data)} questions")

    # Process questions using batch assessment
    from benchmark_qed.autoq.data_model.question import Question

    questions = [
        Question(
            id=q.get("question_id", q.get("id", str(i))),
            text=q.get("text", q.get("question_text", "")),
        )
        for i, q in enumerate(questions_data)
    ]

    async def run_batch_assessment() -> list:
        return await cluster_rater.assess_batch(questions)

    rich_print(f"Assessing cluster relevance for {len(questions)} questions...")
    results = asyncio.run(run_batch_assessment())

    # Save results with clusters
    output_file = config.output_dir / "reference.json"
    save_cluster_references_to_json(
        results,
        output_file,
        include_clusters=True,
        clusters=cluster_rater.clusters,
    )

    rich_print(f"[green]Saved reference data to {output_file}[/green]")

    # Print cache stats
    cache_stats = relevance_rater.get_cache_stats()
    if cache_stats.get("caching_enabled"):
        rich_print(f"Cache stats: {cache_stats['cache_hits']} hits, {cache_stats['cache_misses']} misses")

    if print_model_usage:
        rich_print("Model usage statistics:")
        rich_print(llm_client.get_usage())

    usage_file = config.output_dir / "model_usage.json"
    usage_file.write_text(json.dumps(llm_client.get_usage()), encoding="utf-8")


@app.command()
def retrieval_scores(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to the retrieval scores configuration JSON file."),
    ],
    *,
    print_model_usage: Annotated[
        bool,
        typer.Option(help="Whether to print the model usage statistics."),
    ] = False,
    max_concurrent: Annotated[
        int,
        typer.Option(help="Maximum concurrent relevance assessments."),
    ] = 8,
) -> None:
    """Evaluate retrieval metrics (precision, recall, fidelity) for RAG methods.

    Compares multiple RAG methods on retrieval quality metrics and runs
    statistical significance tests.
    """
    from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.rationale_rater import (
        RationaleRelevanceRater,
    )
    from benchmark_qed.autoe.retrieval_metrics.scoring.fidelity import FidelityMetric
    from benchmark_qed.autoe.retrieval_scores import (
        load_clusters_from_json,
        run_retrieval_evaluation,
    )

    config = load_config(config_path, RetrievalScoresConfig)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse fidelity metric
    fidelity_metric = (
        FidelityMetric.JENSEN_SHANNON
        if config.fidelity_metric == "js"
        else FidelityMetric.TOTAL_VARIATION
    )

    # Initialize LLM client
    llm_client = ModelFactory.create_chat_model(config.llm_config)

    # Initialize relevance rater with caching
    relevance_rater = RationaleRelevanceRater(
        llm_client=llm_client,
        llm_config=config.llm_config,
        cache_dir=config.cache_dir,
        cache_enabled=config.cache_dir is not None,
    )

    # Load clusters
    rich_print(f"Loading clusters from {config.clusters_path}...")
    clusters = load_clusters_from_json(config.clusters_path)
    rich_print(f"Loaded {len(clusters)} clusters")

    # Prepare RAG methods list
    rag_methods = [
        {"name": method.name, "retrieval_results_path": str(method.retrieval_results_path)}
        for method in config.rag_methods
    ]

    # Run evaluation
    asyncio.run(
        run_retrieval_evaluation(
            relevance_rater=relevance_rater,
            rag_methods=rag_methods,
            question_sets=config.question_sets,
            reference_dir=config.reference_dir,
            clusters=clusters,
            output_dir=config.output_dir,
            relevance_threshold=config.relevance_threshold,
            context_id_key=config.context_id_key,
            context_text_key=config.context_text_key,
            run_significance_test=config.run_significance_test,
            significance_alpha=config.significance_alpha,
            significance_correction=config.significance_correction,
            fidelity_metric=fidelity_metric,
            max_concurrent=max_concurrent,
        )
    )

    rich_print(f"\n[green]Results saved to {config.output_dir}[/green]")

    # Print cache stats
    cache_stats = relevance_rater.get_cache_stats()
    if cache_stats.get("caching_enabled"):
        rich_print(
            f"Cache stats: {cache_stats['cache_hits']} hits, "
            f"{cache_stats['cache_misses']} misses "
            f"({cache_stats['hit_rate_percent']}% hit rate)"
        )

    if print_model_usage:
        rich_print("Model usage statistics:")
        rich_print(llm_client.get_usage())

    usage_file = config.output_dir / "model_usage.json"
    usage_file.write_text(json.dumps(llm_client.get_usage()), encoding="utf-8")
