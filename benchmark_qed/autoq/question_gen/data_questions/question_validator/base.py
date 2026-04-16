# Copyright (c) 2025 Microsoft Corporation.
"""Base class for batch question validation with KMeans clustering."""

from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from itertools import starmap
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.cluster import KMeans

from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.config.defaults import LLM_PARAMS, RANDOM_SEED
from benchmark_qed.config.utils import load_template_file

if TYPE_CHECKING:
    from pathlib import Path
    from string import Template

    from benchmark_qed.autoq.data_model.question import Question
    from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)


class BatchQuestionValidator(ABC):
    """
    Base class for batch question validation using KMeans clustering.

    Uses clustering to group similar questions together, enabling duplicate
    detection within batches. Each batch is validated by an LLM that checks
    for quality issues and marks duplicates.

    Subclasses should:
    1. Provide a default validation prompt via `_get_default_prompt_path()`
    2. Optionally customize question formatting via `_format_question_for_validation()`
    """

    def __init__(
        self,
        llm: ChatModel,
        llm_params: dict[str, Any] = LLM_PARAMS,
        validation_prompt: Template | None = None,
        batch_size: int = 15,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        """
        Initialize the BatchQuestionValidator.

        Parameters
        ----------
        llm : ChatModel
            Language model for validation.
        llm_params : dict[str, Any]
            Parameters for the LLM.
        validation_prompt : Template | None
            Custom prompt for batch validation. If None, uses default.
        batch_size : int
            Target number of questions per batch. Default 15.
        random_seed : int
            Random seed for KMeans clustering. Default from config.
        """
        self.llm = llm
        self.llm_params: dict[str, Any] = llm_params.copy()
        self.llm_params["response_format"] = {"type": "json_object"}
        self.batch_size = batch_size
        self.random_seed = random_seed

        # Load validation prompt
        if validation_prompt:
            self.validation_prompt: Template = validation_prompt
        else:
            self.validation_prompt = self._load_default_prompt()

    @abstractmethod
    def _get_default_prompt_path(self) -> Path:
        """
        Return the path to the default validation prompt file.

        Returns
        -------
        Path
            Path to the prompt file.
        """

    def _load_default_prompt(self) -> Template:
        """Load the default prompt from file."""
        return load_template_file(self._get_default_prompt_path())

    def _format_question_for_validation(
        self, idx: int, question: Question
    ) -> dict[str, Any]:
        """
        Format a question for inclusion in the validation prompt.

        Subclasses can override to include additional fields.

        Parameters
        ----------
        idx : int
            Index of the question in the batch.
        question : Question
            The question to format.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the question.
        """
        return {
            "id": idx,
            "text": question.text,
            "category": (
                question.attributes.get("abstract_categories", "unknown")
                if question.attributes
                else "unknown"
            ),
        }

    async def filter_valid_questions(
        self,
        questions: list[Question],
    ) -> list[Question]:
        """
        Validate and return only valid questions.

        Uses KMeans clustering to group similar questions together for batch
        validation, which helps identify duplicates and near-duplicates.

        Parameters
        ----------
        questions : list[Question]
            Questions to validate.

        Returns
        -------
        list[Question]
            Questions that passed validation.
        """
        if not questions:
            return []

        # Calculate number of clusters based on batch size
        num_clusters = max(1, math.ceil(len(questions) / self.batch_size))

        # Cluster questions by embedding for duplicate detection
        clustered_batches = self._cluster_questions_for_validation(
            questions, num_clusters
        )
        log.info(
            "Clustered %s questions into %s batches (target batch size: %s)",
            len(questions),
            len(clustered_batches),
            self.batch_size,
        )

        passed_questions: list[Question] = []
        failed_count = 0
        duplicate_count = 0

        # Process each cluster
        for batch in clustered_batches:
            batch_passed, batch_failed, batch_duplicates = await self._validate_batch(
                batch
            )
            passed_questions.extend(batch_passed)
            failed_count += batch_failed
            duplicate_count += batch_duplicates

        log.info(
            "Batch validation: %s passed, %s failed, %s duplicates removed out of %s total",
            len(passed_questions),
            failed_count,
            duplicate_count,
            len(questions),
        )
        return passed_questions

    async def _validate_batch(
        self,
        batch: list[Question],
    ) -> tuple[list[Question], int, int]:
        """
        Validate a single batch of questions.

        Parameters
        ----------
        batch : list[Question]
            Questions to validate in this batch.

        Returns
        -------
        tuple[list[Question], int, int]
            (passed_questions, failed_count, duplicate_count)
        """
        # Format questions for validation
        questions_for_validation = list(
            starmap(self._format_question_for_validation, enumerate(batch))
        )

        prompt = self.validation_prompt.substitute(
            questions=json.dumps(questions_for_validation, indent=2)
        )

        try:
            messages = [{"role": "system", "content": prompt}]
            response = await self.llm.chat(
                messages=messages,
                **self.llm_params,
            )
            _, parsed = try_parse_json_object(response.output.content)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            log.warning("Batch validation failed: %s, keeping all questions", e)
            return batch, 0, 0

        if not parsed or not isinstance(parsed, dict):
            # Parse failed, keep all questions from this batch
            log.warning(
                "Failed to parse batch validation response, keeping all questions"
            )
            return batch, 0, 0

        results = parsed.get("results", [])

        # Build set of passing IDs and count failures
        passing_ids: set[int] = set()
        failed_count = 0
        duplicate_count = 0

        for result in results:
            if isinstance(result, dict):
                raw_id = result.get("id")
                try:
                    qid = int(raw_id)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    log.warning("Invalid question id: %s, skipping", raw_id)
                    continue
                is_pass = bool(result.get("pass"))
                if is_pass and 0 <= qid < len(batch):
                    passing_ids.add(qid)
                elif not is_pass:
                    reasoning = result.get("reasoning", "unknown")
                    if "duplicate" in reasoning.lower():
                        duplicate_count += 1
                    else:
                        failed_count += 1
                    log.debug(
                        "Question failed batch validation: %s - %s",
                        batch[qid].text[:50] if 0 <= qid < len(batch) else "?",
                        reasoning,
                    )

        # Collect passing questions
        passed_questions = [q for j, q in enumerate(batch) if j in passing_ids]
        return passed_questions, failed_count, duplicate_count

    def _cluster_questions_for_validation(
        self,
        questions: list[Question],
        num_clusters: int,
    ) -> list[list[Question]]:
        """
        Cluster questions using KMeans for batch validation.

        Groups similar questions together so duplicates end up in the same batch.

        Parameters
        ----------
        questions : list[Question]
            Questions to cluster.
        num_clusters : int
            Target number of clusters.

        Returns
        -------
        list[list[Question]]
            List of question batches (clusters).
        """
        # Filter to questions with embeddings
        questions_with_embeddings = [q for q in questions if q.embedding is not None]
        questions_without_embeddings = [q for q in questions if q.embedding is None]

        if len(questions_with_embeddings) < num_clusters:
            # Not enough questions, just return as single batch
            return [questions]

        # Build embedding matrix
        embeddings = np.array([q.embedding for q in questions_with_embeddings])

        # Adjust num_clusters if we have fewer questions
        actual_num_clusters = min(num_clusters, len(questions_with_embeddings))

        # Run KMeans
        kmeans = KMeans(
            n_clusters=actual_num_clusters,
            random_state=self.random_seed,
            n_init=10,  # type: ignore[arg-type]
        )
        labels = kmeans.fit_predict(embeddings)

        # Group questions by cluster
        clusters: dict[int, list[Question]] = {}
        for q, label in zip(questions_with_embeddings, labels, strict=True):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(q)

        # Convert to list of batches
        batches = list(clusters.values())

        # Add questions without embeddings to a separate batch
        if questions_without_embeddings:
            batches.append(questions_without_embeddings)

        return batches
