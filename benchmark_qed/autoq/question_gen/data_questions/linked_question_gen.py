# Copyright (c) 2025 Microsoft Corporation.
"""Data-linked question generation module.

Generate multi-hop style questions by combining local questions that share named entities.
Similar to HotpotQA bridge questions but using entities as the linking mechanism.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.asyncio import tqdm_asyncio

import benchmark_qed.config.defaults as defs
from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.autod.sampler.sampling.mmr_sampler import MMRTextSampler
from benchmark_qed.autoq.data_model.enums import QuestionType
from benchmark_qed.autoq.data_model.question import Question
from benchmark_qed.autoq.prompts import data_questions as prompts_data_questions
from benchmark_qed.autoq.prompts.data_questions import linked_questions
from benchmark_qed.autoq.question_gen.base import BaseQuestionGen, QuestionGenResult
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.local_claim_assertion_gen import (
    LocalClaimAssertionGenerator,
)
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.ranking import (
    calculate_dense_ranks,
)
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.validator import (
    AssertionValidator,
)
from benchmark_qed.autoq.question_gen.data_questions.question_validator import (
    LinkedQuestionValidator,
)
from benchmark_qed.config.utils import load_template_file

if TYPE_CHECKING:
    from string import Template

    import tiktoken

    from benchmark_qed.autod.data_processor.embedding import TextEmbedder
    from benchmark_qed.autoq.config import AssertionConfig, AssertionPromptConfig
    from benchmark_qed.autoq.sampler.question_sampler import QuestionSampler
    from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

DATA_LINKED_PROMPTS_PATH = Path(linked_questions.__file__).parent
ASSERTION_PROMPTS_PATH = Path(prompts_data_questions.__file__).parent / "assertions"


@dataclass
class EntityQuestionContext:
    """Data class for storing the context for generating entity questions."""

    entity: str
    """The shared entity name."""

    local_questions: list[Question]
    """The local questions that share this entity."""

    claims: list[dict[str, Any]]
    """Combined claims from local questions with source info."""

    max_questions_to_generate: int = 2
    """Maximum questions to generate for this entity group."""


@dataclass
class ClaimWithSource:
    """Claim with source question tracking."""

    claim_id: str
    statement: str
    score: int
    sources: list[dict[str, Any]] = field(default_factory=list)
    source_question_id: str = ""


@dataclass
class QuestionFilterStats:
    """Statistics for tracking filtered questions during generation."""

    total_raw: int = 0
    accepted: int = 0
    skipped_invalid_claims: int = 0
    skipped_few_claims: int = 0
    skipped_low_quality: int = 0
    skipped_single_document: int = 0
    skipped_parse_error: int = 0
    skipped_failed_validation: int = 0

    # Track filtered questions for debugging
    filtered_questions: list[dict[str, Any]] = field(default_factory=list)

    def add_filtered(self, question: dict[str, Any], reason: str, details: str = "") -> None:
        """Add a filtered question to the log."""
        self.filtered_questions.append({
            "text": question.get("text", "")[:200],
            "question_type": question.get("question_type", "unknown"),
            "reason": reason,
            "details": details,
            "quality": question.get("quality", {}),
            "validation": question.get("validation", {}),
        })

    def log_summary(self) -> None:
        """Log a summary of filtering statistics."""
        total_skipped = (
            self.skipped_invalid_claims
            + self.skipped_few_claims
            + self.skipped_low_quality
            + self.skipped_single_document
            + self.skipped_parse_error
            + self.skipped_failed_validation
        )
        if total_skipped > 0:
            log.info(
                "Question filtering summary: %d accepted, %d skipped "
                "(invalid_claims=%d, few_claims=%d, low_quality=%d, single_doc=%d, parse_error=%d, failed_validation=%d)",
                self.accepted,
                total_skipped,
                self.skipped_invalid_claims,
                self.skipped_few_claims,
                self.skipped_low_quality,
                self.skipped_single_document,
                self.skipped_parse_error,
                self.skipped_failed_validation,
            )


class DataLinkedQuestionGen(BaseQuestionGen):
    """
    Generate data-linked questions from local questions sharing named entities.

    Creates harder, multi-hop style questions by combining information from
    multiple local questions that mention the same entity. Similar to HotpotQA
    bridge questions but using named entities as the linking mechanism.

    The pipeline:
    1. Group local questions by shared named entities
    2. Generate entity questions using claims as context
    3. Generate assertions using map + dedupe approach
    4. Validate assertions against sources

    Supports optional assertion generation using LocalClaimAssertionGenerator.
    """

    def __init__(
        self,
        llm: ChatModel,
        text_embedder: TextEmbedder,
        local_questions: list[Question],
        token_encoder: tiktoken.Encoding | None = None,
        question_sampler: QuestionSampler | None = None,
        assertion_config: AssertionConfig | None = None,
        assertion_prompt_config: AssertionPromptConfig | None = None,
        llm_params: dict[str, Any] = defs.LLM_PARAMS,
        json_mode: bool = True,
        generation_user_prompt: Template | None = None,
        concurrent_coroutines: int = 32,
        random_seed: int = defs.RANDOM_SEED,
        min_questions_per_entity: int = 2,
        max_questions_per_entity: int = 10,
        min_quality_score: int = 4,
        question_types: list[str] | None = None,
        enable_batch_validation: bool = True,  # Run batch validation to filter bad questions
        mmr_lambda: float = 0.7,  # MMR trade-off: 0=diversity, 1=quality
        type_balance_weight: float = 0.5,  # Type-balance penalty in MMR (0=ignore, 1=strong)
    ) -> None:
        if assertion_config is None:
            from benchmark_qed.autoq.config import AssertionConfig
            assertion_config = AssertionConfig()
        if assertion_prompt_config is None:
            from benchmark_qed.autoq.config import AssertionPromptConfig
            assertion_prompt_config = AssertionPromptConfig()

        # Default to bridge questions primarily, with comparison and intersection as secondary options
        self.question_types = question_types or ["bridge", "comparison", "intersection", "temporal"]

        self.assertion_config = assertion_config
        self.random_seed = random_seed
        self.min_questions_per_entity = min_questions_per_entity
        self.max_questions_per_entity = max_questions_per_entity
        self.min_quality_score = min_quality_score
        self.enable_batch_validation = enable_batch_validation
        self.mmr_lambda = mmr_lambda
        self.type_balance_weight = type_balance_weight

        # Create default MMR sampler if none provided
        if question_sampler is None:
            question_sampler = MMRTextSampler(
                random_seed=random_seed,
                lambda_param=mmr_lambda,
            )
        else:
            question_sampler.random_seed = self.random_seed

        super().__init__(llm, llm_params, question_sampler)
        self.text_embedder = text_embedder
        self.token_encoder = token_encoder

        # Assertion generation setup
        self.assertion_generator: LocalClaimAssertionGenerator | None = None
        self.assertion_validator: AssertionValidator | None = None
        linked_assertion_config = assertion_config.linked
        max_assertions = linked_assertion_config.max_assertions
        if max_assertions is None or max_assertions > 0:
            # Create validator if validation is enabled
            validator = None
            if linked_assertion_config.enable_validation:
                validator = AssertionValidator(
                    llm=llm,
                    llm_params=llm_params,
                    min_criterion_score=linked_assertion_config.min_validation_score,
                    # Use local validation prompt (linked assertions are fact-focused)
                    validation_prompt=assertion_prompt_config.local_validation_prompt.template,
                    concurrent_validations=linked_assertion_config.concurrent_llm_calls,
                )
                self.assertion_validator = validator

            self.assertion_generator = LocalClaimAssertionGenerator(
                llm=llm,
                llm_params=llm_params,
                max_assertions=max_assertions,
                validator=validator,
                system_prompt=assertion_prompt_config.local_assertion_gen_prompt.template,
                max_concurrent_questions=linked_assertion_config.max_concurrent_questions,
            )

        self.json_mode = json_mode
        if json_mode:
            self.llm_params["response_format"] = {"type": "json_object"}
        else:
            self.llm_params.pop("response_format", None)

        # Question validator setup
        self.question_validator: LinkedQuestionValidator | None = None
        if enable_batch_validation:
            self.question_validator = LinkedQuestionValidator(
                llm=llm,
                llm_params=llm_params,
                batch_size=15,
                random_seed=random_seed,
            )

        # Load prompts for each question type
        self.system_prompts: dict[str, Template] = {}

        # Bridge questions
        self.system_prompts["bridge"] = load_template_file(
            DATA_LINKED_PROMPTS_PATH / "bridge_question_system_prompt.txt"
        )

        # Temporal questions (sequence/timing of events)
        self.system_prompts["temporal"] = load_template_file(
            DATA_LINKED_PROMPTS_PATH / "temporal_question_system_prompt.txt"
        )

        # Comparison questions
        self.system_prompts["comparison"] = load_template_file(
            DATA_LINKED_PROMPTS_PATH / "comparison_question_system_prompt.txt"
        )

        # Intersection questions
        self.system_prompts["intersection"] = load_template_file(
            DATA_LINKED_PROMPTS_PATH / "intersection_question_system_prompt.txt"
        )

        self.generation_user_prompt: Template = (
            generation_user_prompt
            or load_template_file(
                DATA_LINKED_PROMPTS_PATH / "linked_question_user_prompt.txt"
            )
        )

        self.local_questions = local_questions
        self.concurrent_coroutines = concurrent_coroutines
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(
            self.concurrent_coroutines
        )

    async def agenerate(
        self,
        num_questions: int = defs.NUM_QUESTIONS,
        oversample_factor: float = defs.OVERSAMPLE_FACTOR,
    ) -> QuestionGenResult:
        """Async function to generate entity questions from local questions."""
        num_candidate_questions = math.ceil(num_questions * oversample_factor)

        # Step 1: Group local questions by shared entities
        entity_contexts = self._generate_entity_contexts()

        if not entity_contexts:
            log.warning("No valid entity groups found")
            return QuestionGenResult(
                selected_questions=[],
                candidate_questions=[],
            )

        # Step 2: Calculate questions per entity proportionally by claim count
        # Total candidates divided by question types, then distributed by claims
        num_question_types = len(self.question_types)
        candidates_per_type = math.ceil(num_candidate_questions / num_question_types)

        total_claims = sum(len(ctx.claims) for ctx in entity_contexts)
        for ctx in entity_contexts:
            claim_count = len(ctx.claims)
            if claim_count < 2:
                ctx.max_questions_to_generate = 0  # Can't do multi-hop with < 2 claims
            else:
                # Distribute candidates_per_type across entities by claim proportion
                # Then multiply by number of types (each entity generates all types)
                proportion = claim_count / total_claims if total_claims > 0 else 0
                questions_for_entity = math.ceil(candidates_per_type * proportion)
                # Each entity generates up to this many per type, times num types
                ctx.max_questions_to_generate = max(1, questions_for_entity) * num_question_types

        total_max = sum(ctx.max_questions_to_generate for ctx in entity_contexts)
        log.info(
            "Generated %s entity contexts from %s local questions "
            "(total %s claims, targeting ~%s candidates: %s per type x %s types)",
            len(entity_contexts),
            len(self.local_questions),
            total_claims,
            total_max,
            candidates_per_type,
            num_question_types,
        )
        log.info("Question types to generate: %s", self.question_types)

        # Step 3: Generate questions for each entity context
        stats = QuestionFilterStats()
        results: list[Question] = []
        for i in range(0, len(entity_contexts), self.concurrent_coroutines):
            batch = entity_contexts[i : i + self.concurrent_coroutines]
            log.info(
                "Processing entity groups %s to %s of %s...",
                i,
                min(i + self.concurrent_coroutines, len(entity_contexts)),
                len(entity_contexts),
            )
            batch_results = await tqdm_asyncio.gather(*[
                self._agenerate_questions_for_entity(context, stats)
                for context in batch
            ])
            batch_questions = [
                question for result in batch_results for question in result
            ]
            results.extend(batch_questions)

        # Log filtering summary
        stats.log_summary()

        # Track pipeline stats
        pipeline_stats: dict[str, Any] = {
            "entity_groups": len(entity_contexts),
            "target_candidates": total_max,
            "llm_returned": stats.total_raw,
            "after_generation_filter": stats.accepted,
            "generation_filter_stats": {
                "skipped_invalid_claims": stats.skipped_invalid_claims,
                "skipped_few_claims": stats.skipped_few_claims,
                "skipped_low_quality": stats.skipped_low_quality,
                "skipped_single_document": stats.skipped_single_document,
                "skipped_parse_error": stats.skipped_parse_error,
                "skipped_failed_validation": stats.skipped_failed_validation,
            },
            "filtered_samples": stats.filtered_questions[:20],  # Sample of filtered questions for debugging
            "generated": len(results),
        }

        log.info(
            "Generated %s candidate questions from %s entity groups",
            len(results),
            len(entity_contexts),
        )

        # Step 3: Generate embeddings for questions (needed for MMR selection)
        results = await self._ensure_embeddings(results)
        # Step 4: Compute retrieval difficulty for each question (compares to source claims)
        await self._compute_retrieval_difficulty(results)
        # Step 5: Compute combined score using RRF (quality + difficulty)
        self._compute_combined_score(results)

        # Step 5.5: Run batch validation to filter out bad questions
        # This is cheaper than assertion generation, so do it first
        if self.question_validator is not None:
            pre_validation_count = len(results)
            results = await self.question_validator.filter_valid_questions(results)
            pipeline_stats["after_batch_validation"] = len(results)
            pipeline_stats["batch_validation_filtered"] = pre_validation_count - len(results)
            log.info(
                "After batch validation: %s questions remain (filtered %s)",
                len(results),
                pre_validation_count - len(results),
            )
        else:
            pipeline_stats["after_batch_validation"] = len(results)
            pipeline_stats["batch_validation_filtered"] = 0

        # Step 6: Generate assertions for ALL candidates (before selection)
        max_assertions = (
            self.assertion_config.linked.max_assertions
            if self.assertion_config
            else None
        )
        invalid_assertions: list[dict[str, Any]] = []
        if (
            max_assertions is None or max_assertions > 0
        ) and self.assertion_generator is not None:
            log.info(
                "Generating assertions for %s candidate questions", len(results)
            )
            invalid_assertions = await self._generate_assertions_for_questions(results)
            if invalid_assertions:
                log.info(
                    "Collected %s invalid assertions for review",
                    len(invalid_assertions),
                )

        # Step 7: Filter out questions with no valid assertions
        questions_with_assertions = [
            q for q in results
            if q.attributes and q.attributes.get("assertion_count", 0) > 0
        ]
        filtered_out = len(results) - len(questions_with_assertions)
        pipeline_stats["after_assertion_filter"] = len(questions_with_assertions)
        pipeline_stats["assertion_filter_removed"] = filtered_out
        log.info(
            "Filtered to %s questions with valid assertions (removed %s with 0 valid assertions)",
            len(questions_with_assertions),
            filtered_out,
        )

        # Step 8: Recompute combined score with assertion_count included
        # Now that we have assertion counts, include them in ranking
        self._compute_combined_score(questions_with_assertions)

        # Step 9: Select best questions from filtered set
        final_questions = self.select(
            candidate_questions=questions_with_assertions, top_k=num_questions
        )
        pipeline_stats["selected"] = len(final_questions)

        # Type distribution stats
        type_dist: dict[str, int] = {}
        for q in final_questions:
            qtype = q.attributes.get("question_subtype", "unknown") if q.attributes else "unknown"
            type_dist[qtype] = type_dist.get(qtype, 0) + 1
        pipeline_stats["type_distribution"] = type_dist

        # Quality score stats
        quality_scores = [
            q.attributes.get("quality_score", 0) if q.attributes else 0
            for q in final_questions
        ]
        if quality_scores:
            pipeline_stats["quality_scores"] = {
                "min": min(quality_scores),
                "max": max(quality_scores),
                "avg": sum(quality_scores) / len(quality_scores),
            }

        log.info("Pipeline stats: %s", pipeline_stats)

        result = QuestionGenResult(
            selected_questions=final_questions,
            candidate_questions=results,
        )
        # Attach invalid assertions for debugging (accessible via result.invalid_assertions)
        result.invalid_assertions = invalid_assertions  # type: ignore[attr-defined]
        # Attach filter stats for debugging (accessible via result.filter_stats)
        result.filter_stats = stats  # type: ignore[attr-defined]
        # Attach pipeline stats
        result.pipeline_stats = pipeline_stats  # type: ignore[attr-defined]
        return result

    def select(
        self,
        candidate_questions: list[Question],
        top_k: int = 50,
        **kwargs: Any,
    ) -> list[Question]:
        """Select questions with proportional distribution across question types.

        Uses proportional sampling based on candidate distribution, then MMR
        within each type to select diverse, high-quality questions.

        Args:
            candidate_questions: List of candidate questions to select from.
            top_k: Number of questions to select.
            mmr_lambda: Trade-off between quality (1.0) and diversity (0.0).
            **kwargs: Additional arguments (unused, for compatibility).

        Returns
        -------
        list[Question]
            Selected questions with proportional types and high quality.

        """
        if len(candidate_questions) <= top_k:
            return candidate_questions

        # Group questions by type
        by_type: dict[str, list[Question]] = {}
        for q in candidate_questions:
            qtype = q.attributes.get("question_subtype", "unknown") if q.attributes else "unknown"
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(q)

        log.info(
            "Question type distribution in candidates: %s",
            {k: len(v) for k, v in by_type.items()},
        )

        if not by_type:
            return []

        # No type boosting - retrieval_difficulty in combined_score naturally favors bridge questions
        log.info("Using combined_score for selection (includes retrieval_difficulty)")

        # Select using MMR across all candidates with type balancing
        selected = self._mmr_select(
            candidate_questions, top_k, type_balance_weight=self.type_balance_weight
        )

        # Log final distribution
        final_dist: dict[str, int] = {}
        for q in selected:
            qtype = q.attributes.get("question_subtype", "unknown") if q.attributes else "unknown"
            final_dist[qtype] = final_dist.get(qtype, 0) + 1
        log.info("Question type distribution after selection: %s", final_dist)

        return selected

    def _mmr_select(
        self,
        questions: list[Question],
        k: int,
        type_balance_weight: float = 0.5,
    ) -> list[Question]:
        """Select k questions using type-aware MMR.

        Extends standard MMR with a type-balance penalty to encourage diverse
        question type distribution. The formula becomes:

            MMR(q) = λ * quality(q) - (1-λ) * max_similarity(q, selected)
                     - β * type_saturation(type(q), selected)

        Where type_saturation increases as a question type becomes over-represented
        relative to the target distribution (equal across types).

        Args:
            questions: List of candidate questions.
            k: Number of questions to select.
            type_balance_weight: Weight (β) for type-balance penalty (0=ignore, 1=strong).
                Default 0.3 provides moderate type balancing.

        Returns
        -------
        list[Question]
            Selected questions with balanced type distribution.

        """
        if k <= 0:
            return []
        if len(questions) <= k:
            return questions

        # Build embedding matrix and normalize
        embeddings = []
        for q in questions:
            if q.embedding is not None:
                embeddings.append(q.embedding)
            else:
                # Fallback: random embedding (shouldn't happen)
                rng = np.random.default_rng()
                embeddings.append(rng.standard_normal(768))
        embeddings = np.array(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_normalized = embeddings / norms

        # Get quality scores
        quality_scores = np.array([
            q.attributes.get("combined_score", 0.5) if q.attributes else 0.5
            for q in questions
        ])
        # Normalize quality scores to [0, 1]
        q_min, q_max = quality_scores.min(), quality_scores.max()
        if q_max > q_min:
            quality_scores = (quality_scores - q_min) / (q_max - q_min)

        # Get question types
        question_types = [
            q.attributes.get("question_subtype", "unknown") if q.attributes else "unknown"
            for q in questions
        ]
        unique_types = list(set(question_types))
        num_types = max(len(unique_types), 1)

        # Target distribution: equal across types
        target_per_type = k / num_types

        # MMR selection with type-balance penalty
        selected_indices: list[int] = []
        selected_embeddings: list[np.ndarray] = []
        selected_type_counts: dict[str, int] = dict.fromkeys(unique_types, 0)
        remaining_indices = set(range(len(questions)))

        # First selection: highest quality
        first_idx = int(np.argmax(quality_scores))
        selected_indices.append(first_idx)
        selected_embeddings.append(embeddings_normalized[first_idx])
        selected_type_counts[question_types[first_idx]] += 1
        remaining_indices.remove(first_idx)

        # Get lambda from sampler (quality-diversity tradeoff)
        lam = getattr(self.question_sampler, "lambda_param", 0.5)

        # Subsequent selections
        for _ in range(k - 1):
            if not remaining_indices:
                break

            best_idx = None
            best_score = float("-inf")

            for idx in remaining_indices:
                # Quality term
                quality = quality_scores[idx]

                # Diversity term: max similarity to selected
                if selected_embeddings:
                    similarities = [
                        float(np.dot(embeddings_normalized[idx], sel_emb))
                        for sel_emb in selected_embeddings
                    ]
                    max_sim = max(similarities)
                else:
                    max_sim = 0.0

                # Type-balance penalty: how saturated is this type?
                qtype = question_types[idx]
                current_count = selected_type_counts.get(qtype, 0)
                # Saturation: how much this type exceeds its fair share
                # saturation = current_count / target_per_type (capped at 1)
                # Penalty increases as type fills up
                if target_per_type > 0:
                    type_saturation = min(current_count / target_per_type, 2.0) / 2.0
                else:
                    type_saturation = 0.0

                # MMR score with type balance
                mmr_score = (
                    lam * quality
                    - (1 - lam) * max_sim
                    - type_balance_weight * type_saturation
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_embeddings.append(embeddings_normalized[best_idx])
                selected_type_counts[question_types[best_idx]] += 1
                remaining_indices.remove(best_idx)

        log.info(
            "Type-aware MMR selected %s items (type_balance_weight=%s, type_counts=%s)",
            len(selected_indices),
            type_balance_weight,
            selected_type_counts,
        )

        return [questions[i] for i in selected_indices]

    async def _ensure_embeddings(
        self, questions: list[Question]
    ) -> list[Question]:
        """Ensure all questions have embeddings, generating them if missing.

        Args:
            questions: List of questions to check/embed.

        Returns
        -------
        list[Question]
            Same list of questions with embeddings populated.

        """
        questions_needing_embeddings = [
            q for q in questions if q.embedding is None
        ]

        if not questions_needing_embeddings:
            return questions

        log.info(
            "Generating embeddings for %s questions",
            len(questions_needing_embeddings),
        )

        # Convert to TextUnits for batch embedding
        text_units = [
            TextUnit(id=q.id, short_id=None, text=q.text)
            for q in questions_needing_embeddings
        ]
        text_units = await self.text_embedder.embed_batch(
            text_units, batch_size=32
        )

        # Copy embeddings back to questions
        for q, tu in zip(questions_needing_embeddings, text_units, strict=True):
            q.embedding = tu.text_embedding

        return questions

    async def _compute_retrieval_difficulty(self, questions: list[Question]) -> None:
        """Compute retrieval difficulty for each question.

        Retrieval difficulty measures how hard it is to answer the question
        via simple embedding similarity search. It's computed as:
            retrieval_difficulty = 1 - max_source_similarity

        Where max_source_similarity is the maximum cosine similarity between
        the question embedding and any of its source claim texts.

        Bridge questions should have LOWER similarity (higher difficulty) because
        the entity name is replaced with an indirect reference.
        Comparison/intersection questions should have HIGHER similarity (lower
        difficulty) because they use entity names directly.

        Args:
            questions: List of questions with embeddings to compute difficulty for.

        """
        # Collect all unique claim texts that need embedding
        claim_texts_to_embed: dict[str, str] = {}  # claim_id -> text
        question_claim_ids: dict[str, list[str]] = {}  # question_id -> [claim_ids]

        for question in questions:
            if question.attributes is None:
                continue
            source_claims = question.attributes.get("source_claims", [])
            claim_ids = []
            for claim in source_claims:
                if isinstance(claim, dict):
                    claim_id = claim.get("claim_id", "")
                    statement = claim.get("statement", "")
                    if claim_id and statement:
                        claim_texts_to_embed[claim_id] = statement
                        claim_ids.append(claim_id)
            question_claim_ids[question.id] = claim_ids

        if not claim_texts_to_embed:
            log.debug("No claim texts to embed for retrieval difficulty")
            for question in questions:
                if question.attributes:
                    question.attributes["retrieval_difficulty"] = 0.5
                    question.attributes["max_source_similarity"] = 0.5
            return

        # Embed all claim texts
        claim_text_units = [
            TextUnit(id=cid, short_id=None, text=text)
            for cid, text in claim_texts_to_embed.items()
        ]
        log.debug("Embedding %d claim texts for retrieval difficulty", len(claim_text_units))
        claim_text_units = await self.text_embedder.embed_batch(claim_text_units, batch_size=32)

        # Build lookup for claim embeddings
        claim_embeddings: dict[str, list[float]] = {}
        for tu in claim_text_units:
            if tu.text_embedding is not None:
                claim_embeddings[tu.id] = tu.text_embedding

        # Compute similarity for each question
        for question in questions:
            if question.embedding is None or question.attributes is None:
                continue

            claim_ids = question_claim_ids.get(question.id, [])
            if not claim_ids:
                question.attributes["retrieval_difficulty"] = 0.5
                question.attributes["max_source_similarity"] = 0.5
                continue

            # Compute similarity to each source claim
            q_emb = np.array(question.embedding)
            q_norm = np.linalg.norm(q_emb)
            if q_norm == 0:
                question.attributes["retrieval_difficulty"] = 0.5
                question.attributes["max_source_similarity"] = 0.5
                continue

            q_emb_normalized = q_emb / q_norm

            similarities = []
            for claim_id in claim_ids:
                if claim_id in claim_embeddings:
                    claim_emb = np.array(claim_embeddings[claim_id])
                    claim_norm = np.linalg.norm(claim_emb)
                    if claim_norm > 0:
                        claim_emb_normalized = claim_emb / claim_norm
                        sim = float(np.dot(q_emb_normalized, claim_emb_normalized))
                        similarities.append(sim)

            if similarities:
                max_sim = max(similarities)
                question.attributes["max_source_similarity"] = max_sim
                question.attributes["retrieval_difficulty"] = 1.0 - max_sim
            else:
                question.attributes["max_source_similarity"] = 0.5
                question.attributes["retrieval_difficulty"] = 0.5

    def _compute_combined_score(
        self, questions: list[Question], k: int = 60
    ) -> None:
        """Compute combined score using Reciprocal Rank Fusion (RRF).

        Combines quality_score and retrieval_difficulty rankings into a single
        score using RRF formula:
            combined_score = sum(weight_i/(k + rank_i) for each ranking dimension)

        Weights:
            - quality_score: 1.0 (primary signal)
            - retrieval_difficulty: 1.0 (favors bridge questions with indirect references)

        Higher combined_score = better overall.

        Args:
            questions: List of questions to compute combined score for.
            k: RRF constant (default 60, standard value).

        """
        # Filter to questions with attributes
        valid_questions = [q for q in questions if q.attributes is not None]

        if not valid_questions:
            return

        # Rank by quality_score (descending - higher is better)
        quality_ranks = calculate_dense_ranks(
            valid_questions,
            key_func=lambda q: (
                q.attributes.get("quality_score", 0) if q.attributes else 0
            ),
            reverse=True,
        )

        # Rank by retrieval_difficulty (descending - higher is better)
        # Bridge questions naturally have higher retrieval_difficulty
        difficulty_ranks = calculate_dense_ranks(
            valid_questions,
            key_func=lambda q: (
                q.attributes.get("retrieval_difficulty", 0) if q.attributes else 0
            ),
            reverse=True,
        )

        # Compute RRF combined score with equal weights
        for question in valid_questions:
            if question.attributes is None:
                continue

            q_rank = quality_ranks.get(id(question), len(valid_questions))
            d_rank = difficulty_ranks.get(id(question), len(valid_questions))

            # RRF: quality + retrieval_difficulty (bridge questions rank higher on difficulty)
            rrf_score = (
                1.0 / (k + q_rank)
                + 1.0 / (k + d_rank)
            )

            question.attributes["combined_score"] = rrf_score
            question.attributes["quality_rank"] = q_rank
            question.attributes["difficulty_rank"] = d_rank

    def _generate_entity_contexts(
        self,
    ) -> list[EntityQuestionContext]:
        """Generate entity contexts by grouping local questions by shared entities.

        Note: The max_questions_to_generate field on each context will be set
        by agenerate() based on the oversample_factor calculation.
        """
        # Group questions by entity
        entity_to_questions = self._group_questions_by_entity()

        # Filter entities based on minimum questions threshold
        valid_entities = [
            entity
            for entity, questions in entity_to_questions.items()
            if len(questions) >= self.min_questions_per_entity
        ]

        log.info(
            "Found %s entities with >= %s questions (from %s total entities)",
            len(valid_entities),
            self.min_questions_per_entity,
            len(entity_to_questions),
        )

        # Create contexts for valid entities
        contexts = []
        for entity in valid_entities:
            questions = entity_to_questions[entity]

            # Limit questions per entity if needed
            if len(questions) > self.max_questions_per_entity:
                # Select most related questions by embedding similarity
                questions = self._select_related_questions(
                    questions, self.max_questions_per_entity
                )

            # Skip if selection reduced below minimum (edge case)
            if len(questions) < self.min_questions_per_entity:
                continue

            # Collect claims from questions
            claims = self._collect_claims_from_questions(questions)

            if not claims:
                log.debug("Skipping entity %s - no claims found", entity)
                continue

            contexts.append(
                EntityQuestionContext(
                    entity=entity,
                    local_questions=questions,
                    claims=claims,
                    # max_questions_to_generate will be set by agenerate()
                )
            )

        return contexts

    def _group_questions_by_entity(self) -> dict[str, list[Question]]:
        """Group local questions by their named entities."""
        entity_to_questions: dict[str, list[Question]] = defaultdict(list)

        for question in self.local_questions:
            if question.attributes and "named_entities" in question.attributes:
                for entity in question.attributes["named_entities"]:
                    # Normalize entity name (lowercase, strip whitespace)
                    normalized = entity.strip().lower()
                    if normalized:
                        entity_to_questions[normalized].append(question)

        return entity_to_questions

    def _select_related_questions(
        self, questions: list[Question], max_count: int
    ) -> list[Question]:
        """Select diverse questions about the same entity using MMR.

        We want questions that cover different aspects/facts about the entity
        so we can generate interesting multi-hop questions that combine them.

        Uses MMR sampler with high lambda (quality-focused) since these are
        input questions for generation, not final selection.

        Args:
            questions: List of local questions sharing an entity.
            max_count: Maximum number of questions to select.

        Returns
        -------
        list[Question]
            Selected questions that can be combined into multi-hop questions.

        """
        if len(questions) <= max_count:
            return questions

        # Convert Questions to TextUnits
        text_units: list[TextUnit] = []
        question_map: dict[str, Question] = {}
        for q in questions:
            if q.embedding is not None:
                tu = TextUnit(
                    id=q.id,
                    short_id=None,
                    text=q.text,
                    text_embedding=q.embedding,
                    attributes={"quality_score": 1.0},  # Equal quality for input selection
                )
                text_units.append(tu)
                question_map[q.id] = q

        if len(text_units) < 2:
            # Fall back to random selection
            rng = random.Random(self.random_seed)  # noqa: S311
            return rng.sample(questions, min(max_count, len(questions)))

        # Use MMR sampler - high lambda to start from centroid, then diversify
        selected_units = self.question_sampler.sample(
            text_units=text_units,
            sample_size=max_count,
            quality_attributes="quality_score",
        )

        return [question_map[tu.id] for tu in selected_units]

    def _collect_claims_from_questions(
        self, questions: list[Question]
    ) -> list[dict[str, Any]]:
        """Collect and format claims from questions with source tracking."""
        all_claims = []
        claim_id = 0

        for question in questions:
            if not question.attributes or "claims" not in question.attributes:
                continue

            for claim in question.attributes["claims"]:
                all_claims.append({
                    "claim_id": f"c{claim_id}",
                    "statement": claim.get("statement", ""),
                    "score": claim.get("score", 50),
                    "sources": claim.get("sources", []),
                    "source_question_id": question.id,
                })
                claim_id += 1

        return all_claims

    async def _agenerate_questions_for_entity(
        self,
        context: EntityQuestionContext,
        stats: QuestionFilterStats | None = None,
    ) -> list[Question]:
        """Generate questions for a single entity context using all configured question types."""
        all_questions: list[Question] = []

        # Distribute max questions across question types
        num_types = len(self.question_types)
        questions_per_type = max(1, context.max_questions_to_generate // num_types)

        for question_type in self.question_types:
            try:
                questions = await self._agenerate_questions_of_type(
                    context=context,
                    question_type=question_type,
                    max_questions=questions_per_type,
                    stats=stats,
                )
                all_questions.extend(questions)
            except (ValueError, KeyError, AttributeError, TypeError):
                log.exception(
                    "Error generating %s questions for entity %s",
                    question_type,
                    context.entity,
                )

        return all_questions

    async def _agenerate_questions_of_type(
        self,
        context: EntityQuestionContext,
        question_type: str,
        max_questions: int,
        stats: QuestionFilterStats | None = None,
    ) -> list[Question]:
        """Generate questions of a specific type for an entity context."""
        try:
            async with self.semaphore:
                log.debug(
                    "Generating %s questions for entity '%s' (max %d)",
                    question_type,
                    context.entity,
                    max_questions,
                )

                # Get the appropriate system prompt for this question type
                system_prompt_template = self.system_prompts.get(
                    question_type, self.system_prompts["bridge"]
                )

                # Format claims for prompt (grouped by source question)
                claims_text = self._format_claims_for_prompt(
                    context.claims, context.local_questions
                )

                # Build prompts
                system_prompt = system_prompt_template.substitute(
                    max_questions=max_questions
                )
                user_prompt = self.generation_user_prompt.substitute(
                    shared_entity=context.entity,
                    claims=claims_text,
                    max_questions=max_questions,
                )

                # Generate questions
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                result = await self.llm.chat(messages=messages, **self.llm_params)

                # Parse response
                questions = self._parse_question_response(
                    result.output.content, context, stats
                )

                # Tag questions with their type
                for q in questions:
                    if q.attributes:
                        q.attributes["question_subtype"] = question_type

                return questions

        except (ValueError, KeyError, AttributeError, TypeError):
            log.exception(
                "Error generating %s questions for entity %s",
                question_type,
                context.entity,
            )
            return []

    def _format_claims_for_prompt(
        self, claims: list[dict[str, Any]], local_questions: list[Question]
    ) -> str:
        """Format claims for the generation prompt, grouped by source question."""
        # Build question text lookup
        question_texts = {q.id: q.text for q in local_questions}

        # Group claims by source question
        claims_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for claim in claims:
            source_id = claim.get("source_question_id", "unknown")
            claims_by_question[source_id].append(claim)

        # Format grouped claims (same format as local assertion generator)
        lines = []
        for source_id, question_claims in claims_by_question.items():
            question_text = question_texts.get(source_id, "Unknown question")
            lines.append(f"Question: {question_text}")
            lines.append("Claims:")
            for claim in question_claims:
                lines.append(f"ID: {claim['claim_id']}")
                lines.append(f"Statement: {claim['statement']}")
                lines.append("")
            lines.append("")  # Extra blank line between questions

        return "\n".join(lines).strip()

    def _find_failed_quality_metrics(
        self,
        quality_scores: dict[str, Any],
    ) -> list[tuple[str, float]]:
        """Find quality metrics that fall below the minimum threshold.

        Checks all numeric scores except 'composite' and 'reasoning'
        against ``self.min_quality_score``.

        Args:
            quality_scores: Dict of metric names to score values.

        Returns
        -------
        list[tuple[str, float]]
            List of (metric, score) tuples that failed the threshold.
        """
        non_score_fields = {"composite", "reasoning"}
        failed: list[tuple[str, float]] = []
        for metric, score in quality_scores.items():
            if metric in non_score_fields:
                continue
            try:
                score_num = float(score)
            except (ValueError, TypeError):
                continue  # Skip non-numeric fields
            if score_num < self.min_quality_score:
                failed.append((metric, score_num))
        return failed

    def _validate_bridge_question(
        self,
        raw: dict[str, Any],
        text: str,
    ) -> str | None:
        """Validate bridge-type question has proper indirect reference.

        Checks that the bridge question has a non-empty replaced_entity
        and indirect_reference, and that the indirect_reference appears
        in the question text.

        Args:
            raw: Raw question dict from the LLM response.
            text: The question text (already stripped).

        Returns
        -------
        str | None
            A failure reason string if validation fails, or None if valid.
        """
        replaced_entity = raw.get("replaced_entity", "")
        indirect_reference = raw.get("indirect_reference", "")
        null_values = ("null", "none", "n/a", "")

        if (
            not replaced_entity
            or replaced_entity.lower() in null_values
        ):
            return "No replaced_entity"

        if (
            not indirect_reference
            or indirect_reference.lower() in null_values
        ):
            return "No indirect_reference"

        if indirect_reference.lower() not in text.lower():
            return (
                f"indirect_reference "
                f"'{indirect_reference[:30]}' not in text"
            )

        return None

    def _parse_question_response(
        self,
        response: str,
        context: EntityQuestionContext,
        stats: QuestionFilterStats | None = None,
    ) -> list[Question]:
        """Parse LLM response into Question objects with answerability safeguards."""
        _, parsed = try_parse_json_object(response)

        if not parsed or not isinstance(parsed, dict):
            log.debug("Failed to parse question response as JSON")
            if stats:
                stats.skipped_parse_error += 1
            return []

        questions = []
        raw_questions = parsed.get("questions", [])
        if not isinstance(raw_questions, list):
            raw_questions = []

        if stats:
            stats.total_raw += len(raw_questions)

        # Build set of valid claim IDs for validation
        valid_claim_ids = {c["claim_id"] for c in context.claims}

        for raw in raw_questions:
            if not isinstance(raw, dict):
                continue
            text = raw.get("text") or ""
            text = text.strip()
            if not text:
                continue

            # Get question type from response (prompts use "question_type")
            question_type = raw.get("question_type", "bridge")
            source_claim_ids = raw.get("source_claim_ids", [])

            # SAFEGUARD A: Validate all claim IDs exist
            invalid_ids = [
                cid for cid in source_claim_ids
                if cid not in valid_claim_ids
            ]
            if invalid_ids:
                log.debug(
                    "Question references non-existent claims %s, skipping: %s",
                    invalid_ids,
                    text[:50],
                )
                if stats:
                    stats.skipped_invalid_claims += 1
                    stats.add_filtered(raw, "invalid_claims", f"Invalid IDs: {invalid_ids}")
                continue

            # SAFEGUARD B: Require at least 2 claims (multi-hop requirement)
            if len(source_claim_ids) < 2:
                log.debug(
                    "Question requires fewer than 2 claims (%s), skipping: %s",
                    len(source_claim_ids),
                    text[:50],
                )
                if stats:
                    stats.skipped_few_claims += 1
                    stats.add_filtered(raw, "few_claims", f"Only {len(source_claim_ids)} claims")
                continue

            # Extract quality scores from quality sub-dict
            quality_scores = self._extract_quality_scores(raw)

            # SAFEGUARD C: Filter by quality scores
            failed_metrics = self._find_failed_quality_metrics(
                quality_scores
            )
            if failed_metrics:
                failed_str = ", ".join(
                    f"{m}={s}" for m, s in failed_metrics
                )
                log.debug(
                    "Question has low quality scores (%s), skipping: %s",
                    failed_str,
                    text[:50],
                )
                if stats:
                    stats.skipped_low_quality += 1
                    stats.add_filtered(raw, "low_quality", failed_str)
                continue

            # SAFEGUARD D: Filter bridge questions without proper indirect reference
            if question_type == "bridge":
                bridge_failure = self._validate_bridge_question(
                    raw, text
                )
                if bridge_failure:
                    log.debug(
                        "Bridge question validation failed (%s), skipping: %s",
                        bridge_failure,
                        text[:50],
                    )
                    if stats:
                        stats.skipped_single_document += 1
                        stats.add_filtered(raw, "single_document", bridge_failure)
                    continue

            # Collect sources from referenced claims
            references: list[str] = []

            for claim in context.claims:
                if claim["claim_id"] in source_claim_ids:
                    for source in claim.get("sources", []):
                        source_text = source.get("text", "")
                        if source_text:
                            references.append(source_text)

            # Build source_questions with id and text for easier review
            source_questions = [
                {"id": q.id, "text": q.text}
                for q in context.local_questions
            ]

            # Build source_claims with full claim info including sources for assertion gen
            source_claims = [
                {
                    "claim_id": claim["claim_id"],
                    "statement": claim["statement"],
                    "score": claim.get("score", 50),
                    "source_question_id": claim.get("source_question_id"),
                    "sources": claim.get("sources", []),  # Include sources for assertions
                }
                for claim in context.claims
                if claim["claim_id"] in source_claim_ids
            ]

            question = Question(
                id=str(uuid.uuid4()),
                text=text,
                question_type=QuestionType.DATA_LINKED,
                references=references[:30],  # Limit references
                attributes={
                    "entity": context.entity,
                    "question_subtype": question_type,
                    "source_claims": source_claims,
                    "source_questions": source_questions,
                    "claim_reasoning": raw.get("claim_reasoning", ""),
                    "draft_answer": raw.get("draft_answer", ""),
                    "claim_count": len(source_claims),
                    "reference_count": len(references),
                    "quality": quality_scores,
                    "quality_score": quality_scores.get("composite", 3.0),
                    # Bridge-specific fields for debugging
                    "replaced_entity": raw.get("replaced_entity", ""),
                    "indirect_reference": raw.get("indirect_reference", ""),
                    "reference_topic": raw.get("reference_topic", ""),
                    "question_topic": raw.get("question_topic", ""),
                },
            )
            questions.append(question)
            if stats:
                stats.accepted += 1

        return questions

    def _extract_quality_scores(
        self, raw_question: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract quality scores from LLM response and compute composite score.

        Args:
            raw_question: Raw question dict from LLM response with quality sub-dict.

        Returns
        -------
        dict[str, Any]
            Dictionary with individual scores, reasoning, and composite score.

        """
        # Get quality sub-dict (or empty dict if not present)
        quality_data = raw_question.get("quality", {})
        if not isinstance(quality_data, dict):
            quality_data = {}

        # Determine question type to use appropriate metrics
        question_type = raw_question.get("question_type", "bridge")

        # Base metrics for all question types
        base_metrics = ["naturalness", "answerability", "clarity"]

        # Type-specific metrics
        type_specific_metrics = {
            "bridge": ["bridge_relevance", "uniqueness"],
            "comparison": ["comparison_validity"],
            "intersection": ["commonality_depth"],
            "temporal": ["temporal_validity"],
        }

        metrics = base_metrics + type_specific_metrics.get(question_type, ["bridge_relevance"])

        scores: dict[str, Any] = {}
        score_values: list[float] = []

        for metric in metrics:
            score = quality_data.get(metric, 3)
            if not isinstance(score, int | float):
                score = 3  # Default middle score

            # Clamp score to valid range
            score = max(1, min(5, float(score)))
            scores[metric] = score
            score_values.append(score)

        # Store reasoning for quality scores
        scores["reasoning"] = quality_data.get("reasoning", "")

        # Compute composite score (simple average, normalized to 0-1 for MMR)
        if score_values:
            composite = sum(score_values) / len(score_values)
            # Normalize to 0-1 range for MMR (score range is 1-5)
            scores["composite"] = (composite - 1) / 4
        else:
            scores["composite"] = 0.5  # Default middle value

        return scores

    async def _generate_assertions_for_questions(
        self, questions: list[Question]
    ) -> list[dict[str, Any]]:
        """Generate assertions for all questions using LocalClaimAssertionGenerator.

        The generator handles validation internally if a validator was configured.

        Returns
        -------
            List of invalid assertion records for debugging/analysis (empty for now,
            invalid assertions are filtered internally by the generator).
        """
        invalid_assertions_log: list[dict[str, Any]] = []

        if self.assertion_generator is None:
            return invalid_assertions_log

        async def generate_for_question(question: Question) -> None:
            if question.attributes is None:
                question.attributes = {}

            # Entity questions store claims as 'source_claims'
            claims = question.attributes.get("source_claims", [])

            if self.assertion_generator is None:
                return

            result = await self.assertion_generator.agenerate_assertions(
                question.text,
                claims=claims,
            )

            # Generator already validates if validator is configured
            assertions = result.assertions

            # Store assertions in question attributes
            question.attributes["assertions"] = [
                {
                    "statement": a.statement,
                    "score": a.score,
                    "sources": a.sources,
                    "reasoning": a.reasoning,
                    "attributes": a.attributes,
                }
                for a in assertions
            ]
            question.attributes["assertion_count"] = len(assertions)

        # Process questions with concurrency limit
        max_concurrent = (
            self.assertion_config.linked.max_concurrent_questions
            if self.assertion_config
            else None
        )

        if max_concurrent and max_concurrent > 0:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def with_semaphore(q: Question) -> None:
                async with semaphore:
                    await generate_for_question(q)

            await tqdm_asyncio.gather(*[
                with_semaphore(q) for q in questions
            ])
        else:
            for question in questions:
                await generate_for_question(question)

        return invalid_assertions_log
