# Copyright (c) 2025 Microsoft Corporation.
"""Data-entity question generation module.

Generate multi-hop style questions by combining local questions that share named entities.
Similar to HotpotQA bridge questions but using entities as the linking mechanism.
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm.asyncio import tqdm_asyncio

import benchmark_qed.config.defaults as defs
from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.autod.sampler.sampling.mmr_sampler import MMRTextSampler
from benchmark_qed.autoq.data_model.enums import QuestionType
from benchmark_qed.autoq.data_model.question import Question
from benchmark_qed.autoq.prompts import data_questions as prompts_data_questions
from benchmark_qed.autoq.prompts.data_questions import entity_questions
from benchmark_qed.autoq.question_gen.base import BaseQuestionGen, QuestionGenResult
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.local_claim_assertion_gen import (
    LocalClaimAssertionGenerator,
)
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.validator import (
    AssertionValidator,
)
from benchmark_qed.autoq.sampler.question_sampler import QuestionSampler
from benchmark_qed.config.utils import load_template_file

if TYPE_CHECKING:
    from string import Template

    import tiktoken

    from benchmark_qed.autod.data_processor.embedding import TextEmbedder
    from benchmark_qed.autoq.config import AssertionConfig, AssertionPromptConfig
    from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

DATA_ENTITY_PROMPTS_PATH = Path(entity_questions.__file__).parent
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


class DataEntityQuestionGen(BaseQuestionGen):
    """
    Generate data-entity questions from local questions sharing named entities.

    Creates harder, multi-hop style questions by combining information from
    multiple local questions that mention the same entity. Similar to HotpotQA
    bridge questions but using named entities as the linking mechanism.

    The pipeline:
    1. Group local questions by shared named entities
    2. Generate entity questions using claims as context
    3. Generate assertions using map + dedupe approach
    4. Validate assertions against sources

    Supports optional assertion generation using EntityClaimAssertionGenerator.
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
        generation_system_prompt: Template | None = None,
        generation_user_prompt: Template | None = None,
        concurrent_coroutines: int = 32,
        random_seed: int = defs.RANDOM_SEED,
        min_questions_per_entity: int = 2,
        max_questions_per_entity: int = 10,
        min_quality_score: int = 4,
        question_types: list[str] | None = None,
        use_embedding_clustering: bool = True,  # Set False to use quality-only selection
        enable_batch_validation: bool = True,  # Run batch validation to filter bad questions
    ) -> None:
        from benchmark_qed.autoq.config import AssertionConfig, AssertionPromptConfig

        if assertion_config is None:
            assertion_config = AssertionConfig()
        if assertion_prompt_config is None:
            assertion_prompt_config = AssertionPromptConfig()

        # Default to all question types for maximum variety
        self.question_types = question_types or ["bridge", "comparison", "intersection"]
        
        self.assertion_config = assertion_config
        self.random_seed = random_seed
        self.min_questions_per_entity = min_questions_per_entity
        self.max_questions_per_entity = max_questions_per_entity
        self.min_quality_score = min_quality_score
        self.use_embedding_clustering = use_embedding_clustering
        self.enable_batch_validation = enable_batch_validation

        if question_sampler is not None:
            question_sampler.random_seed = self.random_seed
        elif use_embedding_clustering:
            # Use MMR (Maximal Marginal Relevance) for diversity + quality
            # MMR penalizes items similar to already-selected items,
            # which helps avoid near-duplicate questions
            question_sampler = QuestionSampler(
                sampler=MMRTextSampler(
                    random_seed=self.random_seed,
                    lambda_param=0.5,  # Balance quality and diversity
                ),
                sampler_params={
                    "quality_attributes": ["combined_score"],
                },
                random_seed=self.random_seed,
            )
        else:
            # No sampler - will use quality-only selection in select() fallback
            question_sampler = None
        super().__init__(llm, llm_params, question_sampler)
        self.text_embedder = text_embedder
        self.token_encoder = token_encoder

        # Assertion generation setup
        self.assertion_generator: LocalClaimAssertionGenerator | None = None
        self.assertion_validator: AssertionValidator | None = None
        entity_assertion_config = assertion_config.entity
        max_assertions = entity_assertion_config.max_assertions
        if max_assertions is None or max_assertions > 0:
            if entity_assertion_config.enable_validation:
                self.assertion_validator = AssertionValidator(
                    llm=llm,
                    llm_params=llm_params,
                    min_criterion_score=entity_assertion_config.min_validation_score,
                    # Use local validation prompt (entity assertions are fact-focused)
                    validation_prompt=assertion_prompt_config.local_validation_prompt.template,
                    concurrent_validations=entity_assertion_config.concurrent_llm_calls,
                )

            self.assertion_generator = LocalClaimAssertionGenerator(
                llm=llm,
                llm_params=llm_params,
                max_assertions=max_assertions,
            )

        self.json_mode = json_mode
        if json_mode:
            self.llm_params["response_format"] = {"type": "json_object"}
        else:
            self.llm_params.pop("response_format", None)

        # Load prompts for each question type
        self.system_prompts: dict[str, Template] = {}
        
        # Bridge questions
        self.system_prompts["bridge"] = (
            generation_system_prompt
            or load_template_file(
                DATA_ENTITY_PROMPTS_PATH / "bridge_question_system_prompt.txt"
            )
        )
        
        # Comparison questions
        self.system_prompts["comparison"] = load_template_file(
            DATA_ENTITY_PROMPTS_PATH / "comparison_question_system_prompt.txt"
        )
        
        # Intersection questions
        self.system_prompts["intersection"] = load_template_file(
            DATA_ENTITY_PROMPTS_PATH / "intersection_question_system_prompt.txt"
        )
        
        # Keep legacy attribute for compatibility
        self.generation_system_prompt = self.system_prompts["bridge"]
        
        self.generation_user_prompt: Template = (
            generation_user_prompt
            or load_template_file(
                DATA_ENTITY_PROMPTS_PATH / "entity_question_user_prompt.txt"
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

        # Step 2: Calculate max_questions_per_entity based on claim count
        # More claims = more potential question combinations, so scale accordingly
        # Base: 1 question per 2-3 claims (need at least 2 claims for multi-hop)
        total_claims = sum(len(ctx.claims) for ctx in entity_contexts)
        for ctx in entity_contexts:
            # Scale max questions by proportion of claims this context has
            # Minimum 1, and scale by claim richness
            claim_count = len(ctx.claims)
            if claim_count < 2:
                ctx.max_questions_to_generate = 0  # Can't do multi-hop with < 2 claims
            else:
                # More claims = more combinations possible
                # Use ceiling of (claims - 1) / 2 as base, capped by oversample needs
                base_max = max(1, (claim_count - 1) // 2)
                # Scale by oversample factor and proportion of total claims
                proportion = claim_count / total_claims if total_claims > 0 else 0
                scaled_max = math.ceil(num_candidate_questions * proportion)
                ctx.max_questions_to_generate = min(base_max, max(1, scaled_max))

        total_max = sum(ctx.max_questions_to_generate for ctx in entity_contexts)
        log.info(
            "Generated %s entity contexts from %s local questions "
            "(total %s claims, ~%s max questions based on claim distribution)",
            len(entity_contexts),
            len(self.local_questions),
            total_claims,
            total_max,
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

        log.info(
            "Generated %s candidate questions from %s entity groups",
            len(results),
            len(entity_contexts),
        )

        # Step 3: Generate embeddings for questions (only if using embedding clustering)
        if self.use_embedding_clustering:
            results = await self._ensure_embeddings(results)
            # Step 4: Compute retrieval difficulty for each question
            self._compute_retrieval_difficulty(results)
            # Step 5: Compute combined score using RRF (quality + difficulty + assertions)
            self._compute_combined_score(results)
        else:
            # Skip embedding-based metrics, use quality_score only
            log.info("Embedding clustering disabled - using quality_score only for selection")
            for q in results:
                if q.attributes:
                    q.attributes["combined_score"] = q.attributes.get("quality_score", 0.5)

        # Step 5.5: Run batch validation to filter out bad questions
        # This is cheaper than assertion generation, so do it first
        if self.enable_batch_validation:
            pre_validation_count = len(results)
            results = await self._batch_validate_questions(results)
            log.info(
                "After batch validation: %s questions remain (filtered %s)",
                len(results),
                pre_validation_count - len(results),
            )

        # Step 6: Generate assertions for ALL candidates (before selection)
        max_assertions = (
            self.assertion_config.entity.max_assertions
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

        result = QuestionGenResult(
            selected_questions=final_questions,
            candidate_questions=results,
        )
        # Attach invalid assertions for debugging (accessible via result.invalid_assertions)
        result.invalid_assertions = invalid_assertions  # type: ignore[attr-defined]
        # Attach filter stats for debugging (accessible via result.filter_stats)
        result.filter_stats = stats  # type: ignore[attr-defined]
        return result

    def select(
        self,
        candidate_questions: list[Question],
        top_k: int = 50,
        **kwargs: Any,
    ) -> list[Question]:
        """Select questions using K-means clustering for diversity.

        Uses K-means to cluster questions by embedding, then selects the
        highest quality question from each cluster. This naturally handles
        deduplication since similar questions end up in the same cluster.

        Args:
            candidate_questions: List of candidate questions to select from.
            top_k: Number of questions to select.
            **kwargs: Additional arguments passed to the sampler.

        Returns
        -------
        list[Question]
            Selected questions with diverse topics and high quality.

        """
        if len(candidate_questions) <= top_k:
            return candidate_questions

        # Use the question sampler (KmeansTextSampler) for selection
        # It clusters by embedding and picks best quality from each cluster
        if self.question_sampler:
            return self.question_sampler.sample(
                questions=candidate_questions, sample_size=top_k, **kwargs
            )

        # Fallback: simple quality-based selection
        sorted_questions = sorted(
            candidate_questions,
            key=lambda q: (
                q.attributes.get("quality_score", 0.5) if q.attributes else 0.5
            ),
            reverse=True,
        )
        return sorted_questions[:top_k]

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
        from benchmark_qed.autod.data_model.text_unit import TextUnit

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

    def _compute_retrieval_difficulty(self, questions: list[Question]) -> None:
        """Compute retrieval difficulty for each question.

        Retrieval difficulty measures how hard it is to answer the question
        via simple embedding similarity search. It's computed as:
            retrieval_difficulty = 1 - max_source_similarity

        Where max_source_similarity is the maximum cosine similarity between
        the question embedding and any of its source local question embeddings.

        Lower max similarity = harder to retrieve answer = higher difficulty.

        Args:
            questions: List of questions with embeddings to compute difficulty for.

        """
        import numpy as np

        # Build lookup for local question embeddings
        local_q_embeddings: dict[str, list[float]] = {}
        for lq in self.local_questions:
            if lq.embedding is not None:
                local_q_embeddings[lq.id] = lq.embedding

        for question in questions:
            if question.embedding is None or question.attributes is None:
                continue

            # Get source question IDs from attributes
            source_questions = question.attributes.get("source_questions", [])
            source_ids = [sq["id"] for sq in source_questions if isinstance(sq, dict)]

            if not source_ids:
                question.attributes["retrieval_difficulty"] = 0.5
                question.attributes["max_source_similarity"] = 0.5
                continue

            # Compute similarity to each source local question
            q_emb = np.array(question.embedding)
            q_norm = np.linalg.norm(q_emb)
            if q_norm == 0:
                question.attributes["retrieval_difficulty"] = 0.5
                question.attributes["max_source_similarity"] = 0.5
                continue

            q_emb_normalized = q_emb / q_norm

            similarities = []
            for source_id in source_ids:
                if source_id in local_q_embeddings:
                    source_emb = np.array(local_q_embeddings[source_id])
                    source_norm = np.linalg.norm(source_emb)
                    if source_norm > 0:
                        source_emb_normalized = source_emb / source_norm
                        sim = float(np.dot(q_emb_normalized, source_emb_normalized))
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

        Combines quality_score, retrieval_difficulty, assertion_count, and
        document_count rankings into a single score using RRF formula:
            combined_score = sum(weight_i/(k + rank_i) for each ranking dimension)

        Weights:
            - quality_score: 1.0 (primary signal)
            - assertion_count: 1.0 (evaluation coverage)
            - document_count: 1.0 (retrieval complexity)
            - retrieval_difficulty: 0.5 (secondary, may bias toward bridge)

        Higher combined_score = better overall.

        Args:
            questions: List of questions to compute combined score for.
            k: RRF constant (default 60, standard value).

        """
        from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.ranking import (
            calculate_dense_ranks,
        )

        # Filter to questions with attributes
        valid_questions = [q for q in questions if q.attributes is not None]

        if not valid_questions:
            return

        # Compute document count for each question
        for question in valid_questions:
            if question.attributes is None:
                continue
            source_claims = question.attributes.get("source_claims", [])
            doc_ids = set()
            for claim in source_claims:
                for source in claim.get("sources", []):
                    if isinstance(source, dict) and "document_id" in source:
                        doc_ids.add(source["document_id"])
            question.attributes["document_count"] = len(doc_ids)

        # Rank by quality_score (descending - higher is better)
        quality_ranks = calculate_dense_ranks(
            valid_questions,
            key_func=lambda q: (
                q.attributes.get("quality_score", 0) if q.attributes else 0
            ),
            reverse=True,
        )

        # Rank by retrieval_difficulty (descending - higher is better)
        difficulty_ranks = calculate_dense_ranks(
            valid_questions,
            key_func=lambda q: (
                q.attributes.get("retrieval_difficulty", 0) if q.attributes else 0
            ),
            reverse=True,
        )

        # Rank by assertion_count (descending - more is better)
        assertion_ranks = calculate_dense_ranks(
            valid_questions,
            key_func=lambda q: (
                q.attributes.get("assertion_count", 0) if q.attributes else 0
            ),
            reverse=True,
        )

        # Rank by document_count (descending - more docs is better/harder)
        doc_count_ranks = calculate_dense_ranks(
            valid_questions,
            key_func=lambda q: (
                q.attributes.get("document_count", 0) if q.attributes else 0
            ),
            reverse=True,
        )

        # Compute RRF combined score with weights
        for question in valid_questions:
            if question.attributes is None:
                continue

            q_rank = quality_ranks.get(id(question), len(valid_questions))
            d_rank = difficulty_ranks.get(id(question), len(valid_questions))
            a_rank = assertion_ranks.get(id(question), len(valid_questions))
            doc_rank = doc_count_ranks.get(id(question), len(valid_questions))

            # RRF with weights: quality=1.0, assertions=1.0, docs=1.0, difficulty=1.0
            rrf_score = (
                1.0 / (k + q_rank)
                + 1.0 / (k + a_rank)
                + 1.0 / (k + doc_rank)
                + 1.0 / (k + d_rank)
            )

            question.attributes["combined_score"] = rrf_score
            question.attributes["quality_rank"] = q_rank
            question.attributes["difficulty_rank"] = d_rank
            question.attributes["assertion_rank"] = a_rank
            question.attributes["doc_count_rank"] = doc_rank

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
        """Select diverse questions about the same entity using MMR-style selection.

        We want questions that cover different aspects/facts about the entity
        so we can generate interesting multi-hop questions that combine them.

        Algorithm (MMR-style for diversity):
        1. Start with question closest to centroid (most representative)
        2. Iteratively add questions that are MOST similar to already-selected
           (so they can be meaningfully combined into multi-hop questions)

        The intuition: questions that are semantically related can be combined
        into multi-hop questions (e.g., "Biden signed bill" + "Bill allocates $50B"
        → "How much did Biden's bill allocate?"). Unrelated questions about the
        same entity are hard to combine meaningfully.

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

        import numpy as np

        # Note: This method is sync, so we can't generate embeddings here.
        # Questions without embeddings will be handled by random selection fallback.
        # For best results, ensure local_questions have embeddings before calling.

        # Filter to questions with valid embeddings
        questions_with_embeddings = [
            q for q in questions if q.embedding is not None
        ]

        if len(questions_with_embeddings) < 2:
            # Fall back to random selection
            import random
            rng = random.Random(self.random_seed)  # noqa: S311
            return rng.sample(questions, min(max_count, len(questions)))

        # Build embedding matrix
        embeddings = np.array([q.embedding for q in questions_with_embeddings])

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_normalized = embeddings / norms

        # Start with question closest to centroid (most representative)
        centroid = np.mean(embeddings_normalized, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        similarities_to_centroid = embeddings_normalized @ centroid
        first_idx = int(np.argmax(similarities_to_centroid))

        selected_indices = [first_idx]
        selected_embeddings = [embeddings_normalized[first_idx]]

        # Iteratively add most similar questions (can be combined with selected)
        while len(selected_indices) < max_count:
            best_idx = -1
            best_score = float("-inf")  # Higher is more similar

            for i in range(len(questions_with_embeddings)):
                if i in selected_indices:
                    continue

                # Average similarity to selected questions
                avg_sim = sum(
                    float(np.dot(embeddings_normalized[i], sel_emb))
                    for sel_emb in selected_embeddings
                ) / len(selected_embeddings)

                # We want maximum avg_sim (most related to selected cluster)
                if avg_sim > best_score:
                    best_score = avg_sim
                    best_idx = i

            if best_idx >= 0:
                selected_indices.append(best_idx)
                selected_embeddings.append(embeddings_normalized[best_idx])
            else:
                break

        return [questions_with_embeddings[i] for i in selected_indices]

    async def _batch_validate_questions(
        self,
        questions: list[Question],
        batch_size: int = 25,
    ) -> list[Question]:
        """Run batch validation on questions using a separate LLM judge.

        Args:
            questions: List of questions to validate.
            batch_size: Number of questions per validation batch.

        Returns
        -------
        list[Question]
            Questions that passed validation.

        """
        if not questions:
            return []

        # Load validation prompt
        validation_prompt_path = DATA_ENTITY_PROMPTS_PATH / "batch_validation_prompt.txt"
        if not validation_prompt_path.exists():
            log.warning("Batch validation prompt not found, skipping validation")
            return questions

        validation_prompt = load_template_file(validation_prompt_path)

        passed_questions: list[Question] = []
        failed_count = 0

        # Process in batches
        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            
            # Format questions for validation (include entity for relevance check)
            questions_for_validation = [
                {
                    "id": j,
                    "text": q.text,
                    "type": q.attributes.get("question_subtype", "unknown") if q.attributes else "unknown",
                    "entity": q.attributes.get("entity", "") if q.attributes else "",
                }
                for j, q in enumerate(batch)
            ]

            import json
            prompt = validation_prompt.substitute(
                questions=json.dumps(questions_for_validation, indent=2)
            )

            try:
                messages = [{"role": "user", "content": prompt}]
                response = await self.llm.chat(
                    messages=messages,
                    **self.llm_params,
                )
                _, parsed = try_parse_json_object(response.output.content)

                if parsed and isinstance(parsed, dict):
                    results = parsed.get("results", [])
                    
                    # Build set of passing IDs
                    passing_ids = set()
                    for result in results:
                        if isinstance(result, dict):
                            qid = result.get("id")
                            if result.get("pass", False):
                                passing_ids.add(qid)
                            else:
                                reason = result.get("reason", "unknown")
                                log.debug(
                                    "Question failed batch validation: %s - %s",
                                    batch[qid].text[:50] if qid < len(batch) else "?",
                                    reason,
                                )
                                failed_count += 1

                    # Add passing questions
                    for j, q in enumerate(batch):
                        if j in passing_ids:
                            passed_questions.append(q)
                else:
                    # Parse failed, keep all questions from this batch
                    log.warning("Failed to parse batch validation response, keeping all questions")
                    passed_questions.extend(batch)

            except Exception as e:
                log.warning("Batch validation failed: %s, keeping all questions", e)
                passed_questions.extend(batch)

        log.info(
            "Batch validation: %s passed, %s failed out of %s total",
            len(passed_questions),
            failed_count,
            len(questions),
        )
        return passed_questions

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
            text = raw.get("text", "").strip()
            if not text:
                continue

            # Get question type from response (prompts use "question_type")
            question_type = raw.get("question_type", "bridge")
            source_claim_ids = raw.get("source_claim_ids", [])

            # SAFEGUARD A: Validate all claim IDs exist
            invalid_ids = [cid for cid in source_claim_ids if cid not in valid_claim_ids]
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

            # SAFEGUARD C: Filter by quality scores - ALL scores must meet threshold
            # Skip non-score fields like 'composite' and 'reasoning'
            non_score_fields = {"composite", "reasoning"}
            failed_metrics = []
            for metric, score in quality_scores.items():
                if metric in non_score_fields:
                    continue
                # Convert score to int if it's a string, skip non-numeric values
                try:
                    score_num = float(score) if isinstance(score, str) else float(score)
                except (ValueError, TypeError):
                    continue  # Skip non-numeric fields
                if score_num < self.min_quality_score:
                    failed_metrics.append((metric, score_num))
            if failed_metrics:
                failed_str = ", ".join(f"{m}={s}" for m, s in failed_metrics)
                log.debug(
                    "Question has low quality scores (%s), skipping: %s",
                    failed_str,
                    text[:50],
                )
                if stats:
                    stats.skipped_low_quality += 1
                    stats.add_filtered(raw, "low_quality", failed_str)
                continue

            # SAFEGUARD C2: Filter by self-validation check
            validation = raw.get("validation", {})
            if isinstance(validation, dict):
                passes_all = validation.get("passes_all_checks", True)
                if passes_all is False:
                    issues = validation.get("issues", [])
                    issues_str = ", ".join(issues) if issues else "validation failed"
                    log.debug(
                        "Question failed self-validation (%s), skipping: %s",
                        issues_str,
                        text[:50],
                    )
                    if stats:
                        stats.skipped_failed_validation += 1
                        stats.add_filtered(raw, "failed_validation", issues_str)
                    continue

            # SAFEGUARD D: Filter bridge questions without proper indirect reference
            if question_type == "bridge":
                replaced_entity = raw.get("replaced_entity", "")
                indirect_reference = raw.get("indirect_reference", "")
                
                # Check if replaced_entity is empty or null
                if not replaced_entity or replaced_entity.lower() in ("null", "none", "n/a", ""):
                    log.debug(
                        "Bridge question has no replaced_entity (single-hop), skipping: %s",
                        text[:50],
                    )
                    if stats:
                        stats.skipped_single_document += 1
                        stats.add_filtered(raw, "single_document", "No replaced_entity")
                    continue
                
                # Check if indirect_reference is empty
                if not indirect_reference or indirect_reference.lower() in ("null", "none", "n/a", ""):
                    log.debug(
                        "Bridge question has no indirect_reference, skipping: %s",
                        text[:50],
                    )
                    if stats:
                        stats.skipped_single_document += 1
                        stats.add_filtered(raw, "single_document", "No indirect_reference")
                    continue
                
                # Check if the indirect_reference actually appears in the question
                if indirect_reference.lower() not in text.lower():
                    log.debug(
                        "Bridge question indirect_reference '%s' not found in text, skipping: %s",
                        indirect_reference[:30],
                        text[:50],
                    )
                    if stats:
                        stats.skipped_single_document += 1
                        stats.add_filtered(raw, "single_document", f"indirect_reference '{indirect_reference[:30]}' not in text")
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
                question_type=QuestionType.DATA_ENTITY,
                references=references[:30],  # Limit references
                attributes={
                    "entity": context.entity,
                    "question_subtype": question_type,
                    "source_claims": source_claims,
                    "source_questions": source_questions,
                    "claim_reasoning": raw.get("claim_reasoning", ""),
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
            "bridge": ["bridge_relevance"],
            "comparison": ["comparison_validity"],
            "intersection": ["commonality_depth"],
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
        """Generate assertions for all questions.
        
        Returns:
            List of invalid assertion records for debugging/analysis.
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

            assertions = result.assertions
            total_generated = len(assertions)

            # Validate assertions if validator is configured
            if self.assertion_validator and assertions:
                validation_summary = await self.assertion_validator.validate_assertions(
                    assertions, question.text
                )
                
                # Log and collect invalid assertions with reasons
                if validation_summary.invalid_assertions:
                    log.debug(
                        "Question '%s...' had %d/%d assertions invalidated:",
                        question.text[:50],
                        len(validation_summary.invalid_assertions),
                        total_generated,
                    )
                    for invalid_result in validation_summary.invalid_assertions:
                        scores = invalid_result.scores
                        log.debug(
                            "  - INVALID: '%s...' | grounding=%d, relevance=%d, verifiability=%d | %s",
                            invalid_result.assertion.statement[:60],
                            scores.grounding,
                            scores.relevance,
                            scores.verifiability,
                            scores.reasoning[:100] if scores.reasoning else "No reason",
                        )
                        # Collect for file output
                        invalid_assertions_log.append({
                            "question_id": question.id,
                            "question_text": question.text,
                            "assertion_statement": invalid_result.assertion.statement,
                            "assertion_sources": invalid_result.assertion.sources,
                            "grounding_score": scores.grounding,
                            "relevance_score": scores.relevance,
                            "verifiability_score": scores.verifiability,
                            "reasoning": scores.reasoning,
                            "source_claims": claims,
                        })
                
                assertions = validation_summary.valid_assertions

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
            self.assertion_config.entity.max_concurrent_questions
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
