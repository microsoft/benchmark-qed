# Copyright (c) 2025 Microsoft Corporation.
"""Entity extraction module for activity question generation in AutoQ."""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import tiktoken
from tqdm.asyncio import tqdm_asyncio

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.text_utils import (
    num_tokens,
    try_parse_json_object,
)
from benchmark_qed.autod.sampler.clustering.constraint_kmeans import (
    ConstraintKmeansClustering,
)
from benchmark_qed.autoq.data_model.activity import Entity
from benchmark_qed.autoq.prompts import activity_questions
from benchmark_qed.config.defaults import LLM_PARAMS
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

PROMPTS_PATH = Path(activity_questions.__file__).parent


@dataclass
class EntityExtractionResult:
    """Result of the entity extraction process."""

    entities: list[Entity]
    input_tokens: int
    output_tokens: int
    llm_calls: int


class EntityExtractor:
    """Extract entities that are relevant to a given user-task-dataset context using a map-reduce approach."""

    def __init__(
        self,
        llm: ChatModel,
        token_encoder: tiktoken.Encoding | None = None,
        map_system_prompt: Template | None = None,
        map_user_prompt: Template | None = None,
        reduce_system_prompt: Template | None = None,
        reduce_user_prompt: Template | None = None,
        map_llm_params: dict[str, Any] = LLM_PARAMS,
        reduce_llm_params: dict[str, Any] = LLM_PARAMS,
        json_mode: bool = True,
        max_data_tokens: int = 8000,
        concurrent_coroutines: int = 32,
    ) -> None:
        self.llm = llm
        self.token_encoder = token_encoder
        self.map_system_prompt: str = (
            map_system_prompt
            or load_template_file(
                PROMPTS_PATH / "map_entity_extraction_system_prompt.txt"
            )
        ).template
        self.map_user_prompt: Template = map_user_prompt or load_template_file(
            PROMPTS_PATH / "map_entity_extraction_user_prompt.txt"
        )
        self.map_llm_params = map_llm_params
        self.reduce_system_prompt: str = (
            reduce_system_prompt
            or load_template_file(
                PROMPTS_PATH / "reduce_entity_extraction_system_prompt.txt"
            )
        ).template
        self.reduce_user_prompt: Template = reduce_user_prompt or load_template_file(
            PROMPTS_PATH / "reduce_entity_extraction_user_prompt.txt"
        )
        self.reduce_llm_params = reduce_llm_params

        self.max_data_tokens = max_data_tokens

        if json_mode:
            self.map_llm_params["response_format"] = {"type": "json_object"}
            self.reduce_llm_params["response_format"] = {"type": "json_object"}
        else:
            self.map_llm_params.pop("response_format", None)
            self.reduce_llm_params.pop("response_format", None)

        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(concurrent_coroutines)

        # create a constraint kmeans clustering model for clustering the text units for summarization
        self.cluster_model: ConstraintKmeansClustering = ConstraintKmeansClustering(
            token_encoder=token_encoder,
        )

    async def aextract(
        self,
        text_units: list[TextUnit],
        persona: str,
        task: str,
        num_entities: int = 5,
        **kwargs: Any,
    ) -> EntityExtractionResult:
        """
        Perform a map-reduce entity extraction.

        - Step 1: Run parallel LLM calls on source texts to generate summaries for each batch of source texts.
        - Step 2: Combine the answers from step 1 to generate the final answer.
        """
        # Step 1: Generate answers for each batch of community short summaries
        llm_calls, prompt_tokens, output_tokens = {}, {}, {}

        clusters = self.cluster_model.cluster(
            text_units=text_units,
            max_cluster_token_size=self.max_data_tokens,
        )
        msg = f"Generating {len(clusters)} map responses..."
        log.info(msg)
        map_responses = await tqdm_asyncio.gather(*[
            self._map_response_single_batch(
                persona=persona,
                task=task,
                num_entities=num_entities,
                context_data=cluster.convert_to_text(),
                **self.map_llm_params,
            )
            for cluster in clusters
        ])

        llm_calls["map"] = sum(response.llm_calls for response in map_responses)
        prompt_tokens["map"] = sum(response.input_tokens for response in map_responses)
        output_tokens["map"] = sum(response.output_tokens for response in map_responses)

        # Step 2: Combine the intermediate answers from step 2 to generate the final answer
        reduce_response = await self._reduce_response(
            persona=persona,
            task=task,
            num_entities=num_entities,
            map_responses=map_responses,
            **self.reduce_llm_params,
        )
        llm_calls["reduce"] = reduce_response.llm_calls
        prompt_tokens["reduce"] = reduce_response.input_tokens
        output_tokens["reduce"] = reduce_response.output_tokens

        return EntityExtractionResult(
            entities=reduce_response.entities,
            llm_calls=sum(llm_calls.values()),
            input_tokens=sum(prompt_tokens.values()),
            output_tokens=sum(output_tokens.values()),
        )

    async def _map_response_single_batch(
        self,
        persona: str,
        task: str,
        context_data: str,
        num_entities: int = 5,
        **llm_kwargs: Any,
    ) -> EntityExtractionResult:
        """Extract relevant entities from a single batch of source texts."""
        map_messages = [
            {"role": "system", "content": self.map_system_prompt},
            {
                "role": "user",
                "content": self.map_user_prompt.substitute(
                    context_data=context_data,
                    persona=persona,
                    task=task,
                    num_entities=num_entities,
                ),
            },
        ]
        input_tokens = num_tokens(
            map_messages[0]["content"] + map_messages[1]["content"], self.token_encoder
        )

        try:
            async with self.semaphore:
                model_response = await self.llm.chat(
                    messages=map_messages,
                    **llm_kwargs,
                )
                map_response = model_response.output.content
                log.debug("Map response: %s", map_response)
            try:
                # parse search response json
                processed_response = self._parse_extraction_response(map_response)
            except ValueError:
                log.warning(
                    "Warning: Error parsing map response json - skipping this batch"
                )
                processed_response = []

            return EntityExtractionResult(
                entities=processed_response,
                llm_calls=1,
                input_tokens=input_tokens,
                output_tokens=num_tokens(map_response, self.token_encoder),
            )

        except Exception:
            log.exception("Exception in _map_response_single_batch")
            return EntityExtractionResult(
                entities=[],
                llm_calls=1,
                input_tokens=input_tokens,
                output_tokens=0,
            )

    def _parse_extraction_response(
        self, response: str, num_entities: int | None = None
    ) -> list[Entity]:
        """Parse the map response json and return a list of entities.

        Parameters
        ----------
        map_response: str
            The map response json string

        Returns
        -------
        list[Entity]
            A list of entities extracted from the map response
        """
        map_response, j = try_parse_json_object(response)
        if j == {}:
            return []

        parsed_elements = json.loads(map_response).get("entities")
        if not parsed_elements or not isinstance(parsed_elements, list):
            return []

        entities = [
            Entity(
                name=element["entity_name"],
                description=element["entity_description"],
                relevance_score=int(element.get("relevance_score", 50)),
            )
            for element in parsed_elements
            if "entity_name" in element and "entity_description" in element
        ]

        if num_entities is not None:
            # get top N entities by relevance score
            entities = sorted(entities, key=lambda x: x.relevance_score, reverse=True)[
                :num_entities
            ]

        return entities

    def _prep_reduce_context(
        self,
        map_responses: list[EntityExtractionResult],
    ) -> str:
        """Prepare the context for the reduce step."""
        candidate_entities = [
            entity for map_response in map_responses for entity in map_response.entities
        ]
        candidate_entities = sorted(
            candidate_entities,
            key=lambda x: x.relevance_score,
            reverse=True,
        )

        header = "entity_name|entity_description|relevance_score"
        data = [header]
        total_tokens = num_tokens(header, self.token_encoder)
        for entity in candidate_entities:
            formatted_response_data = [
                entity.name,
                entity.description,
                str(entity.relevance_score),
            ]
            formatted_response_text = "|".join(formatted_response_data)
            if (
                total_tokens + num_tokens(formatted_response_text, self.token_encoder)
                > self.max_data_tokens
            ):
                break
            data.append(formatted_response_text)
            total_tokens += num_tokens(formatted_response_text, self.token_encoder)
        return "\n".join(data)

    async def _reduce_response(
        self,
        map_responses: list[EntityExtractionResult],
        persona: str,
        task: str,
        num_entities: int = 5,
        **llm_kwargs: Any,
    ) -> EntityExtractionResult:
        """Combine all intermediate responses from single batches into a final answer."""
        text_data = self._prep_reduce_context(map_responses)
        if text_data.strip() == "":
            return EntityExtractionResult(
                entities=[],
                llm_calls=0,
                input_tokens=0,
                output_tokens=0,
            )

        try:
            reduce_messages = [
                {"role": "system", "content": self.reduce_system_prompt},
                {
                    "role": "user",
                    "content": self.reduce_user_prompt.substitute(
                        map_entities=text_data,
                        persona=persona,
                        task=task,
                        num_entities=num_entities,
                    ),
                },
            ]

            response = await self.llm.chat(
                messages=reduce_messages,
                **llm_kwargs,
            )

            return EntityExtractionResult(
                entities=self._parse_extraction_response(
                    response.output.content, num_entities=num_entities
                ),
                llm_calls=1,
                input_tokens=num_tokens(
                    reduce_messages[0]["content"] + reduce_messages[1]["content"],
                    self.token_encoder,
                ),
                output_tokens=num_tokens(response.output.content, self.token_encoder),
            )
        except Exception:
            log.exception("Exception in reduce_response")
            return EntityExtractionResult(
                entities=[],
                llm_calls=0,
                input_tokens=0,
                output_tokens=0,
            )
