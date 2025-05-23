""""""

import asyncio
import json
import logging
from typing import Any

import tiktoken
from tqdm.asyncio import tqdm_asyncio

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.text_utils import (
    num_tokens,
    try_parse_json_object,
)
from benchmark_qed.autod.prompts.dataset_summary_prompts import (
    MAP_SUMMARY_SYSTEM_PROMPT,
    MAP_SUMMARY_USER_PROMPT,
    NO_DATA_ANSWER,
    REDUCE_SUMMARY_SYSTEM_PROMPT,
    REDUCE_SUMMARY_USER_PROMPT,
)
from benchmark_qed.autod.sampler.clustering.constraint_kmeans import (
    ConstraintKmeansClustering,
)
from benchmark_qed.autod.summarization.base import (
    BaseSummarizer,
    SummarizationResult,
)
from benchmark_qed.config.defaults import LLM_PARAMS
from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)


class GlobalSummarizer(BaseSummarizer):
    """Summarize a given dataset consisting of multiple text units using a map-reduce approach."""

    def __init__(
        self,
        llm: ChatModel,
        token_encoder: tiktoken.Encoding | None = None,
        map_system_prompt: str = MAP_SUMMARY_SYSTEM_PROMPT,
        map_user_prompt: str = MAP_SUMMARY_USER_PROMPT,
        reduce_system_prompt: str = REDUCE_SUMMARY_SYSTEM_PROMPT,
        reduce_user_prompt: str = REDUCE_SUMMARY_USER_PROMPT,
        map_llm_params: dict[str, Any] = LLM_PARAMS,
        reduce_llm_params: dict[str, Any] = LLM_PARAMS,
        response_type: str = "single paragraph",
        json_mode: bool = True,
        max_data_tokens: int = 8000,
        concurrent_coroutines: int = 32,
    ):
        super().__init__(
            llm=llm,
            token_encoder=token_encoder,
        )
        self.map_system_prompt = map_system_prompt
        self.map_user_prompt = map_user_prompt
        self.map_llm_params: dict[str, Any] = map_llm_params.copy()
        self.reduce_system_prompt = reduce_system_prompt
        self.reduce_user_prompt = reduce_user_prompt
        self.reduce_llm_params: dict[str, Any] = reduce_llm_params.copy()
        self.response_type = response_type

        self.max_data_tokens = max_data_tokens

        if json_mode:
            self.map_llm_params["response_format"] = {"type": "json_object"}
        else:
            # remove response_format key if json_mode is False
            self.map_llm_params.pop("response_format", None)

        self.reduce_llm_params.pop("response_format", None)

        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(concurrent_coroutines)

        # create a constraint kmeans clustering model for clustering the text units for summarization
        self.cluster_model: ConstraintKmeansClustering = ConstraintKmeansClustering(
            token_encoder=token_encoder,
        )

    async def asummarize(
        self,
        text_units: list[TextUnit],
        **kwargs: Any,
    ) -> SummarizationResult:
        """
        Perform a map-reduce summarization.
        - Step 1: Run parallel LLM calls on source texts to generate summaries for each batch of source texts.
        - Step 2: Combine the answers from step 1 to generate the final answer
        """
        # Step 1: Generate answers for each batch of community short summaries
        llm_calls, prompt_tokens, output_tokens = {}, {}, {}

        clusters = self.cluster_model.cluster(
            text_units=text_units,
            max_cluster_token_size=self.max_data_tokens,
        )
        log.info(f"Generating {len(clusters)} map responses...")
        map_responses = await tqdm_asyncio.gather(*[
            self._map_response_single_batch(
                context_data=cluster.convert_to_text(), **self.map_llm_params
            )
            for cluster in clusters
        ])

        llm_calls["map"] = sum(response.llm_calls for response in map_responses)
        prompt_tokens["map"] = sum(response.input_tokens for response in map_responses)
        output_tokens["map"] = sum(response.output_tokens for response in map_responses)

        # Step 2: Combine the intermediate answers from step 2 to generate the final answer
        reduce_response = await self._reduce_response(
            map_responses=map_responses,
            **self.reduce_llm_params,
        )
        llm_calls["reduce"] = reduce_response.llm_calls
        prompt_tokens["reduce"] = reduce_response.input_tokens
        output_tokens["reduce"] = reduce_response.output_tokens

        return SummarizationResult(
            summary=reduce_response.summary,
            llm_calls=sum(llm_calls.values()),
            input_tokens=sum(prompt_tokens.values()),
            output_tokens=sum(output_tokens.values()),
        )

    async def _map_response_single_batch(
        self,
        context_data: str,
        **llm_kwargs,
    ) -> SummarizationResult:
        """Generate summary for a single batch of source texts"""
        summarization_messages = [
            {"role": "system", "content": self.map_system_prompt},
            {
                "role": "user",
                "content": self.map_user_prompt.format(dataset_data=context_data),
            },
        ]
        input_tokens = num_tokens(
            summarization_messages[0]["content"] + summarization_messages[1]["content"],
            self.token_encoder,
        )

        try:
            async with self.semaphore:
                model_response = await self.llm.chat(
                    messages=summarization_messages,
                    **llm_kwargs,
                )
                map_response = model_response.output.content
                log.debug("Map response: %s", map_response)
            try:
                # parse search response json
                processed_response = self._parse_map_response(map_response)
            except ValueError:
                log.warning(
                    "Warning: Error parsing map response json - skipping this batch"
                )
                processed_response = []

            return SummarizationResult(
                summary=processed_response,
                llm_calls=1,
                input_tokens=input_tokens,
                output_tokens=num_tokens(map_response, self.token_encoder),
            )

        except Exception:
            log.exception("Exception in _map_response_single_batch")
            return SummarizationResult(
                summary=[{"topic": "", "score": 0}],
                llm_calls=1,
                input_tokens=input_tokens,
                output_tokens=0,
            )

    def _parse_map_response(self, map_response: str) -> list[dict[str, Any]]:
        """Parse the map response json and return a list of key topics.

        Parameters
        ----------
        map_response: str
            The map response json string

        Returns
        -------
        list[dict[str, Any]]
            A list of key topics, each key topic is a dictionary with "topic" and "score" keys
        """
        map_response, j = try_parse_json_object(map_response)
        if j == {}:
            return [{"topic": "", "score": 0}]

        parsed_elements = json.loads(map_response).get("topics")
        if not parsed_elements or not isinstance(parsed_elements, list):
            return [{"topic": "", "score": 0}]

        return [
            {
                "topic": element["description"],
                "score": int(element["score"]),
            }
            for element in parsed_elements
            if "description" in element and "score" in element
        ]

    def _prep_reduce_context(
        self,
        map_responses: list[SummarizationResult],
    ) -> str:
        """Prepare the context for the reduce step."""
        key_points = []
        for index, response in enumerate(map_responses):
            if not isinstance(response.summary, list):
                continue
            for element in response.summary:
                if not isinstance(element, dict):
                    continue
                if (
                    "topic" not in element
                    or "score" not in element
                    or element["score"] <= 0
                ):
                    continue
                key_points.append({
                    "analyst": index,
                    "topic": element["topic"],
                    "score": element["score"],
                })

        key_points = sorted(
            key_points,
            key=lambda x: x["score"],  # type: ignore
            reverse=True,  # type: ignore
        )

        data = []
        text_data = ""
        total_tokens = 0
        for point in key_points:
            formatted_response_data = []
            formatted_response_data.append(f"----Analyst {point['analyst'] + 1}----")
            formatted_response_data.append(
                f"Importance Score: {point['score']}"  # type: ignore
            )
            formatted_response_data.append(point["topic"])  # type: ignore
            formatted_response_text = "\n".join(formatted_response_data)
            if (
                total_tokens + num_tokens(formatted_response_text, self.token_encoder)
                > self.max_data_tokens
            ):
                break
            data.append(formatted_response_text)
            total_tokens += num_tokens(formatted_response_text, self.token_encoder)
        text_data = "\n\n".join(data)
        return text_data

    async def _reduce_response(
        self,
        map_responses: list[SummarizationResult],
        **llm_kwargs,
    ) -> SummarizationResult:
        """Combine all intermediate responses from single batches into a final answer."""
        text_data = self._prep_reduce_context(map_responses)
        if text_data.strip() == "":
            return SummarizationResult(
                summary=NO_DATA_ANSWER,
                llm_calls=0,
                input_tokens=0,
                output_tokens=0,
            )

        try:
            summarization_messages = [
                {"role": "system", "content": self.reduce_system_prompt},
                {
                    "role": "user",
                    "content": self.reduce_user_prompt.format(
                        map_summaries=text_data, response_type=self.response_type
                    ),
                },
            ]

            response = await self.llm.chat(
                messages=summarization_messages,
                **llm_kwargs,
            )

            return SummarizationResult(
                summary=response.output.content,
                llm_calls=1,
                input_tokens=num_tokens(
                    summarization_messages[0]["content"]
                    + summarization_messages[1]["content"],
                    self.token_encoder,
                ),
                output_tokens=num_tokens(response.output.content, self.token_encoder),
            )
        except Exception:
            log.exception("Exception in reduce_response")
            return SummarizationResult(
                summary="",
                llm_calls=0,
                input_tokens=0,
                output_tokens=0,
            )
