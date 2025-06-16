# Copyright (c) 2025 Microsoft Corporation.
"""Autoq CLI for generating questions."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from benchmark_qed.autod import prompts as autod_prompts
from benchmark_qed.autoe.prompts import pairwise as pairwise_prompts
from benchmark_qed.autoe.prompts import reference as reference_prompts
from benchmark_qed.autoq.prompts import data_questions as data_questions_prompts
from benchmark_qed.autoq.prompts.activity_questions import (
    activity_context as activity_context_prompts,
)
from benchmark_qed.autoq.prompts.activity_questions import (
    global_questions as activity_global_prompts,
)
from benchmark_qed.autoq.prompts.activity_questions import (
    local_questions as activity_local_prompts,
)
from benchmark_qed.autoq.prompts.data_questions import (
    global_questions as data_global_prompts,
)
from benchmark_qed.autoq.prompts.data_questions import (
    local_questions as data_local_prompts,
)

app: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)


class ConfigType(StrEnum):
    """Enum for the configuration type."""

    autoq = "autoq"
    autoe_pairwise = "autoe_pairwise"
    autoe_reference = "autoe_reference"


AUTOQ_CONTENT = """## Input Configuration
input:
  dataset_path: ./input
  input_type: json
  text_column: body_nitf # The column in the dataset that contains the text to be processed. Modify this based on your dataset.
  metadata_columns: [headline, firstcreated] # Additional metadata columns to include in the input. Modify this based on your dataset.
  file_encoding: utf-8-sig

## Encoder configuration
encoding:
  model_name: o200k_base
  chunk_size: 600
  chunk_overlap: 100

## Sampling Configuration
sampling:
  num_clusters: 20 # adjust this based on your dataset size and the number of questions you want to generate
  num_samples_per_cluster: 10
  random_seed: 42

## LLM Configuration
chat_model:
  model: gpt-4.1
  auth_type: api_key # or azure_managed_identity
  api_key: ${OPENAI_API_KEY} # remove this if using azure_managed_identity
  llm_provider: openai.chat # or azure.openai.chat | azure.inference.chat
  # init_args:
  #   Additional initialization arguments for the LLM can be added here.
  #   For example, you can set the model version or other parameters.
  #   api_version: 2024-12-01-preview
  #   azure_endpoint: https://<instance>.openai.azure.com
  # call_args:
  #   Additional arguments for the LLM call can be added here.
  #   For example, you can set the temperature, max tokens, etc.
  #   temperature: 0.0
  #   seed: 42
  # custom_providers: # When implementing a custom LLM provider, you can add it here.
  #   - model_type: chat
  #     name: custom.chat # This name should match the llm_provider above
  #     module: custom_test.custom_provider
  #     model_class: CustomChatModel

embedding_model:
  model: text-embedding-3-large
  auth_type: api_key # or azure_managed_identity
  api_key: ${OPENAI_API_KEY} # remove this if using azure_managed_identity
  llm_provider: openai.embedding # or azure.openai.embedding | azure.inference.embedding
  # init_args:
  #   Additional initialization arguments for the LLM can be added here.
  #   For example, you can set the model version or other parameters.
  #   api_version: 2024-12-01-preview
  #   azure_endpoint: https://<instance>.openai.azure.com
  # call_args:
  #   Additional arguments for the LLM call can be added here.
  #   For example, you can set the temperature, max tokens, etc.
  #   temperature: 0.0
  #   seed: 42
  # custom_providers: # When implementing a custom LLM provider, you can add it here.
  #   - model_type: chat
  #     name: custom.chat # This name should match the llm_provider above
  #     module: custom_test.custom_provider
  #     model_class: CustomChatModel

## Question Generation Configuration
data_local:
  num_questions: 10
  oversample_factor: 2.0
data_global:
  num_questions: 10
  oversample_factor: 2.0
activity_local:
  num_questions: 10
  oversample_factor: 2.0
  num_personas: 5 # adjust this based on the number of questions you want to generate
  num_tasks_per_persona: 2 # adjust this based on the number of questions you want to generate
  num_entities_per_task: 5 # adjust this based on the number of questions you want to generate
activity_global:
  num_questions: 10
  oversample_factor: 2.0
  num_personas: 5 # adjust this based on the number of questions you want to generate
  num_tasks_per_persona: 2 # adjust this based on the number of questions you want to generate
  num_entities_per_task: 5 # adjust this based on the number of questions you want to generate

concurrent_requests: 8

activity_questions_prompt_config:
  activity_context_prompt_config:
    data_summary_prompt_config:
      map_summary_system_prompt:
        prompt: prompts/activity_questions/activity_context/data_summary/map_summary_system_prompt.txt
      map_summary_user_prompt:
        prompt: prompts/activity_questions/activity_context/data_summary/map_summary_user_prompt.txt
      reduce_summary_system_prompt:
        prompt: prompts/activity_questions/activity_context/data_summary/reduce_summary_system_prompt.txt
      reduce_summary_user_prompt:
        prompt: prompts/activity_questions/activity_context/data_summary/reduce_summary_user_prompt.txt
    activity_identification_prompt:
      prompt: prompts/activity_questions/activity_context/activity_identification_prompt.txt
    map_entity_extraction_system_prompt:
      prompt: prompts/activity_questions/activity_context/map_entity_extraction_system_prompt.txt
    map_entity_extraction_user_prompt:
      prompt: prompts/activity_questions/activity_context/map_entity_extraction_user_prompt.txt
    reduce_entity_extraction_system_prompt:
      prompt: prompts/activity_questions/activity_context/reduce_entity_extraction_system_prompt.txt
    reduce_entity_extraction_user_prompt:
      prompt: prompts/activity_questions/activity_context/reduce_entity_extraction_user_prompt.txt
  activity_global_prompt_config:
    global_generation_system_prompt:
      prompt: prompts/activity_questions/activity_global/global_generation_system_prompt.txt
    global_generation_user_prompt:
      prompt: prompts/activity_questions/activity_global/global_generation_user_prompt.txt
  activity_local_prompt_config:
    local_generation_system_prompt:
      prompt: prompts/activity_questions/activity_local/local_generation_system_prompt.txt
    local_generation_user_prompt:
      prompt: prompts/activity_questions/activity_local/local_generation_user_prompt.txt

data_questions_prompt_config:
  claim_extraction_system_prompt:
    prompt: prompts/data_questions/claim_extraction_system_prompt.txt
  data_global_prompt_config:
    global_extraction_input_prompt:
      prompt: prompts/data_questions/data_global/global_extraction_input_prompt.txt
    global_extraction_prompt:
      prompt: prompts/data_questions/data_global/global_extraction_prompt.txt
  data_local_prompt_config:
    local_extraction_prompt:
      prompt: prompts/data_questions/data_local/local_extraction_prompt.txt
    local_generation_prompt:
      prompt: prompts/data_questions/data_local/local_generation_prompt.txt
    local_text_input_prompt:
      prompt: prompts/data_questions/data_local/local_text_input_prompt.txt
"""

AUTOE_PAIRWISE_CONTENT = """## Input Configuration
base:
  name: vector_rag
  answer_base_path: input/vector_rag  # The path to the base answers that you want to compare other RAG answers to. Modify this based on your dataset.
others: # List of other conditions to compare against the base.
  - name: lazygraphrag
    answer_base_path: input/lazygraphrag
  - name: graphrag_global
    answer_base_path: input/graphrag_global
question_sets: # List of question sets to use for scoring.
  - activity_global
  - activity_local

## Scoring Configuration
# criteria:
#   - name: "criteria name"
#     description: "criteria description"
trials: 4 # Number of trials to repeat the scoring process for each question. Should be an even number to allow for counterbalancing.

## LLM Configuration
llm_config:
  model: gpt-4.1
  auth_type: api_key # or azure_managed_identity
  api_key: ${OPENAI_API_KEY} # remove this if using azure_managed_identity
  llm_provider: openai.chat # or azure.openai.chat | azure.inference.chat
  concurrent_requests: 4
  # init_args:
  #   Additional initialization arguments for the LLM can be added here.
  #   For example, you can set the model version or other parameters.
  #   api_version: 2024-12-01-preview
  #   azure_endpoint: https://<instance>.openai.azure.com
  # call_args:
  #   Additional arguments for the LLM call can be added here.
  #   For example, you can set the temperature, max tokens, etc.
  #   temperature: 0.0
  #   seed: 42
  # custom_providers: # When implementing a custom LLM provider, you can add it here.
  #   - model_type: chat
  #     name: custom.chat # This name should match the llm_provider above
  #     module: custom_test.custom_provider
  #     model_class: CustomChatModel

prompts_config:
  user_prompt:
    prompt: prompts/pairwise_user_prompt.txt
  system_prompt:
    prompt: prompts/pairwise_system_prompt.txt"""


AUTOE_REFERENCE_CONTENT = """## Input Configuration
reference:
  name: lazygraphrag
  answer_base_path: input/lazygraphrag/activity_global.json # The path to the reference answers. Modify this based on your dataset.

generated:
  - name: vector_rag
    answer_base_path: input/vector_rag/activity_global.json # The path to the generated answers. Modify this based on your dataset.

## Scoring Configuration
score_min: 1
score_max: 10
# criteria:
#   - name: "criteria name"
#     description: "criteria description"
trials: 4 # Number of trials to repeat the scoring process for each question. Should be an even number to allow for counterbalancing.

## LLM Configuration
llm_config:
  model: gpt-4.1
  auth_type: api_key # or azure_managed_identity
  api_key: ${OPENAI_API_KEY} # remove this if using azure_managed_identity
  llm_provider: openai.chat # or azure.openai.chat | azure.inference.chat
  concurrent_requests: 4
  # init_args:
  #   Additional initialization arguments for the LLM can be added here.
  #   For example, you can set the model version or other parameters.
  #   api_version: 2024-12-01-preview
  #   azure_endpoint: https://<instance>.openai.azure.com
  # call_args:
  #   Additional arguments for the LLM call can be added here.
  #   For example, you can set the temperature, max tokens, etc.
  #   temperature: 0.0
  #   seed: 42
  # custom_providers: # When implementing a custom LLM provider, you can add it here.
  #   - model_type: chat
  #     name: custom.chat # This name should match the llm_provider above
  #     module: custom_test.custom_provider
  #     model_class: CustomChatModel

prompts_config:
  user_prompt: prompts/reference_user_prompt.txt
  system_prompt: prompts/reference_system_prompt.txt"""


def __copy_prompts(prompts_path: Path, output_path: Path) -> None:
    """Copy prompts from the prompts directory to the output directory."""
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    for prompt_file in prompts_path.iterdir():
        if prompt_file.is_file() and prompt_file.suffix == ".txt":
            target_file = output_path / prompt_file.name
            target_file.write_text(
                prompt_file.read_text(encoding="utf-8"), encoding="utf-8"
            )
            typer.echo(f"Copied {prompt_file} to {target_file}")


@app.command()
def init(
    config_type: Annotated[
        ConfigType,
        typer.Argument(
            help="The type of configuration to generate. Options are: autoq, autoe_pairwise, autoe_reference."
        ),
    ],
    root: Annotated[
        Path, typer.Argument(help="The path to root directory with the input folder.")
    ],
) -> None:
    """Generate settings file."""
    input_folder = root / "input"
    if not input_folder.exists():
        input_folder.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Input folder created at {input_folder}")
        typer.echo(
            "Please place your input files in the 'input' folder before running, or modify the settings.yaml to point to your input files."
        )

    settings = root / "settings.yaml"
    prompts_folder = root / "prompts"
    if config_type == ConfigType.autoq:
        settings.write_text(AUTOQ_CONTENT, encoding="utf-8")
        __copy_prompts(
            Path(autod_prompts.__file__).parent,
            prompts_folder / "activity_questions" / "activity_context" / "data_summary",
        )
        __copy_prompts(
            Path(activity_context_prompts.__file__).parent,
            prompts_folder / "activity_questions" / "activity_context",
        )
        __copy_prompts(
            Path(activity_global_prompts.__file__).parent,
            prompts_folder / "activity_questions" / "activity_global",
        )
        __copy_prompts(
            Path(activity_local_prompts.__file__).parent,
            prompts_folder / "activity_questions" / "activity_local",
        )
        __copy_prompts(
            Path(data_global_prompts.__file__).parent,
            prompts_folder / "data_questions" / "data_global",
        )
        __copy_prompts(
            Path(data_local_prompts.__file__).parent,
            prompts_folder / "data_questions" / "data_local",
        )
        __copy_prompts(
            Path(data_questions_prompts.__file__).parent,
            prompts_folder / "data_questions",
        )
    elif config_type == ConfigType.autoe_pairwise:
        settings.write_text(AUTOE_PAIRWISE_CONTENT, encoding="utf-8")
        __copy_prompts(Path(pairwise_prompts.__file__).parent, prompts_folder)
    elif config_type == ConfigType.autoe_reference:
        settings.write_text(AUTOE_REFERENCE_CONTENT, encoding="utf-8")
        __copy_prompts(Path(reference_prompts.__file__).parent, prompts_folder)

    typer.echo(f"Configuration file created at {settings}")

    env_file = root / ".env"
    if not env_file.exists():
        env_file.write_text("OPENAI_API_KEY=<API_KEY>", encoding="utf-8")
    typer.echo(
        f"Change the OPENAI_API_KEY placeholder at {env_file} with your actual OPENAI_API_KEY."
    )
