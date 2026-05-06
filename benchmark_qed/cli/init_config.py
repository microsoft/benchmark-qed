# Copyright (c) 2025 Microsoft Corporation.
"""Autoq CLI for generating questions."""

import asyncio
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from graphrag_storage.storage_config import StorageConfig
from graphrag_storage.storage_factory import create_storage

from benchmark_qed.autod.prompts import summarization
from benchmark_qed.autoe.prompts import assertion as assertion_prompts
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
    assertions as autoq_assertion_prompts,
)
from benchmark_qed.autoq.prompts.data_questions import (
    global_questions as data_global_prompts,
)
from benchmark_qed.autoq.prompts.data_questions import (
    linked_questions as data_linked_prompts,
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
    autoe_assertion = "autoe_assertion"


CHAT_MODEL_DEFAULTS = """
  model: gpt-4.1
  auth_type: api_key # or azure_managed_identity
  api_key: ${OPENAI_API_KEY} # remove this if using azure_managed_identity
  llm_provider: openai.chat # or azure.openai.chat | azure.inference.chat
  concurrent_requests: 4 # The number of concurrent requests to send to the model.
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
  #     model_class: CustomChatModel"""

EMBEDDING_MODEL_DEFAULTS = """
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
  #     model_class: CustomChatModel"""

AUTOQ_CONTENT = f"""## Input Configuration
input:
  dataset_path: ./input
  input_type: json
  text_column: body_nitf # The column in the dataset that contains the text to be processed. Modify this based on your dataset.
  metadata_columns: [headline, firstcreated] # Additional metadata columns to include in the input. Modify this based on your dataset.
  file_encoding: utf-8-sig
{{INPUT_STORAGE}}
## Output Storage Configuration
{{OUTPUT_STORAGE}}
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
chat_model: {CHAT_MODEL_DEFAULTS}

embedding_model: {EMBEDDING_MODEL_DEFAULTS}

## Question Generation Configuration
data_local:
  num_questions: 10
  oversample_factor: 2.0
data_global:
  num_questions: 10
  oversample_factor: 2.0
data_linked:
  num_questions: 10
  oversample_factor: 2.0
  min_questions_per_entity: 2 # Minimum local questions sharing an entity to form a group
  max_questions_per_entity: 10 # Maximum local questions per entity group
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
      summary_map_system_prompt:
        prompt: prompts/summarization/summary_map_system_prompt.txt
      summary_map_user_prompt:
        prompt: prompts/summarization/summary_map_user_prompt.txt
      summary_reduce_system_prompt:
        prompt: prompts/summarization/summary_reduce_system_prompt.txt
      summary_reduce_user_prompt:
        prompt: prompts/summarization/summary_reduce_user_prompt.txt
    activity_identification_prompt:
      prompt: prompts/activity_questions/activity_context/activity_identification_prompt.txt
    entity_extraction_map_system_prompt:
      prompt: prompts/activity_questions/activity_context/entity_extraction_map_system_prompt.txt
    entity_extraction_map_user_prompt:
      prompt: prompts/activity_questions/activity_context/entity_extraction_map_user_prompt.txt
    entity_extraction_reduce_system_prompt:
      prompt: prompts/activity_questions/activity_context/entity_extraction_reduce_system_prompt.txt
    entity_extraction_reduce_user_prompt:
      prompt: prompts/activity_questions/activity_context/entity_extraction_reduce_user_prompt.txt
  activity_global_prompt_config:
    activity_global_gen_system_prompt:
      prompt: prompts/activity_questions/activity_global/activity_global_gen_system_prompt.txt
    activity_global_gen_user_prompt:
      prompt: prompts/activity_questions/activity_global/activity_global_gen_user_prompt.txt
  activity_local_prompt_config:
    activity_local_gen_system_prompt:
      prompt: prompts/activity_questions/activity_local/activity_local_gen_system_prompt.txt
    activity_local_gen_user_prompt:
      prompt: prompts/activity_questions/activity_local/activity_local_gen_user_prompt.txt

data_questions_prompt_config:
  claim_extraction_system_prompt:
    prompt: prompts/data_questions/claim_extraction_system_prompt.txt
  data_global_prompt_config:
    data_global_gen_user_prompt:
      prompt: prompts/data_questions/data_global/data_global_gen_user_prompt.txt
    data_global_gen_system_prompt:
      prompt: prompts/data_questions/data_global/data_global_gen_system_prompt.txt
  data_local_prompt_config:
    data_local_gen_system_prompt:
      prompt: prompts/data_questions/data_local/data_local_gen_system_prompt.txt
    data_local_expansion_system_prompt:
      prompt: prompts/data_questions/data_local/data_local_expansion_system_prompt.txt
    data_local_gen_user_prompt:
      prompt: prompts/data_questions/data_local/data_local_gen_user_prompt.txt
  data_linked_prompt_config:
    bridge_question_system_prompt:
      prompt: prompts/data_questions/data_linked/bridge_question_system_prompt.txt
    comparison_question_system_prompt:
      prompt: prompts/data_questions/data_linked/comparison_question_system_prompt.txt
    intersection_question_system_prompt:
      prompt: prompts/data_questions/data_linked/intersection_question_system_prompt.txt
    linked_question_user_prompt:
      prompt: prompts/data_questions/data_linked/linked_question_user_prompt.txt
    batch_validation_prompt:
      prompt: prompts/data_questions/data_linked/batch_validation_prompt.txt

## Assertion Generation Configuration
assertions:
  local:
    max_assertions: 20 # Maximum assertions per question. Set to 0 to disable, or null for unlimited.
    enable_validation: true # Enable to filter low-quality assertions.
    min_validation_score: 3 # Minimum score (1-5) for an assertion to pass validation.
    concurrent_llm_calls: 8 # Concurrent LLM calls for validation.
    max_concurrent_questions: 8 # Parallel questions for assertion generation. Set to 1 for sequential.
  global:
    max_assertions: 20 # Maximum assertions per question. Set to 0 to disable, or null for unlimited.
    enable_validation: true # Enable to filter low-quality assertions.
    min_validation_score: 3 # Minimum score (1-5) for an assertion to pass validation.
    batch_size: 100 # Batch size for map-reduce processing (used when semantic grouping is disabled).
    map_data_tokens: 8000 # Maximum tokens per cluster in the map step (when semantic grouping enabled).
    reduce_data_tokens: 32000 # Maximum input tokens for the reduce step.
    enable_semantic_grouping: true # Group similar claims together before map step for better consolidation.
    validate_map_assertions: true # Validate map assertions before reduce step (filters low-quality early).
    validate_reduce_assertions: true # Validate final assertions after reduce step.
    concurrent_llm_calls: 8 # Concurrent LLM calls for batch processing and validation.
    max_concurrent_questions: 2 # Parallel questions for assertion generation. Set to 1 for sequential.
  linked:
    max_assertions: 20 # Maximum assertions per question. Set to 0 to disable, or null for unlimited.
    enable_validation: true # Enable to filter low-quality assertions.
    min_validation_score: 3 # Minimum score (1-5) for an assertion to pass validation.
    concurrent_llm_calls: 8 # Concurrent LLM calls for validation.
    max_concurrent_questions: 2 # Parallel questions for assertion generation. Set to 1 for sequential.

assertion_prompts:
  local_assertion_gen_prompt:
    prompt: prompts/data_questions/assertions/local_claim_assertion_gen_prompt.txt
  global_assertion_map_prompt:
    prompt: prompts/data_questions/assertions/global_claim_assertion_map_prompt.txt
  global_assertion_reduce_prompt:
    prompt: prompts/data_questions/assertions/global_claim_assertion_reduce_prompt.txt
  local_validation_prompt:
    prompt: prompts/data_questions/assertions/local_validation_prompt.txt
  global_validation_prompt:
    prompt: prompts/data_questions/assertions/global_validation_prompt.txt
"""

AUTOE_ASSERTION_CONTENT = f"""## Storage Configuration
{{STORAGE}}
## Input Configuration
generated:
  name: vector_rag
  answer_base_path: input/vector_rag/activity_global.json
assertions: # List of other conditions to compare against the base.
  assertions_path: input/activity_global_assertions.json # The path to the assertions file. Modify this based on your dataset.

pass_threshold: 0.5 # The threshold for passing the assertion. If the score is above this threshold, the assertion is considered passed.
trials: 4 # Number of trials to repeat the scoring process for each question-assertion pair.

## LLM Configuration
llm_config: {CHAT_MODEL_DEFAULTS}

prompt_config:
  user_prompt:
    prompt: prompts/assertion_user_prompt.txt
  system_prompt:
    prompt: prompts/assertion_system_prompt.txt"""

AUTOE_PAIRWISE_CONTENT = f"""## Storage Configuration
{{STORAGE}}
## Input Configuration
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
llm_config: {CHAT_MODEL_DEFAULTS}

prompt_config:
  user_prompt:
    prompt: prompts/pairwise_user_prompt.txt
  system_prompt:
    prompt: prompts/pairwise_system_prompt.txt"""


AUTOE_REFERENCE_CONTENT = f"""## Storage Configuration
{{STORAGE}}
## Input Configuration
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
llm_config: {CHAT_MODEL_DEFAULTS}

prompt_config:
  user_prompt:
    prompt: prompts/reference_user_prompt.txt
  system_prompt:
    prompt: prompts/reference_system_prompt.txt"""


AUTOQ_INPUT_STORAGE_SNIPPET = """  storage:
    type: blob
    container_name: my-datasets # The blob container name (acts as the root folder).
    connection_string: ${AZURE_STORAGE_CONNECTION_STRING} # Auth option 1: connection string.
    # account_url: https://<account>.blob.core.windows.net # Auth option 2: managed identity (use instead of connection_string).
    # base_dir: path/within/container # Optional prefix path. dataset_path is resolved relative to this."""


AUTOQ_OUTPUT_STORAGE_SNIPPET = """output_storage:
  type: blob
  container_name: my-output # The blob container name (acts as the root folder).
  connection_string: ${AZURE_STORAGE_CONNECTION_STRING} # Auth option 1: connection string.
  # account_url: https://<account>.blob.core.windows.net # Auth option 2: managed identity (use instead of connection_string).
  # base_dir: path/within/container # Optional prefix path. The CLI output argument is resolved relative to this."""


AUTOE_STORAGE_SNIPPET = """input_storage:
  type: blob
  container_name: my-datasets # The blob container name (acts as the root folder).
  connection_string: ${AZURE_STORAGE_CONNECTION_STRING} # Auth option 1: connection string.
  # account_url: https://<account>.blob.core.windows.net # Auth option 2: managed identity (use instead of connection_string).
output_storage:
  type: blob
  container_name: my-output
  connection_string: ${AZURE_STORAGE_CONNECTION_STRING}
  # account_url: https://<account>.blob.core.windows.net"""


def _commentify(active: str, indent: int = 0) -> str:
    """Comment out a YAML block by inserting `# ` after the given leading indent on each line.

    Also escapes ``$`` as ``$$`` so that ``${VAR}`` placeholders inside the commented
    block are not picked up by ``graphrag_common``'s ``string.Template`` env-var
    substitution (which scans the whole file regardless of YAML comments).
    """
    prefix = " " * indent
    out_lines = []
    for line in active.splitlines():
        escaped = line.replace("$", "$$")
        if not escaped.strip():
            out_lines.append(escaped)
        elif escaped.startswith(prefix):
            out_lines.append(prefix + "# " + escaped[indent:])
        else:
            out_lines.append("# " + escaped)
    return "\n".join(out_lines)


def _get_content(config_type: ConfigType) -> str:
    """Get the base template content for a config type."""
    match config_type:
        case ConfigType.autoq:
            return AUTOQ_CONTENT
        case ConfigType.autoe_pairwise:
            return AUTOE_PAIRWISE_CONTENT
        case ConfigType.autoe_reference:
            return AUTOE_REFERENCE_CONTENT
        case ConfigType.autoe_assertion:
            return AUTOE_ASSERTION_CONTENT


def _render_content(
    config_type: ConfigType,
    storage_type: str,
    *,
    container_name: str | None = None,
    account_url: str | None = None,
    connection_string: str | None = None,
    base_dir: str | None = None,
) -> str:
    """Render template content with the storage section in either commented or active form.

    Args:
        config_type: The type of configuration to generate.
        storage_type: Either 'local' (storage config commented out as documentation)
                     or 'blob' (active Azure Blob storage config injected).
        container_name: Optional container name to pre-fill in storage config.
        account_url: Optional account URL to pre-fill in storage config.
        connection_string: Optional connection string to pre-fill in storage config.
        base_dir: Optional base directory to pre-fill in storage config.
    """
    template = _get_content(config_type)
    active = storage_type == "blob"

    if config_type == ConfigType.autoq:
        input_block = (
            AUTOQ_INPUT_STORAGE_SNIPPET
            if active
            else _commentify(AUTOQ_INPUT_STORAGE_SNIPPET, indent=2)
        )
        output_block = (
            AUTOQ_OUTPUT_STORAGE_SNIPPET
            if active
            else _commentify(AUTOQ_OUTPUT_STORAGE_SNIPPET, indent=0)
        )
        content = template.replace("{INPUT_STORAGE}", input_block).replace(
            "{OUTPUT_STORAGE}", output_block
        )
    else:
        storage_block = (
            AUTOE_STORAGE_SNIPPET
            if active
            else _commentify(AUTOE_STORAGE_SNIPPET, indent=0)
        )
        content = template.replace("{STORAGE}", storage_block)

    if not active:
        return content

    # Substitute user-provided values into the active blob storage section.
    # Auth methods are mutually exclusive: pick exactly one active line.
    if container_name:
        content = content.replace("my-datasets", container_name)
        content = content.replace("my-output", container_name)
    if connection_string:
        # connection_string is the active default; just substitute the value.
        content = content.replace(
            "connection_string: ${AZURE_STORAGE_CONNECTION_STRING}",
            f"connection_string: {connection_string}",
        )
    elif account_url:
        # Activate account_url and comment out the default connection_string.
        content = content.replace(
            "# account_url: https://<account>.blob.core.windows.net",
            f"account_url: {account_url}",
        )
        content = content.replace(
            "connection_string: ${AZURE_STORAGE_CONNECTION_STRING}",
            "# connection_string: $${AZURE_STORAGE_CONNECTION_STRING}",
        )
    if base_dir:
        content = content.replace(
            "# base_dir: path/within/container",
            f"base_dir: {base_dir}",
        )

    return content


def __copy_prompts(prompts_path: Path, output_path: Path) -> None:
    """Copy prompts from the prompts directory to the local output directory."""
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    for prompt_file in prompts_path.iterdir():
        if prompt_file.is_file() and prompt_file.suffix == ".txt":
            target_file = output_path / prompt_file.name
            target_file.write_text(
                prompt_file.read_text(encoding="utf-8"), encoding="utf-8"
            )


def __get_prompt_files(prompts_path: Path) -> dict[str, str]:
    """Get prompt file contents as a dict of {filename: content}."""
    result = {}
    for prompt_file in prompts_path.iterdir():
        if prompt_file.is_file() and prompt_file.suffix == ".txt":
            result[prompt_file.name] = prompt_file.read_text(encoding="utf-8")
    return result


def _write_to_local(
    root: Path,
    settings_content: str,
    prompt_mapping: dict[str, dict[str, str]],
) -> None:
    """Write settings and prompts to local filesystem."""
    input_folder = root / "input"
    if not input_folder.exists():
        input_folder.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Input folder created at {input_folder}")
        typer.echo(
            "Please place your input files in the 'input' folder before running, "
            "or modify the settings.yaml to point to your input files."
        )

    settings = root / "settings.yaml"
    settings.write_text(settings_content, encoding="utf-8")

    for folder_path, files in prompt_mapping.items():
        output_path = root / folder_path
        output_path.mkdir(parents=True, exist_ok=True)
        for filename, file_content in files.items():
            (output_path / filename).write_text(file_content, encoding="utf-8")


def _write_to_blob(
    settings_content: str,
    prompt_mapping: dict[str, dict[str, str]],
    *,
    container_name: str | None = None,
    account_url: str | None = None,
    connection_string: str | None = None,
    base_dir: str | None = None,
) -> None:
    """Write settings and prompts to Azure Blob Storage."""
    config = StorageConfig(
        type="blob",
        container_name=container_name,
        account_url=account_url,
        connection_string=connection_string,
        base_dir=base_dir,
    )
    storage = create_storage(config)

    async def _upload() -> None:
        await storage.set("settings.yaml", settings_content)
        await storage.set(".env", "OPENAI_API_KEY=<API_KEY>")
        for folder_path, files in prompt_mapping.items():
            for filename, file_content in files.items():
                await storage.set(f"{folder_path}/{filename}", file_content)

    asyncio.get_event_loop().run_until_complete(_upload())


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
    storage_type: Annotated[
        str,
        typer.Option(
            "--storage-type",
            "-s",
            help="Storage setup mode for generated settings. Use 'blob' to scaffold active Azure Blob storage sections, or 'local' (default) for commented-out storage config.",
        ),
    ] = "local",
    container_name: Annotated[
        str | None,
        typer.Option(
            "--container-name",
            help="The blob container name to pre-fill in the generated storage config.",
        ),
    ] = None,
    account_url: Annotated[
        str | None,
        typer.Option(
            "--account-url",
            help="The storage account URL to pre-fill (uses managed identity for auth).",
        ),
    ] = None,
    connection_string: Annotated[
        str | None,
        typer.Option(
            "--connection-string",
            help="The storage connection string to pre-fill (alternative to --account-url).",
        ),
    ] = None,
    base_dir: Annotated[
        str | None,
        typer.Option(
            "--base-dir",
            help="Base prefix path within the container to pre-fill in storage config.",
        ),
    ] = None,
) -> None:
    """Generate settings file."""
    settings_content = _render_content(
        config_type,
        storage_type,
        container_name=container_name,
        account_url=account_url,
        connection_string=connection_string,
        base_dir=base_dir,
    )

    # Collect prompt files based on config type
    prompt_mapping: dict[str, dict[str, str]] = {}
    match config_type:
        case ConfigType.autoq:
            prompt_mapping["prompts/summarization"] = __get_prompt_files(
                Path(summarization.__file__).parent
            )
            prompt_mapping["prompts/activity_questions/activity_context"] = (
                __get_prompt_files(Path(activity_context_prompts.__file__).parent)
            )
            prompt_mapping["prompts/activity_questions/activity_global"] = (
                __get_prompt_files(Path(activity_global_prompts.__file__).parent)
            )
            prompt_mapping["prompts/activity_questions/activity_local"] = (
                __get_prompt_files(Path(activity_local_prompts.__file__).parent)
            )
            prompt_mapping["prompts/data_questions/data_global"] = __get_prompt_files(
                Path(data_global_prompts.__file__).parent
            )
            prompt_mapping["prompts/data_questions/data_local"] = __get_prompt_files(
                Path(data_local_prompts.__file__).parent
            )
            prompt_mapping["prompts/data_questions/data_linked"] = __get_prompt_files(
                Path(data_linked_prompts.__file__).parent
            )
            prompt_mapping["prompts/data_questions"] = __get_prompt_files(
                Path(data_questions_prompts.__file__).parent
            )
            prompt_mapping["prompts/data_questions/assertions"] = __get_prompt_files(
                Path(autoq_assertion_prompts.__file__).parent
            )
        case ConfigType.autoe_pairwise:
            prompt_mapping["prompts"] = __get_prompt_files(
                Path(pairwise_prompts.__file__).parent
            )
        case ConfigType.autoe_reference:
            prompt_mapping["prompts"] = __get_prompt_files(
                Path(reference_prompts.__file__).parent
            )
        case ConfigType.autoe_assertion:
            prompt_mapping["prompts"] = __get_prompt_files(
                Path(assertion_prompts.__file__).parent
            )

    # Write to blob storage or local filesystem
    if storage_type == "blob" and (account_url or connection_string):
        _write_to_blob(
            settings_content=settings_content,
            prompt_mapping=prompt_mapping,
            container_name=container_name,
            account_url=account_url,
            connection_string=connection_string,
            base_dir=base_dir,
        )
        target = f"blob://{container_name or 'container'}"
        if base_dir:
            target += f"/{base_dir}"
        typer.echo(f"Configuration files uploaded to {target}")
    else:
        _write_to_local(
            root=root,
            settings_content=settings_content,
            prompt_mapping=prompt_mapping,
        )
        typer.echo(f"Configuration file created at {root / 'settings.yaml'}")
        env_file = root / ".env"
        if not env_file.exists():
            env_file.write_text("OPENAI_API_KEY=<API_KEY>", encoding="utf-8")
        typer.echo(
            f"Change the OPENAI_API_KEY placeholder at {env_file} with your actual OPENAI_API_KEY."
        )
