# Copyright (c) 2025 Microsoft Corporation.
"""YAML renderer for the interactive config wizard.

Transforms structured dicts (from the interactive wizard) into well-formatted,
commented YAML strings using a template-based approach to preserve inline
comments and consistent formatting.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import typer
import yaml

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _render_llm_section(provider_dict: dict[str, Any], indent: int = 2) -> str:
    """Render an LLM configuration section as a YAML fragment.

    Parameters
    ----------
    provider_dict:
        Dict (or dataclass) with keys ``llm_provider``, ``model``,
        ``auth_type``, ``init_args`` (from :class:`ProviderResult`).
    indent:
        Number of leading spaces for each line.
    """
    if dataclasses.is_dataclass(provider_dict) and not isinstance(provider_dict, type):
        provider_dict = dataclasses.asdict(provider_dict)

    pad = " " * indent
    lines: list[str] = []

    lines.extend([
        f"{pad}model: {provider_dict['model']}",
        f"{pad}auth_type: {provider_dict['auth_type']}",
    ])

    if provider_dict["auth_type"] == "api_key":
        lines.append(f"{pad}api_key: ${{OPENAI_API_KEY}}")

    lines.extend([
        f"{pad}llm_provider: {provider_dict['llm_provider']}",
        f"{pad}concurrent_requests: 4",
    ])

    init_args = provider_dict.get("init_args") or {}
    if init_args:
        lines.append(f"{pad}init_args:")
        for key, value in init_args.items():
            lines.append(f"{pad}  {key}: {value}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AutoQ
# ---------------------------------------------------------------------------


def render_autoq_yaml(config: dict[str, Any]) -> str:
    """Render a complete AutoQ ``settings.yaml`` from wizard configuration.

    Parameters
    ----------
    config:
        Dict with keys ``chat_provider``, ``embedding_provider``, ``input``,
        ``encoding``, ``sampling``, ``question_types``, ``activity_params``,
        ``assertions``, ``concurrent_requests``.
    """
    inp = config["input"]
    enc = config["encoding"]
    samp = config["sampling"]
    qt = config["question_types"]
    ap = config["activity_params"]
    assrt = config["assertions"]
    concurrent = config["concurrent_requests"]

    # Metadata columns
    meta = inp.get("metadata_columns")
    if meta is not None and isinstance(meta, list) and len(meta) > 0:
        meta_line = f"  metadata_columns: [{', '.join(meta)}]"
    else:
        meta_line = ""

    chat_section = _render_llm_section(config["chat_provider"])
    embedding_section = _render_llm_section(config["embedding_provider"])

    # Build metadata_columns block (include line only if present)
    metadata_block = f"\n{meta_line}" if meta_line else ""

    return f"""\
## Input Configuration
input:
  dataset_path: {inp["dataset_path"]}
  input_type: {inp["input_type"]}
  text_column: {inp["text_column"]}{metadata_block}
  file_encoding: {inp["file_encoding"]}

## Encoder configuration
encoding:
  model_name: {enc["model_name"]}
  chunk_size: {enc["chunk_size"]}
  chunk_overlap: {enc["chunk_overlap"]}

## Sampling Configuration
sampling:
  num_clusters: {samp["num_clusters"]}
  num_samples_per_cluster: {samp["num_samples_per_cluster"]}
  random_seed: {samp["random_seed"]}

## LLM Configuration
chat_model:
{chat_section}

embedding_model:
{embedding_section}

## Question Generation Configuration
data_local:
  num_questions: {qt["data_local"]["num_questions"]}
  oversample_factor: {_fmt_float(qt["data_local"]["oversample_factor"])}
data_global:
  num_questions: {qt["data_global"]["num_questions"]}
  oversample_factor: {_fmt_float(qt["data_global"]["oversample_factor"])}
data_linked:
  num_questions: {qt["data_linked"]["num_questions"]}
  oversample_factor: {_fmt_float(qt["data_linked"]["oversample_factor"])}
  min_questions_per_entity: 2
  max_questions_per_entity: 10
activity_local:
  num_questions: {qt["activity_local"]["num_questions"]}
  oversample_factor: {_fmt_float(qt["activity_local"]["oversample_factor"])}
  num_personas: {ap["num_personas"]}
  num_tasks_per_persona: {ap["num_tasks_per_persona"]}
  num_entities_per_task: {ap["num_entities_per_task"]}
activity_global:
  num_questions: {qt["activity_global"]["num_questions"]}
  oversample_factor: {_fmt_float(qt["activity_global"]["oversample_factor"])}
  num_personas: {ap["num_personas"]}
  num_tasks_per_persona: {ap["num_tasks_per_persona"]}
  num_entities_per_task: {ap["num_entities_per_task"]}

concurrent_requests: {concurrent}

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
    max_assertions: {assrt["max_assertions"]}
    enable_validation: {_fmt_bool(assrt["enable_validation"])}
    min_validation_score: {assrt["min_validation_score"]}
    concurrent_llm_calls: 8
    max_concurrent_questions: 8
  global:
    max_assertions: {assrt["max_assertions"]}
    enable_validation: {_fmt_bool(assrt["enable_validation"])}
    min_validation_score: {assrt["min_validation_score"]}
    batch_size: 100
    map_data_tokens: 8000
    reduce_data_tokens: 32000
    enable_semantic_grouping: true
    validate_map_assertions: true
    validate_reduce_assertions: true
    concurrent_llm_calls: 8
    max_concurrent_questions: 2
  linked:
    max_assertions: {assrt["max_assertions"]}
    enable_validation: {_fmt_bool(assrt["enable_validation"])}
    min_validation_score: {assrt["min_validation_score"]}
    concurrent_llm_calls: 8
    max_concurrent_questions: 2

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


# ---------------------------------------------------------------------------
# AutoE - Pairwise
# ---------------------------------------------------------------------------


def render_autoe_pairwise_yaml(config: dict[str, Any]) -> str:
    """Render a pairwise evaluation ``settings.yaml`` from wizard configuration.

    Parameters
    ----------
    config:
        Dict with keys ``chat_provider``, ``base``, ``others``,
        ``question_sets``, ``trials``, ``criteria``.
    """
    base = config["base"]
    others = config["others"]
    question_sets = config["question_sets"]
    trials = config["trials"]
    criteria = config.get("criteria")

    llm_section = _render_llm_section(config["chat_provider"])

    # others entries
    others_lines = "\n".join(
        f"  - name: {o['name']}\n    answer_base_path: {o['answer_base_path']}"
        for o in others
    )

    # question sets
    qsets_lines = "\n".join(f"  - {qs}" for qs in question_sets)

    # criteria block
    if criteria is not None:
        criteria_lines = "criteria:\n" + "\n".join(
            f'  - name: "{c["name"]}"\n    description: "{c["description"]}"'
            for c in criteria
        )
    else:
        criteria_lines = (
            "# criteria:\n"
            '#   - name: "criteria name"\n'
            '#     description: "criteria description"'
        )

    return f"""\
## Input Configuration
base:
  name: {base["name"]}
  answer_base_path: {base["answer_base_path"]}
others:
{others_lines}
question_sets:
{qsets_lines}

## Scoring Configuration
{criteria_lines}
trials: {trials}

## LLM Configuration
llm_config:
{llm_section}

prompts_config:
  user_prompt:
    prompt: prompts/pairwise_user_prompt.txt
  system_prompt:
    prompt: prompts/pairwise_system_prompt.txt
"""


# ---------------------------------------------------------------------------
# AutoE - Reference
# ---------------------------------------------------------------------------


def render_autoe_reference_yaml(config: dict[str, Any]) -> str:
    """Render a reference evaluation ``settings.yaml`` from wizard configuration.

    Parameters
    ----------
    config:
        Dict with keys ``chat_provider``, ``reference``, ``generated``,
        ``score_min``, ``score_max``, ``trials``.
    """
    ref = config["reference"]
    generated = config["generated"]
    trials = config["trials"]
    score_min = config.get("score_min", 1)
    score_max = config.get("score_max", 10)

    llm_section = _render_llm_section(config["chat_provider"])

    generated_lines = "\n".join(
        f"  - name: {g['name']}\n    answer_base_path: {g['answer_base_path']}"
        for g in generated
    )

    return f"""\
## Input Configuration
reference:
  name: {ref["name"]}
  answer_base_path: {ref["answer_base_path"]}
generated:
{generated_lines}

## Scoring Configuration
score_min: {score_min}
score_max: {score_max}
trials: {trials}

## LLM Configuration
llm_config:
{llm_section}

prompts_config:
  user_prompt:
    prompt: prompts/reference_user_prompt.txt
  system_prompt:
    prompt: prompts/reference_system_prompt.txt
"""


# ---------------------------------------------------------------------------
# AutoE - Assertion
# ---------------------------------------------------------------------------


def render_autoe_assertion_yaml(config: dict[str, Any]) -> str:
    """Render an assertion evaluation ``settings.yaml`` from wizard configuration.

    Parameters
    ----------
    config:
        Dict with keys ``chat_provider``, ``generated``, ``assertions``,
        ``pass_threshold``, ``trials``.
    """
    gen = config["generated"]
    assertions = config["assertions"]
    pass_threshold = config.get("pass_threshold", 0.5)
    trials = config["trials"]

    llm_section = _render_llm_section(config["chat_provider"])

    return f"""\
## Input Configuration
generated:
  name: {gen["name"]}
  answer_base_path: {gen["answer_base_path"]}
assertions:
  assertions_path: {assertions["assertions_path"]}

pass_threshold: {pass_threshold}
trials: {trials}

## LLM Configuration
llm_config:
{llm_section}

prompts_config:
  user_prompt:
    prompt: prompts/assertion_user_prompt.txt
  system_prompt:
    prompt: prompts/assertion_system_prompt.txt
"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_KEYS: dict[str, list[str]] = {
    "autoq": [
        "input",
        "encoding",
        "sampling",
        "chat_model",
        "embedding_model",
        "data_local",
        "data_global",
        "data_linked",
        "activity_local",
        "activity_global",
        "assertions",
    ],
    "autoe_pairwise": [
        "base",
        "others",
        "question_sets",
        "trials",
        "llm_config",
    ],
    "autoe_reference": [
        "reference",
        "generated",
        "score_min",
        "score_max",
        "trials",
        "llm_config",
    ],
    "autoe_assertion": [
        "generated",
        "assertions",
        "pass_threshold",
        "trials",
        "llm_config",
    ],
}


def validate_config(yaml_content: str, config_type: str) -> None:
    """Validate generated YAML against expected structure.

    Parses the YAML and checks that required top-level keys exist and have
    the correct types.  Prompt file paths are **not** validated because they
    are written to disk *after* the settings file is generated.

    Raises :class:`typer.BadParameter` on validation failure.
    """
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as exc:
        msg = f"Generated YAML is not valid: {exc}"
        raise typer.BadParameter(msg) from exc

    if not isinstance(data, dict):
        msg = "Generated YAML root must be a mapping."
        raise typer.BadParameter(msg)

    required = _REQUIRED_KEYS.get(config_type)
    if required is None:
        msg = f"Unknown config type: {config_type!r}"
        raise typer.BadParameter(msg)

    missing = [k for k in required if k not in data]
    if missing:
        msg = f"Missing required keys for {config_type}: {', '.join(missing)}"
        raise typer.BadParameter(msg)

    # Type-check a few critical fields
    try:
        if config_type == "autoq":
            _check_type(data, "input", dict)
            _check_type(data, "encoding", dict)
            _check_type(data, "sampling", dict)
            _check_type(data, "chat_model", dict)
            _check_type(data, "embedding_model", dict)
            _check_type(data, "assertions", dict)
        elif config_type == "autoe_pairwise":
            _check_type(data, "base", dict)
            _check_type(data, "others", list)
            _check_type(data, "question_sets", list)
            _check_type(data, "trials", int)
            _check_type(data, "llm_config", dict)
        elif config_type == "autoe_reference":
            _check_type(data, "reference", dict)
            _check_type(data, "generated", list)
            _check_type(data, "trials", int)
            _check_type(data, "llm_config", dict)
        elif config_type == "autoe_assertion":
            _check_type(data, "generated", dict)
            _check_type(data, "assertions", dict)
            _check_type(data, "trials", int)
            _check_type(data, "llm_config", dict)
    except typer.BadParameter:
        raise
    except Exception as exc:
        msg = f"Validation error for {config_type}: {exc}"
        raise typer.BadParameter(msg) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_type(data: dict[str, Any], key: str, expected: type) -> None:
    """Raise :class:`typer.BadParameter` if *data[key]* is not *expected* type."""
    value = data[key]
    if not isinstance(value, expected):
        msg = f"Key '{key}' should be {expected.__name__}, got {type(value).__name__}"
        raise typer.BadParameter(msg)


def _fmt_bool(value: Any) -> str:
    """Format a Python boolean as a YAML boolean literal."""
    return "true" if value else "false"


def _fmt_float(value: Any) -> str:
    """Format a number, ensuring floats keep a decimal point."""
    if isinstance(value, float):
        return str(value)
    # Integers that should display as floats (e.g. 2 -> 2.0)
    return f"{float(value)}"
