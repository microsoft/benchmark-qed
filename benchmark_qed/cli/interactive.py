# Copyright (c) 2025 Microsoft Corporation.
"""Interactive configuration wizard for benchmark-qed."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer
from rich import print as rich_print
from rich.panel import Panel
from rich.table import Table

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
from benchmark_qed.cli.scaffold import copy_prompts, ensure_input_folder, write_env_file
from benchmark_qed.cli.yaml_renderer import (
    render_autoe_assertion_yaml,
    render_autoe_pairwise_yaml,
    render_autoe_reference_yaml,
    render_autoq_yaml,
    validate_config,
)

app: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FieldDef:
    """Definition of a configurable field shown to the user."""

    name: str
    description: str
    default: Any
    field_type: type = str
    choices: list[str] | None = None


@dataclass
class ProviderResult:
    """Result from the provider selection flow."""

    llm_provider: str
    model: str
    auth_type: str
    init_args: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provider metadata
# ---------------------------------------------------------------------------

CHAT_PROVIDERS: list[tuple[str, str, str]] = [
    ("openai.chat", "OpenAI", "OpenAI API (default)"),
    ("azure.openai.chat", "Azure OpenAI", "Azure-hosted OpenAI models"),
    ("azure.inference.chat", "Azure Inference", "Azure AI Inference endpoint"),
]

EMBEDDING_PROVIDERS: list[tuple[str, str, str]] = [
    ("openai.embedding", "OpenAI", "OpenAI API (default)"),
    ("azure.openai.embedding", "Azure OpenAI", "Azure-hosted OpenAI embeddings"),
    (
        "azure.inference.embedding",
        "Azure Inference",
        "Azure AI Inference endpoint",
    ),
]

AUTH_TYPES: list[tuple[str, str]] = [
    ("api_key", "API Key"),
    ("azure_managed_identity", "Azure Managed Identity"),
]

DEFAULT_CHAT_MODEL = "gpt-4.1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_API_VERSION = "2024-12-01-preview"


# ---------------------------------------------------------------------------
# Guard helpers
# ---------------------------------------------------------------------------


def check_tty() -> None:
    """Abort if stdin is not a terminal (non-interactive context)."""
    if not sys.stdin.isatty():
        typer.echo(
            "Error: Interactive mode requires a terminal. "
            "Use 'benchmark-qed config init' for non-interactive setup.",
            err=True,
        )
        raise typer.Exit(code=1)


def confirm_overwrite(path: Path | str) -> None:
    """Ask for confirmation before overwriting an existing settings file."""
    p = Path(path) if not isinstance(path, Path) else path
    if p.exists():
        typer.confirm(
            f"{p} already exists. Overwrite?",
            abort=True,
        )


# ---------------------------------------------------------------------------
# Selection / display primitives
# ---------------------------------------------------------------------------


def select_option(
    title: str,
    options: list[tuple[str, str]],
) -> str:
    """Display numbered options and return the selected value.

    Parameters
    ----------
    title:
        Prompt title shown to the user.
    options:
        List of ``(value, label)`` tuples.

    Returns
    -------
    The *value* string of the chosen option.
    """
    rich_print(f"\n[bold]{title}[/bold]")
    for idx, (_value, label) in enumerate(options, 1):
        rich_print(f"  [cyan][{idx}][/cyan] {label}")

    choice = typer.prompt(
        "Select",
        type=int,
        default=1,
    )
    if choice < 1 or choice > len(options):
        typer.echo("Invalid choice. Defaulting to 1.")
        choice = 1
    return options[choice - 1][0]


def show_section_defaults(title: str, fields: list[FieldDef]) -> None:
    """Render a Rich table showing current default values for a section."""
    table = Table(title=title, show_header=False, padding=(0, 2))
    table.add_column("Field", style="cyan", min_width=28)
    table.add_column("Default", style="green")
    for f in fields:
        table.add_row(f.name, str(f.default))
    rich_print(table)


def prompt_section(
    title: str,
    fields: list[FieldDef],
) -> dict[str, Any]:
    """Show section defaults and optionally let the user customise them.

    Returns a dict mapping field names to their (possibly user-overridden) values.
    """
    show_section_defaults(title, fields)

    if not typer.confirm("Customize this section?", default=False):
        return {f.name: f.default for f in fields}

    result: dict[str, Any] = {}
    for f in fields:
        if f.choices:
            value = select_option(f.description, [(c, c) for c in f.choices])
        else:
            raw = typer.prompt(
                f.name,
                default=f.default,
                type=f.field_type,
            )
            value = raw
        result[f.name] = value
    return result


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def _prompt_azure_init_args(provider_value: str) -> dict[str, Any]:
    """Prompt for Azure-specific init_args based on provider type."""
    init_args: dict[str, Any] = {}

    if "azure" in provider_value:
        endpoint = typer.prompt("Azure endpoint URL")
        init_args["azure_endpoint"] = endpoint

    if "azure.openai" in provider_value:
        api_version = typer.prompt("API version", default=DEFAULT_API_VERSION)
        init_args["api_version"] = api_version

    return init_args


def prompt_provider(
    purpose: str = "chat",
    *,
    default_model: str | None = None,
) -> ProviderResult:
    """Guide the user through LLM provider selection.

    Parameters
    ----------
    purpose:
        ``"chat"`` or ``"embedding"`` — determines available providers and default model.
    default_model:
        Override the default model name. If *None*, uses the standard default for the purpose.
    """
    providers = CHAT_PROVIDERS if purpose == "chat" else EMBEDDING_PROVIDERS
    model_default = default_model or (
        DEFAULT_CHAT_MODEL if purpose == "chat" else DEFAULT_EMBEDDING_MODEL
    )

    provider_value = select_option(
        f"Select {purpose} LLM provider",
        [(val, label) for val, label, _desc in providers],
    )

    # Auth type
    auth_type = select_option("Authentication type", AUTH_TYPES)

    # Provider-specific init args
    init_args = _prompt_azure_init_args(provider_value)

    # Model name
    model = typer.prompt("Model name", default=model_default)

    return ProviderResult(
        llm_provider=provider_value,
        model=model,
        auth_type=auth_type,
        init_args=init_args,
    )


def prompt_embedding_provider(
    chat_result: ProviderResult,
) -> ProviderResult:
    """Ask whether to reuse the chat provider for embeddings, or configure separately."""
    if typer.confirm("Use the same provider for embeddings?", default=True):
        # Derive the embedding provider from the chat provider
        mapping = {
            "openai.chat": "openai.embedding",
            "azure.openai.chat": "azure.openai.embedding",
            "azure.inference.chat": "azure.inference.embedding",
        }
        emb_provider = mapping.get(chat_result.llm_provider, "openai.embedding")
        return ProviderResult(
            llm_provider=emb_provider,
            model=DEFAULT_EMBEDDING_MODEL,
            auth_type=chat_result.auth_type,
            init_args=dict(chat_result.init_args),
        )
    return prompt_provider("embedding")


# ---------------------------------------------------------------------------
# List collection
# ---------------------------------------------------------------------------


def prompt_list_items(
    item_name: str,
    field_defs: list[FieldDef],
    *,
    min_items: int = 1,
) -> list[dict[str, Any]]:
    """Collect a list of items by prompting the user in a loop.

    Each iteration prompts for each field in *field_defs*, then asks
    "Add another <item_name>?".
    """
    items: list[dict[str, Any]] = []
    while True:
        rich_print(f"\n[bold]  {item_name} #{len(items) + 1}[/bold]")
        item: dict[str, Any] = {}
        for f in field_defs:
            raw = typer.prompt(f"  {f.name}", default=f.default, type=f.field_type)
            item[f.name] = raw
        items.append(item)

        if len(items) >= min_items:
            if not typer.confirm(f"Add another {item_name}?", default=False):
                break
        else:
            typer.echo(f"  (need at least {min_items})")
    return items


def prompt_comma_list(prompt_text: str, default: str = "") -> list[str]:
    """Prompt for a comma-separated list and return split values."""
    raw = typer.prompt(prompt_text, default=default)
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# AutoQ interactive configuration
# ---------------------------------------------------------------------------

_QUESTION_TYPES = [
    "data_local",
    "data_global",
    "data_linked",
    "activity_local",
    "activity_global",
]


def build_autoq_config() -> dict[str, Any]:
    """Walk the user through AutoQ configuration and return a render-ready dict.

    The returned dictionary contains every value needed to render the AutoQ
    YAML settings file.  Keys are organised into logical sections that mirror
    the wizard steps shown to the user.
    """
    rich_print(Panel("[bold]AutoQ — Question Generation[/bold]", expand=False))

    # ── 1. Chat LLM provider ──────────────────────────────────────────────
    chat_result = prompt_provider("chat")

    # ── 2. Embedding LLM ──────────────────────────────────────────────────
    embedding_result = prompt_embedding_provider(chat_result)

    # ── 3. Input section ──────────────────────────────────────────────────
    input_fields = [
        FieldDef("dataset_path", "Path to input dataset", "./input"),
        FieldDef("input_type", "Input file type", "json", choices=["csv", "json"]),
        FieldDef("text_column", "Column containing text", "text"),
        FieldDef(
            "metadata_columns",
            "Metadata columns (comma-separated)",
            "",
        ),
        FieldDef("file_encoding", "File encoding", "utf-8-sig"),
    ]
    input_values = prompt_section("Input", input_fields)

    # Normalise metadata_columns to a list or None
    raw_meta = input_values.get("metadata_columns", "")
    if isinstance(raw_meta, str):
        parts = [s.strip() for s in raw_meta.split(",") if s.strip()]
        input_values["metadata_columns"] = parts or None

    # ── 4. Encoding section ───────────────────────────────────────────────
    encoding_fields = [
        FieldDef("model_name", "Tokeniser model name", "o200k_base"),
        FieldDef("chunk_size", "Chunk size (tokens)", 600, field_type=int),
        FieldDef("chunk_overlap", "Chunk overlap (tokens)", 100, field_type=int),
    ]
    encoding_values = prompt_section("Encoding", encoding_fields)

    # ── 5. Sampling section ───────────────────────────────────────────────
    sampling_fields = [
        FieldDef("num_clusters", "Number of clusters", 20, field_type=int),
        FieldDef(
            "num_samples_per_cluster",
            "Samples per cluster",
            10,
            field_type=int,
        ),
        FieldDef("random_seed", "Random seed", 42, field_type=int),
    ]
    sampling_values = prompt_section("Sampling", sampling_fields)

    # ── 6. Question Types section ─────────────────────────────────────────
    qt_fields: list[FieldDef] = []
    for qt in _QUESTION_TYPES:
        qt_fields.extend([
            FieldDef(
                f"{qt}_num_questions",
                f"{qt} — number of questions",
                10,
                field_type=int,
            ),
            FieldDef(
                f"{qt}_oversample_factor",
                f"{qt} — oversample factor",
                2.0,
                field_type=float,
            ),
        ])

    qt_values = prompt_section("Question Types", qt_fields)

    # Reshape flat values into nested per-type dicts
    question_types: dict[str, dict[str, Any]] = {}
    customised_qt = qt_values != {f.name: f.default for f in qt_fields}
    for qt in _QUESTION_TYPES:
        question_types[qt] = {
            "num_questions": qt_values[f"{qt}_num_questions"],
            "oversample_factor": qt_values[f"{qt}_oversample_factor"],
        }

    # ── 7. Activity question params (only when QT section was customised) ─
    activity_defaults = {
        "num_personas": 5,
        "num_tasks_per_persona": 2,
        "num_entities_per_task": 5,
    }
    if customised_qt:
        activity_fields = [
            FieldDef("num_personas", "Number of personas", 5, field_type=int),
            FieldDef(
                "num_tasks_per_persona",
                "Tasks per persona",
                2,
                field_type=int,
            ),
            FieldDef(
                "num_entities_per_task",
                "Entities per task",
                5,
                field_type=int,
            ),
        ]
        activity_values = prompt_section("Activity Question Params", activity_fields)
    else:
        activity_values = dict(activity_defaults)

    # ── 8. Assertions section ─────────────────────────────────────────────
    assertions_fields = [
        FieldDef("max_assertions", "Max assertions per question", 20, field_type=int),
        FieldDef(
            "enable_validation",
            "Enable assertion validation",
            default=True,
            field_type=bool,
        ),
        FieldDef(
            "min_validation_score",
            "Minimum validation score",
            3,
            field_type=int,
        ),
    ]

    # Display defaults, then offer customisation.
    # We handle the bool field (enable_validation) specially via typer.confirm.
    show_section_defaults("Assertions", assertions_fields)
    if not typer.confirm("Customize this section?", default=False):
        assertions_values = {f.name: f.default for f in assertions_fields}
    else:
        assertions_values: dict[str, Any] = {}
        for f in assertions_fields:
            if f.field_type is bool:
                assertions_values[f.name] = typer.confirm(f.name, default=f.default)
            elif f.choices:
                assertions_values[f.name] = select_option(
                    f.description, [(c, c) for c in f.choices]
                )
            else:
                assertions_values[f.name] = typer.prompt(
                    f.name, default=f.default, type=f.field_type
                )

    # ── 9. Concurrency ────────────────────────────────────────────────────
    concurrent_requests = typer.prompt("Concurrent requests", default=8, type=int)

    # ── Build final config dict ───────────────────────────────────────────
    return {
        "chat_provider": chat_result,
        "embedding_provider": embedding_result,
        "input": input_values,
        "encoding": encoding_values,
        "sampling": sampling_values,
        "question_types": question_types,
        "activity_params": activity_values,
        "assertions": assertions_values,
        "concurrent_requests": concurrent_requests,
    }


# ---------------------------------------------------------------------------
# AutoE interactive configuration flows
# ---------------------------------------------------------------------------

_CONDITION_FIELDS = [
    FieldDef(
        name="name",
        description="Condition name",
        default="",
        field_type=str,
    ),
    FieldDef(
        name="answer_base_path",
        description="Path to answer files",
        default="input/method_name",
        field_type=str,
    ),
]


def _prompt_condition(label: str) -> dict[str, Any]:
    """Prompt for a single condition (name + answer_base_path)."""
    rich_print(f"\n[bold]  {label}[/bold]")
    name = typer.prompt("  name", default="")
    answer_base_path = typer.prompt("  answer_base_path", default="input/method_name")
    return {"name": name, "answer_base_path": answer_base_path}


def _prompt_even_trials(default: int = 4) -> int:
    """Prompt for a trial count and ensure it is even."""
    trials = typer.prompt("Number of trials (must be even)", default=default, type=int)
    if trials % 2 != 0:
        trials += 1
        typer.echo(f"  Trials must be even — rounded up to {trials}.")
    return trials


def build_autoe_pairwise_config() -> dict[str, Any]:
    """Interactive flow for AutoE pairwise evaluation configuration."""
    rich_print(Panel("AutoE — Pairwise Evaluation"))

    # LLM provider
    chat_provider = prompt_provider("chat")

    # Base condition
    base = _prompt_condition("Base condition")

    # Other conditions
    rich_print("\n[bold]Other conditions to compare against the base:[/bold]")
    others = prompt_list_items("condition", _CONDITION_FIELDS, min_items=1)

    # Question sets
    question_sets = prompt_comma_list(
        "Question sets (comma-separated)", "activity_global, activity_local"
    )

    # Trials
    trials = _prompt_even_trials()

    # Custom criteria
    criteria: list[dict[str, Any]] | None = None
    if typer.confirm("Add custom scoring criteria?", default=False):
        criteria = prompt_list_items(
            "criterion",
            [
                FieldDef(
                    name="name",
                    description="Criterion name",
                    default="",
                    field_type=str,
                ),
                FieldDef(
                    name="description",
                    description="Criterion description",
                    default="",
                    field_type=str,
                ),
            ],
        )

    return {
        "chat_provider": chat_provider,
        "base": base,
        "others": others,
        "question_sets": question_sets,
        "trials": trials,
        "criteria": criteria,
    }


def build_autoe_reference_config() -> dict[str, Any]:
    """Interactive flow for AutoE reference evaluation configuration."""
    rich_print(Panel("AutoE — Reference Evaluation"))

    # LLM provider
    chat_provider = prompt_provider("chat")

    # Reference condition
    reference = _prompt_condition("Reference condition")

    # Generated conditions
    rich_print("\n[bold]Generated conditions to evaluate:[/bold]")
    generated = prompt_list_items("generated condition", _CONDITION_FIELDS, min_items=1)

    # Score range
    score_min = typer.prompt("Score minimum", default=1, type=int)
    score_max = typer.prompt("Score maximum", default=10, type=int)

    # Trials
    trials = _prompt_even_trials()

    return {
        "chat_provider": chat_provider,
        "reference": reference,
        "generated": generated,
        "score_min": score_min,
        "score_max": score_max,
        "trials": trials,
    }


def build_autoe_assertion_config() -> dict[str, Any]:
    """Interactive flow for AutoE assertion evaluation configuration."""
    rich_print(Panel("AutoE — Assertion Evaluation"))

    # LLM provider
    chat_provider = prompt_provider("chat")

    # Generated condition
    generated = _prompt_condition("Generated condition")

    # Assertions path
    assertions_path = typer.prompt(
        "Path to assertions file", default="input/assertions.json"
    )

    # Pass threshold
    pass_threshold = typer.prompt("Pass threshold", default=0.5, type=float)

    # Trials
    trials = typer.prompt("Number of trials", default=4, type=int)

    return {
        "chat_provider": chat_provider,
        "generated": generated,
        "assertions": {"assertions_path": assertions_path},
        "pass_threshold": pass_threshold,
        "trials": trials,
    }


# ---------------------------------------------------------------------------
# Config type metadata
# ---------------------------------------------------------------------------

CONFIG_TYPE_OPTIONS: list[tuple[str, str]] = [
    ("autoq", "AutoQ — Question Generation"),
    ("autoe_pairwise", "AutoE — Pairwise Evaluation"),
    ("autoe_reference", "AutoE — Reference Evaluation"),
    ("autoe_assertion", "AutoE — Assertion Evaluation"),
]


# ---------------------------------------------------------------------------
# Prompt copying orchestration
# ---------------------------------------------------------------------------


def _copy_prompts_for_config(config_type: str, prompts_folder: Path) -> None:
    """Copy the appropriate prompt templates for the given config type."""
    match config_type:
        case "autoq":
            copy_prompts(
                Path(summarization.__file__).parent,
                prompts_folder / "summarization",
            )
            copy_prompts(
                Path(activity_context_prompts.__file__).parent,
                prompts_folder / "activity_questions" / "activity_context",
            )
            copy_prompts(
                Path(activity_global_prompts.__file__).parent,
                prompts_folder / "activity_questions" / "activity_global",
            )
            copy_prompts(
                Path(activity_local_prompts.__file__).parent,
                prompts_folder / "activity_questions" / "activity_local",
            )
            copy_prompts(
                Path(data_global_prompts.__file__).parent,
                prompts_folder / "data_questions" / "data_global",
            )
            copy_prompts(
                Path(data_local_prompts.__file__).parent,
                prompts_folder / "data_questions" / "data_local",
            )
            copy_prompts(
                Path(data_linked_prompts.__file__).parent,
                prompts_folder / "data_questions" / "data_linked",
            )
            copy_prompts(
                Path(data_questions_prompts.__file__).parent,
                prompts_folder / "data_questions",
            )
            copy_prompts(
                Path(autoq_assertion_prompts.__file__).parent,
                prompts_folder / "data_questions" / "assertions",
            )
        case "autoe_pairwise":
            copy_prompts(Path(pairwise_prompts.__file__).parent, prompts_folder)
        case "autoe_reference":
            copy_prompts(Path(reference_prompts.__file__).parent, prompts_folder)
        case "autoe_assertion":
            copy_prompts(Path(assertion_prompts.__file__).parent, prompts_folder)


# ---------------------------------------------------------------------------
# Main init command
# ---------------------------------------------------------------------------


@app.command()
def interactive_init(
    root: Annotated[
        Path,
        typer.Argument(help="The root directory for the new benchmark project."),
    ],
) -> None:
    """Interactively create a benchmark-qed configuration."""
    check_tty()

    rich_print(
        Panel(
            "[bold]benchmark-qed[/bold] — Interactive Configuration Wizard",
            subtitle="Press Enter to accept defaults",
        )
    )

    # 1. Select config type
    config_type = select_option("Select configuration type", CONFIG_TYPE_OPTIONS)

    # 2. Run the appropriate builder
    builders = {
        "autoq": build_autoq_config,
        "autoe_pairwise": build_autoe_pairwise_config,
        "autoe_reference": build_autoe_reference_config,
        "autoe_assertion": build_autoe_assertion_config,
    }
    config_dict = builders[config_type]()

    # 3. Render YAML
    renderers = {
        "autoq": render_autoq_yaml,
        "autoe_pairwise": render_autoe_pairwise_yaml,
        "autoe_reference": render_autoe_reference_yaml,
        "autoe_assertion": render_autoe_assertion_yaml,
    }
    yaml_content = renderers[config_type](config_dict)

    # 4. Validate against Pydantic model
    validate_config(yaml_content, config_type)

    # 5. Write files
    root.mkdir(parents=True, exist_ok=True)
    settings_path = root / "settings.yaml"
    confirm_overwrite(settings_path)
    settings_path.write_text(yaml_content, encoding="utf-8")

    prompts_folder = root / "prompts"
    _copy_prompts_for_config(config_type, prompts_folder)

    ensure_input_folder(root)
    write_env_file(root)

    # 6. Success summary
    rich_print(f"\n[green]✅ Configuration created at {settings_path}[/green]")
    rich_print(f"[green]✅ Prompt templates copied to {prompts_folder}/[/green]")
    rich_print(
        "[green]✅ .env file created — update OPENAI_API_KEY before running[/green]"
    )
