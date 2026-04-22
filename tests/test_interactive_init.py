# Copyright (c) 2025 Microsoft Corporation.
"""Tests for the interactive init wizard and YAML renderers."""

from __future__ import annotations

from typing import Any

import pytest
import typer
import yaml
from typer.testing import CliRunner

from benchmark_qed.__main__ import app
from benchmark_qed.cli.interactive import (
    prompt_comma_list,
    select_option,
)
from benchmark_qed.cli.yaml_renderer import (
    _render_llm_section,
    render_autoe_assertion_yaml,
    render_autoe_pairwise_yaml,
    render_autoe_reference_yaml,
    render_autoq_yaml,
    validate_config,
)

# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------


def _openai_chat_provider() -> dict[str, Any]:
    return {
        "llm_provider": "openai.chat",
        "model": "gpt-4.1",
        "auth_type": "api_key",
        "init_args": {},
    }


def _openai_embedding_provider() -> dict[str, Any]:
    return {
        "llm_provider": "openai.embedding",
        "model": "text-embedding-3-large",
        "auth_type": "api_key",
        "init_args": {},
    }


def _azure_chat_provider() -> dict[str, Any]:
    return {
        "llm_provider": "azure.openai.chat",
        "model": "gpt-4.1",
        "auth_type": "api_key",
        "init_args": {
            "azure_endpoint": "https://example.openai.azure.com",
            "api_version": "2024-12-01-preview",
        },
    }


def _azure_managed_identity_provider() -> dict[str, Any]:
    return {
        "llm_provider": "azure.openai.chat",
        "model": "gpt-4.1",
        "auth_type": "azure_managed_identity",
        "init_args": {
            "azure_endpoint": "https://example.openai.azure.com",
            "api_version": "2024-12-01-preview",
        },
    }


def _default_autoq_config() -> dict[str, Any]:
    return {
        "chat_provider": _openai_chat_provider(),
        "embedding_provider": _openai_embedding_provider(),
        "input": {
            "dataset_path": "./input",
            "input_type": "json",
            "text_column": "text",
            "metadata_columns": None,
            "file_encoding": "utf-8",
        },
        "encoding": {
            "model_name": "o200k_base",
            "chunk_size": 600,
            "chunk_overlap": 100,
        },
        "sampling": {
            "num_clusters": 20,
            "num_samples_per_cluster": 10,
            "random_seed": 42,
        },
        "question_types": {
            qt: {"num_questions": 10, "oversample_factor": 2.0}
            for qt in [
                "data_local",
                "data_global",
                "data_linked",
                "activity_local",
                "activity_global",
            ]
        },
        "activity_params": {
            "num_personas": 5,
            "num_tasks_per_persona": 2,
            "num_entities_per_task": 5,
        },
        "assertions": {
            "max_assertions": 20,
            "enable_validation": True,
            "min_validation_score": 3,
        },
        "concurrent_requests": 8,
    }


def _default_pairwise_config() -> dict[str, Any]:
    return {
        "chat_provider": _openai_chat_provider(),
        "base": {"name": "baseline", "answer_base_path": "input/baseline"},
        "others": [
            {"name": "method_a", "answer_base_path": "input/method_a"},
        ],
        "question_sets": ["activity_global", "activity_local"],
        "trials": 4,
        "criteria": None,
    }


def _default_reference_config() -> dict[str, Any]:
    return {
        "chat_provider": _openai_chat_provider(),
        "reference": {"name": "golden", "answer_base_path": "input/golden"},
        "generated": [
            {"name": "method_a", "answer_base_path": "input/method_a"},
        ],
        "score_min": 1,
        "score_max": 10,
        "trials": 4,
    }


def _default_assertion_config() -> dict[str, Any]:
    return {
        "chat_provider": _openai_chat_provider(),
        "generated": {"name": "method_a", "answer_base_path": "input/method_a"},
        "assertions": {"assertions_path": "input/assertions.json"},
        "pass_threshold": 0.5,
        "trials": 4,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. YAML Renderer Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestYamlRenderers:
    """Verify each renderer produces valid, parseable YAML."""

    def test_render_autoq_yaml_produces_valid_yaml(self):
        """render_autoq_yaml returns parseable YAML with all expected sections."""
        config = _default_autoq_config()
        yaml_content = render_autoq_yaml(config)
        parsed = yaml.safe_load(yaml_content)
        assert parsed is not None
        assert "input" in parsed
        assert "chat_model" in parsed
        assert "embedding_model" in parsed
        assert "sampling" in parsed
        assert parsed["concurrent_requests"] == 8

    def test_render_autoq_yaml_includes_question_types(self):
        """AutoQ YAML includes all five question type sections."""
        config = _default_autoq_config()
        yaml_content = render_autoq_yaml(config)
        parsed = yaml.safe_load(yaml_content)
        for qt in [
            "data_local",
            "data_global",
            "data_linked",
            "activity_local",
            "activity_global",
        ]:
            assert qt in parsed, f"Missing question type section: {qt}"
            assert "num_questions" in parsed[qt]

    def test_render_autoq_yaml_includes_assertions(self):
        """AutoQ YAML includes assertions section with local/global/linked."""
        config = _default_autoq_config()
        yaml_content = render_autoq_yaml(config)
        parsed = yaml.safe_load(yaml_content)
        assert "assertions" in parsed
        for section in ["local", "global", "linked"]:
            assert section in parsed["assertions"]

    def test_render_autoq_yaml_with_metadata_columns(self):
        """Metadata columns are included when provided."""
        config = _default_autoq_config()
        config["input"]["metadata_columns"] = ["source", "date"]
        yaml_content = render_autoq_yaml(config)
        parsed = yaml.safe_load(yaml_content)
        assert parsed["input"]["metadata_columns"] == ["source", "date"]

    def test_render_autoq_yaml_without_metadata_columns(self):
        """No metadata_columns key when None is provided."""
        config = _default_autoq_config()
        config["input"]["metadata_columns"] = None
        yaml_content = render_autoq_yaml(config)
        parsed = yaml.safe_load(yaml_content)
        assert "metadata_columns" not in parsed["input"]

    def test_render_autoq_yaml_includes_prompt_configs(self):
        """AutoQ YAML includes prompt configuration sections."""
        config = _default_autoq_config()
        yaml_content = render_autoq_yaml(config)
        parsed = yaml.safe_load(yaml_content)
        assert "activity_questions_prompt_config" in parsed
        assert "data_questions_prompt_config" in parsed
        assert "assertion_prompts" in parsed

    def test_render_autoe_pairwise_yaml(self):
        """Pairwise YAML includes base, others, question_sets."""
        config = _default_pairwise_config()
        yaml_content = render_autoe_pairwise_yaml(config)
        parsed = yaml.safe_load(yaml_content)
        assert parsed is not None
        assert "base" in parsed
        assert "others" in parsed
        assert "question_sets" in parsed
        assert parsed["trials"] == 4
        assert "llm_config" in parsed

    def test_render_autoe_pairwise_yaml_with_criteria(self):
        """Pairwise YAML with custom criteria includes them."""
        config = _default_pairwise_config()
        config["criteria"] = [
            {"name": "accuracy", "description": "Is the answer correct?"},
        ]
        yaml_content = render_autoe_pairwise_yaml(config)
        parsed = yaml.safe_load(yaml_content)
        assert "criteria" in parsed
        assert len(parsed["criteria"]) == 1

    def test_render_autoe_pairwise_yaml_no_criteria_is_commented(self):
        """Pairwise YAML without criteria has commented-out criteria block."""
        config = _default_pairwise_config()
        config["criteria"] = None
        yaml_content = render_autoe_pairwise_yaml(config)
        assert "# criteria:" in yaml_content

    def test_render_autoe_reference_yaml(self):
        """Reference YAML includes score range."""
        config = _default_reference_config()
        yaml_content = render_autoe_reference_yaml(config)
        parsed = yaml.safe_load(yaml_content)
        assert parsed is not None
        assert "reference" in parsed
        assert "generated" in parsed
        assert parsed["score_min"] == 1
        assert parsed["score_max"] == 10
        assert "llm_config" in parsed

    def test_render_autoe_reference_yaml_multiple_generated(self):
        """Reference YAML with multiple generated conditions."""
        config = _default_reference_config()
        config["generated"].append({
            "name": "method_b",
            "answer_base_path": "input/method_b",
        })
        yaml_content = render_autoe_reference_yaml(config)
        parsed = yaml.safe_load(yaml_content)
        assert len(parsed["generated"]) == 2

    def test_render_autoe_assertion_yaml(self):
        """Assertion YAML includes pass_threshold."""
        config = _default_assertion_config()
        yaml_content = render_autoe_assertion_yaml(config)
        parsed = yaml.safe_load(yaml_content)
        assert parsed is not None
        assert "generated" in parsed
        assert "assertions" in parsed
        assert parsed["pass_threshold"] == 0.5
        assert parsed["trials"] == 4
        assert "llm_config" in parsed


# ═══════════════════════════════════════════════════════════════════════════
# 2. LLM Section Rendering Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLlmRendering:
    """Verify _render_llm_section output for different provider configs."""

    def test_openai_provider_includes_api_key(self):
        """OpenAI provider with api_key auth includes ${OPENAI_API_KEY}."""
        provider = _openai_chat_provider()
        section = _render_llm_section(provider)
        assert "${OPENAI_API_KEY}" in section
        assert "model: gpt-4.1" in section
        assert "llm_provider: openai.chat" in section

    def test_azure_provider_includes_init_args(self):
        """Azure provider renders azure_endpoint and api_version in init_args."""
        provider = _azure_chat_provider()
        section = _render_llm_section(provider)
        assert "init_args:" in section
        assert "azure_endpoint: https://example.openai.azure.com" in section
        assert "api_version: 2024-12-01-preview" in section

    def test_managed_identity_omits_api_key(self):
        """azure_managed_identity auth type does NOT include api_key line."""
        provider = _azure_managed_identity_provider()
        section = _render_llm_section(provider)
        assert "api_key:" not in section
        assert "auth_type: azure_managed_identity" in section

    def test_openai_provider_no_init_args_block(self):
        """OpenAI provider with empty init_args omits init_args block."""
        provider = _openai_chat_provider()
        section = _render_llm_section(provider)
        assert "init_args:" not in section

    def test_section_includes_concurrent_requests(self):
        """Every LLM section includes concurrent_requests."""
        provider = _openai_chat_provider()
        section = _render_llm_section(provider)
        assert "concurrent_requests: 4" in section

    def test_custom_indent(self):
        """Custom indent produces properly indented output."""
        provider = _openai_chat_provider()
        section = _render_llm_section(provider, indent=4)
        for line in section.split("\n"):
            assert line.startswith("    ")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Validation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestValidation:
    """Verify validate_config accepts valid YAML and rejects invalid YAML."""

    def test_validate_config_passes_valid_autoq(self):
        """Valid AutoQ YAML passes validation."""
        yaml_content = render_autoq_yaml(_default_autoq_config())
        validate_config(yaml_content, "autoq")

    def test_validate_config_passes_valid_pairwise(self):
        """Valid pairwise YAML passes validation."""
        yaml_content = render_autoe_pairwise_yaml(_default_pairwise_config())
        validate_config(yaml_content, "autoe_pairwise")

    def test_validate_config_passes_valid_reference(self):
        """Valid reference YAML passes validation."""
        yaml_content = render_autoe_reference_yaml(_default_reference_config())
        validate_config(yaml_content, "autoe_reference")

    def test_validate_config_passes_valid_assertion(self):
        """Valid assertion YAML passes validation."""
        yaml_content = render_autoe_assertion_yaml(_default_assertion_config())
        validate_config(yaml_content, "autoe_assertion")

    def test_validate_config_rejects_invalid_yaml(self):
        """Malformed YAML is rejected."""
        with pytest.raises(typer.BadParameter):
            validate_config(": :\n  bad: [", "autoq")

    def test_validate_config_rejects_missing_keys(self):
        """YAML missing required keys is rejected."""
        with pytest.raises(typer.BadParameter, match="Missing required keys"):
            validate_config("foo: bar\n", "autoq")

    def test_validate_config_rejects_wrong_type(self):
        """YAML with wrong types is rejected."""
        yaml_content = render_autoq_yaml(_default_autoq_config())
        parsed = yaml.safe_load(yaml_content)
        parsed["input"] = "not_a_dict"
        bad_yaml = yaml.dump(parsed)
        with pytest.raises(typer.BadParameter, match="should be dict"):
            validate_config(bad_yaml, "autoq")

    def test_validate_config_rejects_unknown_type(self):
        """Unknown config type is rejected."""
        with pytest.raises(typer.BadParameter, match="Unknown config type"):
            validate_config("foo: bar\n", "unknown_type")

    def test_validate_config_rejects_non_mapping_root(self):
        """YAML whose root is not a mapping is rejected."""
        with pytest.raises(typer.BadParameter, match="root must be a mapping"):
            validate_config("- item1\n- item2\n", "autoq")


# ═══════════════════════════════════════════════════════════════════════════
# 4. CLI Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCliIntegration:
    """End-to-end CLI tests using typer.testing.CliRunner."""

    @pytest.fixture(autouse=True)
    def _patch_tty_check(self, monkeypatch):
        """Disable the TTY check so CliRunner can drive the wizard."""
        monkeypatch.setattr("benchmark_qed.cli.interactive.check_tty", lambda: None)

    def test_init_autoq_creates_files(self, tmp_path):
        """benchmark-qed init creates settings.yaml, prompts/, .env."""
        runner = CliRunner()
        # Input sequence: see build_autoq_config for prompt order.
        input_lines = [
            "1",  # config type: autoq
            "1",  # chat provider: OpenAI
            "1",  # auth type: api_key
            "",  # model: accept default gpt-4.1
            "Y",  # use same provider for embeddings
            "N",  # customize input section
            "N",  # customize encoding section
            "N",  # customize sampling section
            "N",  # customize question types section
            "N",  # customize assertions section
            "8",  # concurrent requests
        ]
        input_text = "\n".join(input_lines) + "\n"
        result = runner.invoke(
            app,
            ["init", str(tmp_path)],
            input=input_text,
        )
        assert result.exit_code == 0, (
            f"CLI failed (code={result.exit_code}):\n{result.output}"
        )
        assert (tmp_path / "settings.yaml").exists()
        assert (tmp_path / "prompts").exists()
        assert (tmp_path / ".env").exists()

    def test_init_autoq_settings_yaml_is_valid(self, tmp_path):
        """The generated settings.yaml is parseable and contains expected keys."""
        runner = CliRunner()
        input_lines = [
            "1",
            "1",
            "1",
            "",
            "Y",
            "N",
            "N",
            "N",
            "N",
            "N",
            "8",
        ]
        input_text = "\n".join(input_lines) + "\n"
        result = runner.invoke(
            app,
            ["init", str(tmp_path)],
            input=input_text,
        )
        assert result.exit_code == 0, result.output

        settings = yaml.safe_load(
            (tmp_path / "settings.yaml").read_text(encoding="utf-8")
        )
        assert "chat_model" in settings
        assert "embedding_model" in settings
        assert "input" in settings

    def test_init_autoe_pairwise_creates_files(self, tmp_path):
        """Pairwise init creates correct files."""
        runner = CliRunner()
        input_lines = [
            "2",  # config type: autoe_pairwise
            "1",  # chat provider: OpenAI
            "1",  # auth type: api_key
            "",  # model: default
            "baseline",  # base condition name
            "input/baseline",  # base answer_base_path
            "method_a",  # other condition #1 name
            "input/method_a",  # other condition #1 answer_base_path
            "N",  # add another condition? no
            "",  # question sets: accept default
            "4",  # trials (even)
            "N",  # add custom criteria? no
        ]
        input_text = "\n".join(input_lines) + "\n"
        result = runner.invoke(
            app,
            ["init", str(tmp_path)],
            input=input_text,
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"
        assert (tmp_path / "settings.yaml").exists()
        assert (tmp_path / "prompts").exists()

    def test_init_autoe_reference_creates_files(self, tmp_path):
        """Reference init creates correct files."""
        runner = CliRunner()
        input_lines = [
            "3",  # config type: autoe_reference
            "1",  # chat provider: OpenAI
            "1",  # auth type: api_key
            "",  # model: default
            "golden",  # reference condition name
            "input/golden",  # reference answer_base_path
            "method_a",  # generated condition #1 name
            "input/method_a",  # generated condition #1 answer_base_path
            "N",  # add another generated? no
            "1",  # score min
            "10",  # score max
            "4",  # trials
        ]
        input_text = "\n".join(input_lines) + "\n"
        result = runner.invoke(
            app,
            ["init", str(tmp_path)],
            input=input_text,
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"
        assert (tmp_path / "settings.yaml").exists()

    def test_init_autoe_assertion_creates_files(self, tmp_path):
        """Assertion init creates correct files."""
        runner = CliRunner()
        input_lines = [
            "4",  # config type: autoe_assertion
            "1",  # chat provider: OpenAI
            "1",  # auth type: api_key
            "",  # model: default
            "method_a",  # generated condition name
            "input/method_a",  # generated answer_base_path
            "",  # assertions path: accept default
            "0.5",  # pass threshold
            "4",  # trials
        ]
        input_text = "\n".join(input_lines) + "\n"
        result = runner.invoke(
            app,
            ["init", str(tmp_path)],
            input=input_text,
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"
        assert (tmp_path / "settings.yaml").exists()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Overwrite Protection Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOverwriteProtection:
    """Verify overwrite protection for existing settings files."""

    @pytest.fixture(autouse=True)
    def _patch_tty_check(self, monkeypatch):
        monkeypatch.setattr("benchmark_qed.cli.interactive.check_tty", lambda: None)

    def _autoq_default_input(self) -> str:
        """Input sequence that walks through autoq with all defaults."""
        lines = ["1", "1", "1", "", "Y", "N", "N", "N", "N", "N", "8"]
        return "\n".join(lines) + "\n"

    def test_init_warns_on_existing_settings(self, tmp_path):
        """If settings.yaml exists, overwrite confirmation is prompted."""
        (tmp_path / "settings.yaml").write_text("existing", encoding="utf-8")
        runner = CliRunner()
        # Same autoq defaults + "y" for overwrite confirmation
        input_text = self._autoq_default_input() + "y\n"
        result = runner.invoke(
            app,
            ["init", str(tmp_path)],
            input=input_text,
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"
        content = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
        assert content != "existing"

    def test_init_aborts_on_overwrite_decline(self, tmp_path):
        """Declining overwrite aborts the command."""
        (tmp_path / "settings.yaml").write_text("existing", encoding="utf-8")
        runner = CliRunner()
        input_text = self._autoq_default_input() + "N\n"
        result = runner.invoke(
            app,
            ["init", str(tmp_path)],
            input=input_text,
        )
        assert result.exit_code != 0
        content = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
        assert content == "existing"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Helper Function Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHelpers:
    """Tests for individual interactive helper functions."""

    def test_prompt_comma_list_splits(self):
        """prompt_comma_list splits comma-separated input."""
        result_holder: list[str] = []
        test_app = typer.Typer()

        @test_app.command()
        def _cmd() -> None:
            result_holder.extend(prompt_comma_list("Enter items", default=""))

        runner = CliRunner()
        runner.invoke(test_app, input="alpha, beta, gamma\n")
        assert result_holder == ["alpha", "beta", "gamma"]

    def test_prompt_comma_list_uses_default(self):
        """prompt_comma_list uses default when input is empty."""
        result_holder: list[str] = []
        test_app = typer.Typer()

        @test_app.command()
        def _cmd() -> None:
            result_holder.extend(prompt_comma_list("Enter items", default="a, b"))

        runner = CliRunner()
        runner.invoke(test_app, input="\n")
        assert result_holder == ["a", "b"]

    def test_prompt_comma_list_strips_whitespace(self):
        """prompt_comma_list strips whitespace from items."""
        result_holder: list[str] = []
        test_app = typer.Typer()

        @test_app.command()
        def _cmd() -> None:
            result_holder.extend(prompt_comma_list("Enter", default=""))

        runner = CliRunner()
        runner.invoke(test_app, input="  x ,  y  , z  \n")
        assert result_holder == ["x", "y", "z"]

    def test_select_option_returns_correct_value(self):
        """select_option returns the value of the chosen option."""
        result_holder: list[str] = []
        test_app = typer.Typer()

        @test_app.command()
        def _cmd() -> None:
            val = select_option(
                "Pick one",
                [("val_a", "Label A"), ("val_b", "Label B")],
            )
            result_holder.append(val)

        runner = CliRunner()
        runner.invoke(test_app, input="2\n")
        assert result_holder == ["val_b"]

    def test_select_option_out_of_range_defaults(self):
        """Out-of-range selection defaults to option 1."""
        result_holder: list[str] = []
        test_app = typer.Typer()

        @test_app.command()
        def _cmd() -> None:
            val = select_option(
                "Pick one",
                [("first", "First"), ("second", "Second")],
            )
            result_holder.append(val)

        runner = CliRunner()
        runner.invoke(test_app, input="99\n")
        assert result_holder == ["first"]

    def test_select_option_default_is_one(self):
        """Empty input (Enter) defaults to option 1."""
        result_holder: list[str] = []
        test_app = typer.Typer()

        @test_app.command()
        def _cmd() -> None:
            val = select_option(
                "Pick one",
                [("default_val", "Default"), ("other", "Other")],
            )
            result_holder.append(val)

        runner = CliRunner()
        runner.invoke(test_app, input="\n")
        assert result_holder == ["default_val"]
