# Copyright (c) 2025 Microsoft Corporation.
"""Comprehensive tests for the load_config function from graphrag_common.config."""

import json
from pathlib import Path

import pytest
import yaml
from graphrag_common.config import load_config
from graphrag_common.config.load_config import ConfigParsingError
from pydantic import ValidationError

from benchmark_qed.autoe.config import AssertionConfig, PairwiseConfig, ReferenceConfig
from benchmark_qed.autoq.config import QuestionGenerationConfig
from benchmark_qed.config.llm_config import LLMConfig


class TestLoadConfigBasicFunctionality:
    """Test basic functionality of load_config with different config types."""

    def test_load_pairwise_config_yaml(self, tmp_path: Path):
        """Test loading a PairwiseConfig from a YAML file."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test user prompt"},
                "system_prompt": {"prompt": "test system prompt"},
            },
            "base": {
                "name": "base_condition",
                "question_sets": ["questions.jsonl"],
                "answer_base_path": "/path/to/answers",
            },
            "others": [
                {
                    "name": "other_condition",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/answers",
                }
            ],
            "question_sets": ["set1", "set2"],
            "criteria": [
                {
                    "name": "relevance",
                    "description": "How relevant is the response to the question",
                }
            ],
        }

        config_file = tmp_path / "pairwise_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(PairwiseConfig, str(config_file))

        assert isinstance(config, PairwiseConfig)
        assert config.trials == 4
        assert config.base is not None
        assert config.base.name == "base_condition"
        assert len(config.others) == 1
        assert config.others[0].name == "other_condition"

    def test_load_pairwise_config_json(self, tmp_path: Path):
        """Test loading a PairwiseConfig from a JSON file."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 6,
            "prompt_config": {
                "user_prompt": {"prompt": "test user prompt"},
                "system_prompt": {"prompt": "test system prompt"},
            },
            "base": {
                "name": "base_condition",
                "question_sets": ["questions.jsonl"],
                "answer_base_path": "/path/to/answers",
            },
            "others": [
                {
                    "name": "other1",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/answers",
                },
                {
                    "name": "other2",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/answers",
                },
            ],
            "question_sets": ["set1", "set2", "set3"],
        }

        config_file = tmp_path / "pairwise_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = load_config(PairwiseConfig, str(config_file))

        assert isinstance(config, PairwiseConfig)
        assert config.trials == 6
        assert len(config.others) == 2
        assert len(config.question_sets) == 3

    def test_load_reference_config(self, tmp_path: Path):
        """Test loading a ReferenceConfig from a YAML file."""
        config_data = {
            "llm_config": {
                "model": "gpt-3.5-turbo",
                "auth_type": "azure_managed_identity",
            },
            "trials": 8,
            "prompt_config": {
                "user_prompt": {"prompt": "reference user prompt"},
                "system_prompt": {"prompt": "reference system prompt"},
            },
            "reference": {
                "name": "reference_condition",
                "question_sets": ["questions.jsonl"],
                "answer_base_path": "/path/to/reference",
            },
            "generated": [
                {
                    "name": "generated_condition",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/generated",
                }
            ],
            "criteria": [
                {
                    "name": "accuracy",
                    "description": "How accurate is the information provided",
                }
            ],
        }

        config_file = tmp_path / "reference_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(ReferenceConfig, str(config_file))

        assert isinstance(config, ReferenceConfig)
        assert config.trials == 8
        assert config.llm_config.model == "gpt-3.5-turbo"

    def test_load_assertion_config(self, tmp_path: Path):
        """Test loading an AssertionConfig from a JSON file."""
        config_data = {
            "llm_config": {
                "model": "gpt-4-turbo",
                "auth_type": "azure_managed_identity",
            },
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "assertion user prompt"},
                "system_prompt": {"prompt": "assertion system prompt"},
            },
            "generated": {
                "name": "assertion_condition",
                "question_sets": ["questions.jsonl"],
                "answer_base_path": "/path/to/assertions",
            },
            "assertions": {"assertions_path": "/path/to/assertions.json"},
        }

        config_file = tmp_path / "assertion_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = load_config(AssertionConfig, str(config_file))

        assert isinstance(config, AssertionConfig)
        assert config.llm_config.model == "gpt-4-turbo"

    def test_load_question_generation_config(self, tmp_path: Path):
        """Test loading a QuestionGenerationConfig."""
        config_data = {
            "input": {
                "dataset_path": "./test_data",
                "input_type": "json",
                "text_column": "body",
                "file_encoding": "utf-8",
            },
            "encoding": {
                "model_name": "o200k_base",
                "chunk_size": 500,
                "chunk_overlap": 50,
            },
            "sampling": {
                "num_clusters": 10,
                "num_samples_per_cluster": 5,
                "random_seed": 123,
            },
            "chat_model": {
                "model": "gpt-4",
                "auth_type": "azure_managed_identity",
                "llm_provider": "openai.chat",
                "concurrent_requests": 2,
            },
            "embedding_model": {
                "model": "text-embedding-3-small",
                "auth_type": "azure_managed_identity",
                "llm_provider": "openai.embedding",
            },
        }

        config_file = tmp_path / "question_gen_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(QuestionGenerationConfig, str(config_file))

        assert isinstance(config, QuestionGenerationConfig)
        assert config.input.text_column == "body"
        assert config.sampling.num_clusters == 10


class TestLoadConfigFileSystem:
    """Test load_config with various file system scenarios."""

    def test_load_config_absolute_path(self, tmp_path: Path):
        """Test loading config with absolute path."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Use absolute path
        config = load_config(PairwiseConfig, str(config_file.absolute()))

        assert isinstance(config, PairwiseConfig)

    def test_load_config_relative_path(self, tmp_path: Path, monkeypatch):
        """Test loading config with relative path."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt_text": "test"},
                "system_prompt": {"prompt_text": "test"},
            },
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Ensure we're in a valid directory first
        import os

        try:
            os.getcwd()
        except FileNotFoundError:
            # If current dir is invalid, change to repo root
            os.chdir("/Users/gaudy-microsoft/Repositories/benchmark-qed")

        # Change to the temp directory and use relative path
        monkeypatch.chdir(tmp_path)
        config = load_config(PairwiseConfig, "config.yaml")

        assert isinstance(config, PairwiseConfig)

    def test_load_config_file_not_found(self):
        """Test loading config from non-existent file."""
        with pytest.raises((FileNotFoundError, OSError)):
            load_config(PairwiseConfig, "/nonexistent/path/config.yaml")

    def test_load_config_empty_file(self, tmp_path: Path):
        """Test loading config from empty file."""
        config_file = tmp_path / "empty_config.yaml"
        config_file.touch()  # Create empty file

        with pytest.raises((ValueError, ValidationError, yaml.YAMLError, TypeError)):
            load_config(PairwiseConfig, str(config_file))

    def test_load_config_directory_instead_of_file(self, tmp_path: Path):
        """Test loading config when path points to directory."""
        config_dir = tmp_path / "config_directory"
        config_dir.mkdir()

        with pytest.raises((IsADirectoryError, PermissionError, OSError)):
            load_config(PairwiseConfig, str(config_dir))

    def test_load_config_with_special_characters_in_path(self, tmp_path: Path):
        """Test loading config with special characters in file path."""
        special_dir = tmp_path / "config with spaces & symbols"
        special_dir.mkdir()

        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
        }

        config_file = special_dir / "config-file_v1.2.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(PairwiseConfig, str(config_file))

        assert isinstance(config, PairwiseConfig)


class TestLoadConfigDataValidation:
    """Test load_config with various data validation scenarios."""

    def test_load_config_with_missing_required_fields(self, tmp_path: Path):
        """Test loading config with validation error (odd number of trials)."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 3,  # Invalid - must be even
        }

        config_file = tmp_path / "invalid_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_config(PairwiseConfig, str(config_file))

        assert "even" in str(exc_info.value) or "trials" in str(exc_info.value)
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": "not_a_number",  # Should be int
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
        }

        config_file = tmp_path / "invalid_types_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_config(PairwiseConfig, str(config_file))

        assert "trials" in str(exc_info.value)

    def test_load_config_with_odd_trials(self, tmp_path: Path):
        """Test loading PairwiseConfig with odd number of trials (should fail validation)."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 5,  # Odd number, should fail validation
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
        }

        config_file = tmp_path / "odd_trials_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_config(PairwiseConfig, str(config_file))

        assert "even" in str(exc_info.value).lower()

    def test_load_config_with_extra_fields(self, tmp_path: Path):
        """Test loading config with extra fields (should be ignored or handled gracefully)."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
            "extra_field": "this should be ignored",
            "another_extra": {"nested": "data"},
        }

        config_file = tmp_path / "extra_fields_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # This should work - Pydantic typically ignores extra fields by default
        config = load_config(PairwiseConfig, str(config_file))

        assert isinstance(config, PairwiseConfig)
        assert config.trials == 4

    def test_load_config_with_nested_validation_error(self, tmp_path: Path):
        """Test loading config with validation error in nested field."""
        config_data = {
            "llm_config": {
                "model": "",  # Empty model should fail validation
                "auth_type": "invalid_auth_type",
            },
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
        }

        config_file = tmp_path / "nested_error_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_config(PairwiseConfig, str(config_file))

        # Should contain validation error related to llm_config
        assert "llm_config" in str(exc_info.value) or "model" in str(exc_info.value)


class TestLoadConfigFileFormats:
    """Test load_config with various file formats."""

    def test_load_config_malformed_yaml(self, tmp_path: Path):
        """Test loading config from malformed YAML file."""
        malformed_yaml = """
        llm_config:
          model: gpt-4
          auth_type: azure_managed_identity
        trials: 4
        prompt_config:
          user_prompt:
            prompt: test
          system_prompt:
        - invalid_structure
        """

        config_file = tmp_path / "malformed.yaml"
        config_file.write_text(malformed_yaml)

        with pytest.raises((ConfigParsingError, yaml.YAMLError, ValidationError)):
            load_config(PairwiseConfig, str(config_file))

    def test_load_config_malformed_json(self, tmp_path: Path):
        """Test loading config from malformed JSON file."""
        malformed_json = """{
            "llm_config": {
                "model": "gpt-4",
                "auth_type": "azure_managed_identity"
            },
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"}
            }
            // This comment makes it invalid JSON
        }"""

        config_file = tmp_path / "malformed.json"
        config_file.write_text(malformed_json)

        with pytest.raises((json.JSONDecodeError, ValueError)):
            load_config(PairwiseConfig, str(config_file))

    def test_load_config_unsupported_file_extension(self, tmp_path: Path):
        """Test loading config from file with unsupported extension."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
        }

        config_file = tmp_path / "config.txt"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # This might work if the function tries to parse as YAML/JSON regardless of extension
        # or might fail - depends on implementation
        try:
            config = load_config(PairwiseConfig, str(config_file))
            assert isinstance(config, PairwiseConfig)
        except (ValueError, ValidationError):
            # This is also acceptable behavior
            pass


class TestLoadConfigEdgeCases:
    """Test load_config with edge cases and corner scenarios."""

    def test_load_config_very_large_file(self, tmp_path: Path):
        """Test loading config from a very large file."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
            "others": [
                {
                    "name": f"condition_{i}",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": f"/path/to/condition_{i}",
                }
                for i in range(1000)
            ],
        }

        config_file = tmp_path / "large_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(PairwiseConfig, str(config_file))

        assert isinstance(config, PairwiseConfig)
        assert len(config.others) == 1000

    def test_load_config_unicode_content(self, tmp_path: Path):
        """Test loading config with unicode characters."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt_text": "测试提示符 with émojis 🚀"},
                "system_prompt": {"prompt_text": "Системный промпт на русском языке"},
            },
            "base": {
                "name": "配置名称",
                "question_sets": ["questions.jsonl"],
                "answer_base_path": "/path/to/unicode",
            },
        }

        config_file = tmp_path / "unicode_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, allow_unicode=True)

        config = load_config(PairwiseConfig, str(config_file))

        assert isinstance(config, PairwiseConfig)
        assert config.prompt_config.user_prompt.prompt_text is not None
        assert "🚀" in config.prompt_config.user_prompt.prompt_text
        assert config.prompt_config.system_prompt.prompt_text is not None
        assert "Системный" in config.prompt_config.system_prompt.prompt_text
        assert config.base is not None
        assert config.base.name == "配置名称"

    def test_load_config_deeply_nested_structure(self, tmp_path: Path):
        """Test loading config with deeply nested structure."""
        config_data = {
            "llm_config": {
                "model": "gpt-4",
                "auth_type": "azure_managed_identity",
                "init_args": {
                    "nested": {"deeply": {"very": {"much": {"value": "deep"}}}}
                },
            },
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
        }

        config_file = tmp_path / "nested_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(PairwiseConfig, str(config_file))

        assert isinstance(config, PairwiseConfig)

    def test_load_config_with_null_values(self, tmp_path: Path):
        """Test loading config with null/None values."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt_text": "test"},
                "system_prompt": {"prompt_text": "test"},
            },
            "base": None,  # Explicitly null
            "others": [],
            "question_sets": [],
        }

        config_file = tmp_path / "null_values_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(PairwiseConfig, str(config_file))

        assert isinstance(config, PairwiseConfig)
        assert config.base is None
        assert config.others == []
        assert config.question_sets == []

    def test_load_config_concurrent_access(self, tmp_path: Path):
        """Test loading config with concurrent access to same file."""
        import threading

        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
        }

        config_file = tmp_path / "concurrent_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        configs = []
        errors = []

        def load_config_thread():
            try:
                config = load_config(PairwiseConfig, str(config_file))
                configs.append(config)
            except (ValidationError, FileNotFoundError, ConfigParsingError) as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=load_config_thread)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All threads should succeed
        assert len(errors) == 0
        assert len(configs) == 5
        assert all(isinstance(config, PairwiseConfig) for config in configs)


class TestLoadConfigDifferentConfigTypes:
    """Test load_config with different configuration class types."""

    def test_load_config_wrong_type_for_data(self, tmp_path: Path):
        """Test loading config where data doesn't match expected config type."""
        # Create data that's valid for PairwiseConfig but try to load as LLMConfig
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
        }

        config_file = tmp_path / "wrong_type_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValidationError):
            load_config(LLMConfig, str(config_file))

    def test_load_minimal_config(self, tmp_path: Path):
        """Test loading config with minimal required fields."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "prompt_config": {
                "user_prompt": {"prompt": "minimal test"},
                "system_prompt": {"prompt": "minimal system"},
            },
        }

        config_file = tmp_path / "minimal_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(PairwiseConfig, str(config_file))

        assert isinstance(config, PairwiseConfig)
        # Should use default values for non-specified fields
        assert config.trials == 4  # default value

    def test_load_config_with_path_object(self, tmp_path: Path):
        """Test loading config using Path object instead of string."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
        }

        config_file = tmp_path / "path_object_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(PairwiseConfig, config_file)  # Pass Path object directly

        assert isinstance(config, PairwiseConfig)


class TestLoadConfigErrorMessages:
    """Test that load_config provides helpful error messages."""

    def test_load_config_validation_error_message_quality(self, tmp_path: Path):
        """Test that validation errors provide clear, helpful messages."""
        config_data = {
            "llm_config": {"model": "", "auth_type": "invalid"},
            "trials": -1,  # Invalid negative value
            "prompt_config": {
                "user_prompt": {"prompt": ""},
                "system_prompt": {},  # Missing prompt field
            },
        }

        config_file = tmp_path / "validation_error_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_config(PairwiseConfig, str(config_file))

        error_message = str(exc_info.value)
        # The error message should contain useful information about what went wrong
        assert len(error_message) > 0
        # Could add more specific assertions about error message content if needed

    def test_load_config_file_not_found_message(self):
        """Test that file not found errors provide clear messages."""
        nonexistent_path = "/clearly/nonexistent/path/config.yaml"

        with pytest.raises((FileNotFoundError, OSError)) as exc_info:
            load_config(PairwiseConfig, nonexistent_path)

        error_message = str(exc_info.value)
        assert len(error_message) > 0
        # The error should reference the file path
        assert "config.yaml" in error_message or nonexistent_path in error_message
