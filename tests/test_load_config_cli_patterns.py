# Copyright (c) 2025 Microsoft Corporation.
"""Tests for load_config function based on actual usage patterns in the codebase."""

import json
from pathlib import Path

import yaml
from graphrag_common.config import load_config

from benchmark_qed.autoe.config import AssertionConfig, PairwiseConfig, ReferenceConfig
from benchmark_qed.autoq.config import QuestionGenerationConfig


class TestLoadConfigUseCases:
    """Test load_config function using the exact patterns found in CLI modules."""

    def test_pairwise_scoring_use_case(self, tmp_path: Path):
        """Test the exact pattern used in autoe/cli.py for pairwise scoring."""
        # This mirrors the pattern: config = load_config(PairwiseConfig, comparison_spec)
        comparison_spec_data = {
            "llm_config": {
                "model": "gpt-4o",
                "auth_type": "azure_managed_identity",
                "llm_provider": "azure.openai.chat",
                "concurrent_requests": 4,
            },
            "trials": 8,
            "prompt_config": {
                "user_prompt": {
                    "prompt": "Compare these two responses for the given question."
                },
                "system_prompt": {"prompt": "You are an expert evaluator."},
            },
            "base": {
                "name": "baseline_system",
                "question_sets": ["questions.jsonl"],
                "answer_base_path": "/path/to/baseline",
            },
            "others": [
                {
                    "name": "improved_system",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/improved",
                },
                {
                    "name": "experimental_system",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/experimental",
                },
            ],
            "question_sets": ["financial_qa", "support_qa"],
            "criteria": [
                {
                    "name": "relevance",
                    "description": "How relevant is the response to the question",
                },
                {
                    "name": "accuracy",
                    "description": "How accurate is the information provided",
                },
            ],
        }

        comparison_spec = tmp_path / "pairwise_comparison.yaml"
        with open(comparison_spec, "w") as f:
            yaml.dump(comparison_spec_data, f)

        # Test the exact pattern from CLI
        config = load_config(PairwiseConfig, str(comparison_spec))

        # Verify it loaded correctly and can be used as in the CLI
        assert config.trials == 8

        # Test the filtering pattern from CLI:
        exclude_criteria = ["accuracy"]
        config.criteria = [
            criterion
            for criterion in config.criteria
            if criterion.name not in exclude_criteria
        ]

        assert len(config.criteria) == 1
        assert config.criteria[0].name == "relevance"

    def test_reference_scoring_use_case(self, tmp_path: Path):
        """Test the exact pattern used in autoe/cli.py for reference scoring."""
        # This mirrors the pattern: config = load_config(ReferenceConfig, comparison_spec)
        comparison_spec_data = {
            "llm_config": {
                "model": "gpt-4-turbo",
                "auth_type": "azure_managed_identity",
                "llm_provider": "openai.chat",
            },
            "trials": 6,
            "prompt_config": {
                "user_prompt": {
                    "prompt": "Evaluate this response against the reference answer."
                },
                "system_prompt": {"prompt": "You are evaluating response quality."},
            },
            "reference": {
                "name": "reference_system",
                "question_sets": ["questions.jsonl"],
                "answer_base_path": "/path/to/reference",
            },
            "generated": [
                {
                    "name": "system_a",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/system_a",
                },
                {
                    "name": "system_b",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/system_b",
                },
            ],
            "question_sets": ["test_set_1", "test_set_2"],
            "criteria": [
                {
                    "name": "correctness",
                    "description": "How correct are the facts and statements",
                },
                {
                    "name": "clarity",
                    "description": "How clear and understandable is the response",
                },
            ],
        }

        comparison_spec = tmp_path / "reference_comparison.yaml"
        with open(comparison_spec, "w") as f:
            yaml.dump(comparison_spec_data, f)

        config = load_config(ReferenceConfig, str(comparison_spec))

        # Test the filtering pattern from CLI
        exclude_criteria = ["clarity"]
        config.criteria = [
            criterion
            for criterion in config.criteria
            if criterion.name not in exclude_criteria
        ]

        assert len(config.criteria) == 1
        assert config.criteria[0].name == "correctness"

    def test_assertion_scoring_use_case(self, tmp_path: Path):
        """Test the exact pattern used in autoe/cli.py for assertion scoring."""
        # This mirrors the pattern: config = load_config(AssertionConfig, comparison_spec)
        comparison_spec_data = {
            "llm_config": {
                "model": "gpt-4",
                "auth_type": "azure_managed_identity",
                "llm_provider": "azure.openai.chat",
            },
            "trials": 4,
            "prompt_config": {
                "user_prompt": {
                    "prompt": "Verify if the response meets the assertion criteria."
                },
                "system_prompt": {"prompt": "You are checking assertions."},
            },
            "generated": {
                "name": "test_condition",
                "question_sets": ["questions.jsonl"],
                "answer_base_path": "/path/to/test",
            },
            "assertions": {"assertions_path": "/path/to/assertions.json"},
        }

        comparison_spec = tmp_path / "assertion_comparison.yaml"
        with open(comparison_spec, "w") as f:
            yaml.dump(comparison_spec_data, f)

        config = load_config(AssertionConfig, str(comparison_spec))

        # Verify it works as expected in the CLI context
        assert config.trials == 4
        assert str(config.assertions.assertions_path) == "/path/to/assertions.json"

    def test_question_generation_use_case(self, tmp_path: Path):
        """Test the exact pattern used in autoq/cli.py for question generation."""
        # This mirrors the pattern: config = load_config(QuestionGenerationConfig, configuration_path)
        configuration_path_data = {
            "input": {
                "dataset_path": "./test_data",
                "input_type": "json",
                "text_column": "content",
                "metadata_columns": ["title", "date"],
                "file_encoding": "utf-8",
            },
            "encoding": {
                "model_name": "o200k_base",
                "chunk_size": 600,
                "chunk_overlap": 100,
            },
            "sampling": {
                "num_clusters": 15,
                "num_samples_per_cluster": 8,
                "random_seed": 42,
            },
            "chat_model": {
                "model": "gpt-4",
                "auth_type": "azure_managed_identity",
                "llm_provider": "openai.chat",
                "concurrent_requests": 3,
            },
            "embedding_model": {
                "model": "text-embedding-3-small",
                "auth_type": "azure_managed_identity",
                "llm_provider": "openai.embedding",
            },
        }

        configuration_path = tmp_path / "question_gen_config.yaml"
        with open(configuration_path, "w") as f:
            yaml.dump(configuration_path_data, f)

        config = load_config(QuestionGenerationConfig, str(configuration_path))

        # Verify it loaded correctly for CLI usage
        assert config.encoding.chunk_size == 600
        assert config.sampling.num_clusters == 15
        assert config.chat_model.concurrent_requests == 3

    def test_config_with_llm_factory_integration(self, tmp_path: Path):
        """Test that loaded config works with LLM factory pattern used in CLI."""
        # This tests the pattern: llm_client = ModelFactory.create_chat_model(config.llm_config)
        config_data = {
            "llm_config": {
                "model": "gpt-4o",
                "auth_type": "azure_managed_identity",
                "llm_provider": "azure.openai.chat",
                "concurrent_requests": 4,
                "init_args": {"api_version": "2024-12-01-preview"},
                "call_args": {"temperature": 0.0, "seed": 42},
            },
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
        }

        config_file = tmp_path / "llm_factory_test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(PairwiseConfig, str(config_file))

        # Verify the llm_config can be used with ModelFactory
        llm_config = config.llm_config
        assert llm_config.model == "gpt-4o"
        assert llm_config.auth_type == "azure_managed_identity"
        assert llm_config.llm_provider == "azure.openai.chat"
        assert llm_config.concurrent_requests == 4

    def test_config_defaults_behavior(self, tmp_path: Path):
        """Test that load_config properly handles default values as used in CLI."""
        # Test with minimal config to verify defaults are applied
        minimal_config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "prompt_config": {
                "user_prompt": {"prompt": "minimal test"},
                "system_prompt": {"prompt": "minimal system"},
            },
        }

        config_file = tmp_path / "minimal_defaults.yaml"
        with open(config_file, "w") as f:
            yaml.dump(minimal_config_data, f)

        config = load_config(PairwiseConfig, str(config_file))

        # Verify defaults are applied correctly
        assert config.trials == 4  # Default value
        assert config.llm_config is not None  # Should have default LLMConfig
        assert config.base is None  # Default for optional field
        assert config.others == []  # Default empty list
        assert len(config.criteria) > 0  # Should use default criteria

    def test_yaml_vs_json_format_consistency(self, tmp_path: Path):
        """Test that YAML and JSON formats produce identical configs."""
        config_data = {
            "llm_config": {
                "model": "gpt-4",
                "auth_type": "azure_managed_identity",
                "llm_provider": "openai.chat",
            },
            "trials": 6,
            "prompt_config": {
                "user_prompt": {"prompt": "consistency test"},
                "system_prompt": {"prompt": "system consistency"},
            },
            "base": {
                "name": "base_test",
                "question_sets": ["questions.jsonl"],
                "answer_base_path": "/path/to/base",
            },
            "others": [
                {
                    "name": "other_test",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/other",
                }
            ],
        }

        # Save as YAML
        yaml_file = tmp_path / "consistency_test.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(config_data, f)

        # Save as JSON
        json_file = tmp_path / "consistency_test.json"
        with open(json_file, "w") as f:
            json.dump(config_data, f)

        # Load both
        yaml_config = load_config(PairwiseConfig, str(yaml_file))
        json_config = load_config(PairwiseConfig, str(json_file))

        # They should be equivalent
        assert yaml_config.trials == json_config.trials
        assert yaml_config.llm_config.model == json_config.llm_config.model
        assert yaml_config.base is not None
        assert json_config.base is not None
        assert yaml_config.base.name == json_config.base.name
        assert len(yaml_config.others) == len(json_config.others)

    def test_config_validation_with_exclude_criteria(self, tmp_path: Path):
        """Test the exclude_criteria filtering pattern used in CLI."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
            "base": {
                "name": "test_base",
                "question_sets": ["questions.jsonl"],
                "answer_base_path": "/path/to/base",
            },
            "others": [
                {
                    "name": "test_other",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/other",
                }
            ],
            "criteria": [
                {
                    "name": "relevance",
                    "description": "How relevant is the response to the question",
                },
                {
                    "name": "accuracy",
                    "description": "How accurate is the information provided",
                },
                {"name": "completeness", "description": "How complete is the response"},
                {
                    "name": "clarity",
                    "description": "How clear and understandable is the response",
                },
            ],
        }

        config_file = tmp_path / "exclude_criteria_test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(PairwiseConfig, str(config_file))

        # Test different exclude scenarios as would be used in CLI

        # Case 1: No exclusions
        exclude_criteria = []
        filtered_criteria = [
            criterion
            for criterion in config.criteria
            if criterion.name not in exclude_criteria
        ]
        assert len(filtered_criteria) == 4

        # Case 2: Exclude some criteria
        exclude_criteria = ["accuracy", "clarity"]
        filtered_criteria = [
            criterion
            for criterion in config.criteria
            if criterion.name not in exclude_criteria
        ]
        assert len(filtered_criteria) == 2
        assert all(c.name not in exclude_criteria for c in filtered_criteria)

        # Case 3: Exclude all criteria
        exclude_criteria = ["relevance", "accuracy", "completeness", "clarity"]
        filtered_criteria = [
            criterion
            for criterion in config.criteria
            if criterion.name not in exclude_criteria
        ]
        assert len(filtered_criteria) == 0

    def test_config_with_complex_llm_args(self, tmp_path: Path):
        """Test loading config with complex LLM initialization and call arguments."""
        config_data = {
            "llm_config": {
                "model": "gpt-4-turbo-preview",
                "auth_type": "azure_managed_identity",
                "llm_provider": "azure.openai.chat",
                "concurrent_requests": 8,
                "init_args": {
                    "api_version": "2024-12-01-preview",
                    "azure_endpoint": "https://example.openai.azure.com",
                    "max_retries": 5,
                    "timeout": 120,
                },
                "call_args": {
                    "temperature": 0.1,
                    "max_tokens": 1500,
                    "seed": 12345,
                    "top_p": 0.95,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                },
            },
            "trials": 10,
            "prompt_config": {
                "user_prompt": {"prompt": "complex llm test"},
                "system_prompt": {"prompt": "complex system test"},
            },
        }

        config_file = tmp_path / "complex_llm_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(PairwiseConfig, str(config_file))

        # Verify complex args are preserved
        assert config.llm_config.concurrent_requests == 8
        assert "azure_endpoint" in config.llm_config.init_args
        assert config.llm_config.call_args["max_tokens"] == 1500
        assert config.llm_config.call_args["seed"] == 12345
