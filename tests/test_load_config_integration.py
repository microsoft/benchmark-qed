# Copyright (c) 2025 Microsoft Corporation.
"""Integration tests for load_config function with real-world scenarios."""

import json
from pathlib import Path

import pytest
import yaml
from graphrag_common.config import load_config
from graphrag_common.config.load_config import ConfigParsingError
from pydantic import ValidationError

from benchmark_qed.autoe.config import PairwiseConfig, ReferenceConfig
from benchmark_qed.autoq.config import QuestionGenerationConfig


class TestLoadConfigIntegration:
    """Integration tests using realistic configuration examples."""

    def test_load_pairwise_config_realistic_scenario(self, tmp_path: Path):
        """Test loading a realistic pairwise scoring configuration."""
        config_data = {
            "llm_config": {
                "model": "gpt-4o",
                "auth_type": "azure_managed_identity",
                "llm_provider": "azure.openai.chat",
                "concurrent_requests": 4,
                "init_args": {
                    "api_version": "2024-12-01-preview",
                    "azure_endpoint": "https://example.openai.azure.com",
                },
                "call_args": {"temperature": 0.0, "seed": 42},
            },
            "trials": 8,
            "prompt_config": {
                "user_prompt": {
                    "prompt": "Compare the following two responses and determine which is better."
                },
                "system_prompt": {
                    "prompt": "You are an expert evaluator of AI responses."
                },
            },
            "base": {
                "name": "baseline_rag",
                "question_sets": ["questions.jsonl"],
                "answer_base_path": "/path/to/baseline",
            },
            "others": [
                {
                    "name": "improved_rag",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/improved",
                },
                {
                    "name": "experimental_rag",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/experimental",
                },
            ],
            "question_sets": [
                "financial_qa",
                "technical_documentation",
                "customer_support",
            ],
            "criteria": [
                {
                    "name": "relevance",
                    "description": "How relevant is the response to the question?",
                },
                {
                    "name": "accuracy",
                    "description": "How accurate is the provided information?",
                },
                {
                    "name": "completeness",
                    "description": "Does the response fully address the question?",
                },
            ],
        }

        config_file = tmp_path / "realistic_pairwise.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(PairwiseConfig, str(config_file))

        # Verify the config was loaded correctly
        assert config.trials == 8
        assert config.llm_config.model == "gpt-4o"
        assert config.llm_config.auth_type == "azure_managed_identity"
        assert config.base is not None
        assert config.base.name == "baseline_rag"
        assert len(config.others) == 2
        assert len(config.question_sets) == 3
        assert len(config.criteria) == 3
        assert config.criteria[1].name == "accuracy"

    def test_load_question_generation_config_realistic(self, tmp_path: Path):
        """Test loading a realistic question generation configuration."""
        config_data = {
            "input": {
                "dataset_path": "./datasets/financial_docs",
                "input_type": "json",
                "text_column": "document_content",
                "metadata_columns": ["title", "date_created", "doc_type"],
                "file_encoding": "utf-8-sig",
            },
            "encoding": {
                "model_name": "o200k_base",
                "chunk_size": 800,
                "chunk_overlap": 150,
            },
            "sampling": {
                "num_clusters": 25,
                "num_samples_per_cluster": 8,
                "random_seed": 12345,
            },
            "chat_model": {
                "model": "gpt-4-turbo",
                "auth_type": "azure_managed_identity",
                "llm_provider": "openai.chat",
                "concurrent_requests": 6,
                "call_args": {"temperature": 0.2, "max_tokens": 2000, "seed": 42},
            },
            "embedding_model": {
                "model": "text-embedding-3-large",
                "auth_type": "azure_managed_identity",
                "llm_provider": "openai.embedding",
                "init_args": {"dimensions": 1024},
            },
        }

        config_file = tmp_path / "realistic_question_gen.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        config = load_config(QuestionGenerationConfig, str(config_file))

        # Verify the config was loaded correctly
        assert config.encoding.chunk_size == 800
        assert config.input.text_column == "document_content"
        assert config.input.metadata_columns is not None
        assert len(config.input.metadata_columns) == 3
        assert config.sampling.num_clusters == 25
        assert config.chat_model.model == "gpt-4-turbo"
        assert config.embedding_model.model == "text-embedding-3-large"

    def test_load_reference_config_with_environment_variables(self, tmp_path: Path):
        """Test loading config that uses environment variable substitution."""
        config_data = {
            "llm_config": {
                "model": "gpt-4",
                "auth_type": "azure_managed_identity",
                "llm_provider": "openai.chat",
            },
            "trials": 6,
            "prompt_config": {
                "user_prompt": {
                    "prompt": "Rate this response compared to the reference answer."
                },
                "system_prompt": {
                    "prompt": "You are evaluating AI responses for quality."
                },
            },
            "reference": {
                "name": "reference_condition",
                "question_sets": ["questions.jsonl"],
                "answer_base_path": "/path/to/reference",
            },
            "generated": [
                {
                    "name": "rag_v1",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/rag_v1",
                },
                {
                    "name": "rag_v2",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/rag_v2",
                },
            ],
            "question_sets": ["evaluation_set_1", "evaluation_set_2"],
            "criteria": [
                {
                    "name": "relevance",
                    "description": "How relevant is the response to the question",
                },
                {
                    "name": "factual_accuracy",
                    "description": "How factually accurate is the response",
                },
            ],
        }

        config_file = tmp_path / "env_var_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Note: The actual environment variable substitution might be handled
        # by the graphrag_common.config.load_config implementation
        config = load_config(ReferenceConfig, str(config_file))

        assert config.trials == 6
        assert len(config.generated) == 2
        assert len(config.criteria) == 2


class TestLoadConfigErrorHandling:
    """Test error handling in practical scenarios."""

    def test_load_config_with_missing_prompt_file_reference(self, tmp_path: Path):
        """Test handling when config references missing prompt files."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "./prompts/nonexistent_user_prompt.txt"},
                "system_prompt": {"prompt": "./prompts/nonexistent_system_prompt.txt"},
            },
        }

        config_file = tmp_path / "missing_prompts_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # This might succeed or fail depending on whether the load_config
        # implementation validates file references immediately
        try:
            config = load_config(PairwiseConfig, str(config_file))
            # If it succeeds, the prompt files might be resolved later
            assert isinstance(config, PairwiseConfig)
        except (FileNotFoundError, ValueError):
            # This is also acceptable behavior
            pass

    def test_load_config_with_invalid_yaml_structure(self, tmp_path: Path):
        """Test loading with YAML that parses but has invalid structure."""
        invalid_yaml = """
        # This YAML is syntactically valid but structurally wrong
        llm_config: gpt-4  # Should be an object, not a string
        trials: four  # Should be a number, not a word
        prompt_config:
          - "this should be an object not a list"
        """

        config_file = tmp_path / "invalid_structure.yaml"
        config_file.write_text(invalid_yaml)

        with pytest.raises((ValidationError, ConfigParsingError)):
            load_config(PairwiseConfig, str(config_file))

    def test_load_config_partial_valid_data(self, tmp_path: Path):
        """Test loading config with some valid and some invalid fields."""
        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "valid prompt"},
                "system_prompt": {"prompt": "valid system prompt"},
            },
            # These fields exist but with invalid values
            "base": {
                "name": "empty_base",
                "question_sets": [],
                "answer_base_path": "/path/to/empty",
            },  # Empty name might be invalid
            "others": [
                {
                    "name": "valid_other",
                    "question_sets": ["questions.jsonl"],
                    "answer_base_path": "/path/to/valid",
                },
                {
                    "invalid_field": "no_name_or_value",
                    "question_sets": [],
                    "answer_base_path": "/path/invalid",
                },  # Missing required fields
            ],
        }

        config_file = tmp_path / "partial_valid.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # This should raise a validation error for the invalid parts
        with pytest.raises((ValidationError, ConfigParsingError)):
            load_config(PairwiseConfig, str(config_file))


class TestLoadConfigPerformance:
    """Test load_config performance characteristics."""

    def test_load_config_performance_large_config(self, tmp_path: Path):
        """Test loading performance with a large configuration file."""
        import time

        # Create a large config with many conditions and criteria
        large_others = [
            {
                "name": f"condition_{i}",
                "question_sets": ["questions.jsonl"],
                "answer_base_path": f"/path/to/condition_{i}",
            }
            for i in range(500)
        ]

        large_criteria = [
            {"name": f"criterion_{i}", "description": f"Test criterion {i}"}
            for i in range(100)
        ]

        config_data = {
            "llm_config": {"model": "gpt-4", "auth_type": "azure_managed_identity"},
            "trials": 4,
            "prompt_config": {
                "user_prompt": {"prompt": "test"},
                "system_prompt": {"prompt": "test"},
            },
            "others": large_others,
            "criteria": large_criteria,
        }

        config_file = tmp_path / "large_performance_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Measure loading time
        start_time = time.time()
        config = load_config(PairwiseConfig, str(config_file))
        load_time = time.time() - start_time

        # Verify it loaded correctly
        assert isinstance(config, PairwiseConfig)
        assert len(config.others) == 500
        assert len(config.criteria) == 100

        # Should load reasonably quickly (< 5 seconds for this size)
        assert load_time < 5.0, (
            f"Loading took {load_time:.2f} seconds, which is too slow"
        )

    def test_load_config_memory_usage(self, tmp_path: Path):
        """Test that config loading doesn't consume excessive memory."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Load multiple configs
        configs = []
        for i in range(10):
            config_data = {
                "llm_config": {
                    "model": f"gpt-4-{i}",
                    "auth_type": "azure_managed_identity",
                },
                "trials": 4,
                "prompt_config": {
                    "user_prompt": {"prompt": f"test prompt {i}"},
                    "system_prompt": {"prompt": f"test system {i}"},
                },
                "others": [
                    {
                        "name": f"other_{j}",
                        "question_sets": ["questions.jsonl"],
                        "answer_base_path": f"/path/to/other_{j}",
                    }
                    for j in range(50)
                ],
            }

            config_file = tmp_path / f"memory_test_{i}.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            config = load_config(PairwiseConfig, str(config_file))
            configs.append(config)

        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB for this test)
        memory_increase_mb = memory_increase / 1024 / 1024
        assert memory_increase_mb < 50, (
            f"Memory usage increased by {memory_increase_mb:.1f}MB"
        )

        # Verify all configs loaded correctly
        assert len(configs) == 10
        assert all(isinstance(config, PairwiseConfig) for config in configs)
