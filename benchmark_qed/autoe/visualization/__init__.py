# Copyright (c) 2025 Microsoft Corporation.
"""AutoE visualization package for assertion-based evaluation results."""

# Import assertion visualization functions
from benchmark_qed.autoe.visualization.assertions import (
    get_available_question_sets,
    get_available_rag_methods,
    plot_assertion_accuracy_by_rag_method,
    plot_assertion_score_distribution,
    prepare_assertion_summary_data,
)

# Import utilities for advanced users
from benchmark_qed.autoe.visualization.utils import (
    add_value_labels,
    calculate_bar_width,
    format_method_name,
    format_question_set_name,
    get_color_palette,
    save_figure,
    setup_grid,
    setup_plot_style,
)

__all__ = [
    "add_value_labels",
    "calculate_bar_width",
    "format_method_name",
    "format_question_set_name",
    "get_available_question_sets",
    "get_available_rag_methods",
    "get_color_palette",
    # Assertion-based visualizations
    "plot_assertion_accuracy_by_rag_method",
    "plot_assertion_score_distribution",
    "prepare_assertion_summary_data",
    "save_figure",
    "setup_grid",
    # Utilities
    "setup_plot_style",
]
