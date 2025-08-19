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
    # Assertion-based visualizations
    "plot_assertion_accuracy_by_rag_method",
    "plot_assertion_score_distribution", 
    "prepare_assertion_summary_data",
    "get_available_question_sets",
    "get_available_rag_methods",
    
    # Utilities
    "setup_plot_style",
    "get_color_palette", 
    "save_figure",
    "format_method_name",
    "format_question_set_name",
    "add_value_labels",
    "setup_grid",
    "calculate_bar_width",
]
