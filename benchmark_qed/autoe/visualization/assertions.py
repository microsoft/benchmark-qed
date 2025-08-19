# Copyright (c) 2025 Microsoft Corporation.
"""Visualization functions for AutoE assertion-based evaluation results."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes

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


def plot_assertion_accuracy_by_rag_method(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
    title: str = "Assertion-based Accuracy by RAG Method and Question Set",
    show_values: bool = True,
    sort_by_mean: bool = True,
    save_dpi: int = 300,
) -> tuple[Figure, Axes]:
    """
    Create a grouped bar chart showing assertion-based accuracy by RAG method and question set.
    
    Args:
        results_df: DataFrame containing evaluation results with columns:
                   - 'rag_method': RAG method names
                   - 'question_set': Question set names  
                   - 'overall_accuracy': Accuracy values
        output_path: Optional path to save the visualization
        figsize: Figure size as (width, height) tuple
        title: Chart title
        show_values: Whether to show accuracy values on bars
        sort_by_mean: Whether to sort RAG methods by mean accuracy
        save_dpi: DPI for saved image
        
    Returns:
        Tuple of (matplotlib Figure, matplotlib Axes) objects
        
    Example:
        >>> fig, ax = plot_assertion_accuracy_by_rag_method(
        ...     results_df, 
        ...     output_path=Path("output/chart.png")
        ... )
    """
    # Create pivot table for visualization
    pivot_summary = results_df.pivot(
        index="rag_method", 
        columns="question_set", 
        values="overall_accuracy"
    )
    
    # Set up consistent plotting style
    setup_plot_style()
    
    # Create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Prepare data for plotting
    pivot_summary_reset = pivot_summary.reset_index()
    
    if sort_by_mean:
        # Sort by mean accuracy across question sets (lowest to highest)
        pivot_summary_reset['mean_accuracy'] = pivot_summary_reset.select_dtypes(include='number').mean(axis=1)
        pivot_summary_reset = pivot_summary_reset.sort_values('mean_accuracy', ascending=True)
    
    x = range(len(pivot_summary_reset))
    
    # Get the question sets dynamically (exclude non-data columns)
    question_set_columns = [
        col for col in pivot_summary_reset.columns 
        if col not in ['rag_method', 'mean_accuracy']
    ]
    
    # Calculate appropriate bar width
    width = calculate_bar_width(len(question_set_columns))
    
    # Get consistent colors
    colors = get_color_palette(len(question_set_columns))
    
    for i, question_set in enumerate(question_set_columns):
        values = pivot_summary_reset[question_set].fillna(0)
        bars = ax.bar(
            [pos + width * i for pos in x], 
            values, 
            width,
            label=format_question_set_name(question_set),
            color=colors[i], 
            alpha=0.8
        )
        
        # Add value labels on bars using common utility
        if show_values:
            add_value_labels(ax, bars)
    
    # Customize the chart
    ax.set_xlabel('RAG Methods')
    ax.set_ylabel('Assertion-based Accuracy')
    ax.set_title(title)
    ax.set_xticks([pos + width * (len(question_set_columns) - 1) / 2 for pos in x])
    ax.set_xticklabels(
        [format_method_name(method) for method in pivot_summary_reset['rag_method']], 
        rotation=45, 
        ha='right'
    )
    ax.legend(loc='upper left')
    setup_grid(ax)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    # Save if output path is provided using common utility
    if output_path:
        save_figure(fig, output_path, dpi=save_dpi)
    
    return fig, ax


def plot_assertion_score_distribution(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    title: str = "Assertion Score Distribution by RAG Method",
) -> tuple[Figure, Axes]:
    """
    Create a box plot showing assertion score distributions by RAG method.
    
    Args:
        results_df: DataFrame containing detailed assertion results
        output_path: Optional path to save the visualization
        figsize: Figure size as (width, height) tuple
        title: Chart title
        
    Returns:
        Tuple of (matplotlib Figure, matplotlib Axes) objects
        
    Note:
        This function is a placeholder for future implementation when
        detailed assertion scoring data becomes available.
    """
    # Set up consistent plotting style
    setup_plot_style()
    
    # Create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Placeholder implementation
    ax.text(0.5, 0.5, 'Assertion Score Distribution\n(Future Implementation)', 
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title(title)
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        save_figure(fig, output_path)
    
    return fig, ax


def prepare_assertion_summary_data(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare assertion evaluation results for visualization.
    
    Args:
        results_df: Raw evaluation results DataFrame
        
    Returns:
        Pivot table ready for visualization
    """
    return results_df.pivot(
        index="rag_method", 
        columns="question_set", 
        values="overall_accuracy"
    )


def get_available_question_sets(results_df: pd.DataFrame) -> list[str]:
    """
    Get list of available question sets from results DataFrame.
    
    Args:
        results_df: Evaluation results DataFrame
        
    Returns:
        List of question set names
    """
    return sorted(results_df['question_set'].unique())


def get_available_rag_methods(results_df: pd.DataFrame) -> list[str]:
    """
    Get list of available RAG methods from results DataFrame.
    
    Args:
        results_df: Evaluation results DataFrame
        
    Returns:
        List of RAG method names
    """
    return sorted(results_df['rag_method'].unique())
