# Copyright (c) 2025 Microsoft Corporation.
"""Common utilities for AutoE visualizations."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.container import BarContainer
from rich import print as rich_print


def setup_plot_style() -> None:
    """Set up consistent plotting style for all AutoE visualizations."""
    plt.style.use("default")
    sns.set_palette("Set2")


def get_color_palette(n_colors: int) -> list:
    """
    Get a consistent color palette for visualizations.

    Args:
        n_colors: Number of colors needed

    Returns
    -------
        List of colors from the Set2 palette
    """
    return sns.color_palette("Set2", n_colors)


def save_figure(
    fig: plt.Figure, output_path: Path, dpi: int = 300, bbox_inches: str = "tight"
) -> None:
    """
    Save a matplotlib figure with consistent settings.

    Args:
        fig: Matplotlib figure to save
        output_path: Path where to save the figure
        dpi: Resolution for saved image
        bbox_inches: Bounding box setting for saved image
    """
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    rich_print(f"[bold green]Visualization saved to {output_path.parent}[/bold green]")


def format_method_name(method_name: str) -> str:
    """
    Format RAG method names for display.

    Args:
        method_name: Raw method name (e.g., 'vector_rag')

    Returns
    -------
        Formatted name for display (e.g., 'Vector RAG')
    """
    return method_name.replace("_", " ").title()


def format_question_set_name(question_set: str) -> str:
    """
    Format question set names for display.

    Args:
        question_set: Raw question set name (e.g., 'activity_local')

    Returns
    -------
        Formatted name for display (e.g., 'Activity Local')
    """
    return question_set.replace("_", " ").title()


def add_value_labels(
    ax: plt.Axes,
    bars: BarContainer | Iterable[Any],
    format_str: str = "{:.3f}",
    offset: int = 3,
    fontsize: int = 9,
    fontweight: str = "bold",
) -> None:
    """
    Add value labels on top of bars in a bar chart.

    Args:
        ax: Matplotlib axes object
        bars: Bar container from matplotlib bar plot
        format_str: Format string for values
        offset: Vertical offset for labels
        fontsize: Font size for labels
        fontweight: Font weight for labels
    """
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                format_str.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight=fontweight,
            )


def setup_grid(ax: plt.Axes, alpha: float = 0.3, linestyle: str = "--") -> None:
    """
    Set up consistent grid styling for plots.

    Args:
        ax: Matplotlib axes object
        alpha: Grid transparency
        linestyle: Grid line style
    """
    ax.grid(axis="y", alpha=alpha, linestyle=linestyle)


def calculate_bar_width(n_groups: int, max_width: float = 0.8) -> float:
    """
    Calculate appropriate bar width for grouped bar charts.

    Args:
        n_groups: Number of groups in the chart
        max_width: Maximum total width for all bars

    Returns
    -------
        Appropriate width for individual bars
    """
    if n_groups <= 2:
        return 0.35
    return max_width / n_groups
