# Copyright (c) 2025 Microsoft Corporation.
"""Utils for CLI."""

import pandas as pd
from rich import print as rich_print
from rich.table import Table


def print_df(dataframe: pd.DataFrame, title: str) -> None:
    """Print a DataFrame in a rich format."""
    df_str = dataframe.astype(str).reset_index(drop=True)
    table = Table(title=title)
    for column in df_str.columns:
        table.add_column(column)
    for row in df_str.iterrows():
        table.add_row(*[str(cell) for cell in row[1]])
    rich_print(table)
