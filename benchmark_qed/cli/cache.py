# Copyright (c) 2025 Microsoft Corporation.
"""Commands for inspecting local evaluation caches."""

import json
from pathlib import Path
from typing import Annotated

import typer

from benchmark_qed.cache import inspect_cache

app: typer.Typer = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="Inspect local evaluation caches.",
)


@app.command(name="inspect")
def inspect_command(
    cache_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to a benchmark-qed SQLite cache.",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Show cache schema, namespaces, leases, and recent provenance."""
    details = inspect_cache(cache_path)
    if json_output:
        typer.echo(json.dumps(details, indent=2, sort_keys=True))
        return

    typer.echo(f"Cache: {details['path']}")
    typer.echo(f"Schema version: {details['schema_version']}")
    typer.echo(f"Active leases: {details['active_leases']}")
    typer.echo("Namespaces:")
    for namespace in details["namespaces"]:
        typer.echo(
            f"  {namespace['namespace']}: {namespace['entries']} entries, "
            f"{namespace['configurations']} configurations"
        )
    typer.echo("Recent provenance:")
    for record in details["recent_provenance"]:
        typer.echo(
            f"  {record['created_at']} {record['namespace']}: "
            f"{json.dumps(record['metadata'], sort_keys=True)}"
        )
