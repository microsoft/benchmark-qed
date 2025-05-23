# Copyright (c) 2025 Microsoft Corporation.
"""Main entry point for the benchmark_qed package."""

import asyncio

import dotenv
import typer

from benchmark_qed.cli.autoe_cli import app as relative_scores_app

app: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)

app.add_typer(relative_scores_app, name="autoe", help="Relative scores CLI.")

if __name__ == "__main__":
    dotenv.load_dotenv()
    loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app()
