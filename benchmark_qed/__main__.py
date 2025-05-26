# Copyright (c) 2025 Microsoft Corporation.
"""Main entry point for the benchmark_qed package."""

import asyncio

import dotenv
import typer

from benchmark_qed.autoe.cli import app as autoe_cli

app: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)

app.add_typer(autoe_cli, name="autoe", help="Relative scores CLI.")

if __name__ == "__main__":
    dotenv.load_dotenv()
    loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app()
