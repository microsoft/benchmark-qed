# Copyright (c) 2025 Microsoft Corporation.
"""Main entry point for the benchmark_qed package."""

import asyncio

import dotenv
import typer

from benchmark_qed.autoe.cli import app as autoe_cli
from benchmark_qed.autoq.cli import app as autoq_cli
from benchmark_qed.cli.init_config import app as init_cli
from benchmark_qed.data.cli import app as data_cli

app: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)

app.add_typer(autoe_cli, name="autoe", help="Relative scores CLI.")
app.add_typer(autoq_cli, help="Question generation CLI.")
app.add_typer(init_cli, name="config", help="Configuration initialization CLI.")
app.add_typer(data_cli, name="data", help="Dataset downloader CLI.")


def main() -> None:
    """Run the CLI."""
    dotenv.load_dotenv()
    loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app()


if __name__ == "__main__":
    main()
