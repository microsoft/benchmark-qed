# Copyright (c) 2025 Microsoft Corporation.
"""Prompt configuration models."""

from functools import cached_property
from pathlib import Path
from string import Template
from typing import Self

import typer
from pydantic import BaseModel, Field, model_validator


class PromptConfig(BaseModel):
    """Configuration for prompts used in scoring."""

    prompt: Path | None = Field(
        default=None,
        description="Path to the prompt.",
    )

    prompt_text: str | None = Field(
        default=None,
        description="Text of the prompt.",
    )

    @model_validator(mode="after")
    def check_prompt(self) -> Self:
        """Check if the prompt is set."""
        if self.prompt is None and self.prompt_text is None:
            msg = "Either prompt path or prompt text must be set."
            raise ValueError(msg)
        if self.prompt is not None and self.prompt_text is not None:
            msg = "Only one of prompt path or prompt text can be set."
            raise ValueError(msg)
        return self

    @cached_property
    def template(self) -> Template:
        """Get the prompt template."""
        typer.echo(f"Loading {self.prompt or 'provided text'} as prompt template.")
        if self.prompt:
            return Template(self.prompt.read_text(encoding="utf-8"))
        if self.prompt_text:
            return Template(self.prompt_text)
        msg = "Prompt is not set. Please provide a prompt path or text."
        raise ValueError(msg)
