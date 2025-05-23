"""Text Utilities for LLM."""

import json
import logging
import re
from typing import Any

import tiktoken
from json_repair import repair_json

import benchmark_qed.config.defaults as defs

log: logging.Logger = logging.getLogger(__name__)


def num_tokens(text: str, token_encoder: tiktoken.Encoding | None = None) -> int:
    """Return the number of tokens in the given text."""
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding(defs.ENCODING_MODEL)
    return len(token_encoder.encode(text))  # type: ignore


def try_parse_json_object(
    input: str, verbose: bool = True
) -> tuple[str, dict[str, Any]]:
    """JSON cleaning and formatting utilities."""
    # Sometimes, the LLM returns a json string with some extra description, this function will clean it up.

    result = None
    try:
        # Try parse first
        result = json.loads(input)
    except json.JSONDecodeError:
        if verbose:
            log.info("Warning: Error decoding faulty json, attempting repair")

    if result:
        return input, result

    pattern = r"\{(.*)\}"
    match = re.search(pattern, input, re.DOTALL)
    input = "{" + match.group(1) + "}" if match else input

    # Clean up json string.
    input = (
        input.replace("{{", "{")
        .replace("}}", "}")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("\r", "")
        .replace("\\", " ")
        .strip()
    )

    # Remove JSON Markdown Frame
    input = input.removeprefix("```json")
    if input.endswith("```"):
        input = input[: len(input) - len("```")]

    try:
        result = json.loads(input)
    except json.JSONDecodeError:
        # Fixup potentially malformed json string using json_repair.
        input = str(repair_json(json_str=input, return_objects=False))

        # Generate JSON-string output using best-attempt prompting & parsing techniques.
        try:
            result = json.loads(input)
        except json.JSONDecodeError:
            if verbose:
                log.exception("error loading json, json=%s", input)
            return input, {}
        else:
            if not isinstance(result, dict):
                if verbose:
                    log.exception("not expected dict type. type=%s:", type(result))
                return input, {}
            return input, result
    else:
        return input, result
