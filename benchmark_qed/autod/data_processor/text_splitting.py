"""A module containing the 'Tokenizer', 'TextSplitter', and 'TokenTextSplitter' models."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import tiktoken

import benchmark_qed.config.defaults as defs

EncodedText = list[int]
DecodeFn = Callable[[EncodedText], str]
EncodeFn = Callable[[str], EncodedText]
LengthFn = Callable[[str], int]

log: logging.Logger = logging.getLogger(__name__)


@dataclass
class Tokenizer:
    """Tokenizer data class."""

    chunk_size: int
    """Maximum number of tokens per chunk"""
    chunk_overlap: int
    """Overlap in tokens between chunks"""
    decode: DecodeFn
    """ Function to decode a list of token ids to a string"""
    encode: EncodeFn
    """ Function to encode a string to a list of token ids"""


class TextSplitter(ABC):
    """Text splitter class definition."""

    _chunk_size: int
    _chunk_overlap: int
    _length_function: LengthFn

    def __init__(
        self,
        chunk_size: int = defs.CHUNK_SIZE,
        chunk_overlap: int = defs.CHUNK_OVERLAP,
        length_function: LengthFn = len,
    ):
        """Init method definition."""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    @abstractmethod
    def split_text(self, text: str | list[str]) -> Iterable[str]:
        """Split text method definition."""

    @abstractmethod
    def get_size(self, text: str) -> int:
        """Return the size of the text."""


class TokenTextSplitter(TextSplitter):
    """Token text splitter class definition."""

    _allowed_special: Literal["all"] | set[str]
    _disallowed_special: Literal["all"] | Collection[str]

    def __init__(
        self,
        encoding_name: str = defs.ENCODING_MODEL,
        model_name: str | None = None,
        allowed_special: Literal["all"] | set[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] = "all",
        **kwargs: Any,
    ):
        """Init method definition."""
        super().__init__(**kwargs)
        if model_name is not None:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                log.exception("Model %s not found, using %s", model_name, encoding_name)
                enc = tiktoken.get_encoding(encoding_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special or set()
        self._disallowed_special = disallowed_special

    def encode(self, text: str) -> list[int]:
        """Encode the given text into an int-vector."""
        return self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )

    def get_size(self, text: str) -> int:
        """Return the number of tokens in a string."""
        return len(self.encode(text))

    def split_text(self, text: str | list[str]) -> list[str]:
        """Split text method."""
        if isinstance(text, list):
            text = " ".join(text)
        elif text is None or type(text) is not str or text.strip() == "":
            return []

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            chunk_size=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=lambda text: self.encode(text),
        )

        return split_single_text_on_tokens(text=text, tokenizer=tokenizer)


# Adapted from - https://github.com/langchain-ai/langchain/blob/558191198f3f4f6001c651e3df583e0dfb79a9c5/libs/text-splitters/langchain_text_splitters/base.py#L329
def split_single_text_on_tokens(text: str, tokenizer: Tokenizer) -> list[str]:
    """Split a single text and return chunks using the tokenizer."""
    result = []
    input_ids = tokenizer.encode(text)

    start_idx = 0
    cur_idx = min(start_idx + tokenizer.chunk_size, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]

    while start_idx < len(input_ids):
        chunk_text = tokenizer.decode(list(chunk_ids))
        result.append(chunk_text)
        if cur_idx == len(input_ids):
            break
        start_idx += tokenizer.chunk_size - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.chunk_size, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]

    return result
