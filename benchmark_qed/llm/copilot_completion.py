# Copyright (c) 2025 Microsoft Corporation.
"""``LLMCompletion`` implementation backed by the GitHub Copilot CLI.

Requests are sent through the GitHub Copilot SDK (``github-copilot-sdk``),
which drives a Copilot CLI runtime over JSON-RPC. Each completion runs in a
fresh, tool-less Copilot session whose system prompt is fully replaced by the
caller's system message, so the model behaves like a plain chat completion.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import re
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Unpack

import json_repair
from graphrag_llm.completion.completion import LLMCompletion
from graphrag_llm.middleware import with_middleware_pipeline
from graphrag_llm.types import (
    LLMChoice,
    LLMCompletionMessage,
    LLMCompletionResponse,
    LLMCompletionUsage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine, Iterator
    from concurrent.futures import Future

    from graphrag_cache import Cache, CacheKeyCreator
    from graphrag_llm.config import ModelConfig
    from graphrag_llm.metrics import MetricsProcessor, MetricsStore
    from graphrag_llm.rate_limit import RateLimiter
    from graphrag_llm.retry import Retry
    from graphrag_llm.tokenizer import Tokenizer
    from graphrag_llm.types import (
        LLMCompletionArgs,
        LLMCompletionChunk,
        Metrics,
        ResponseFormat,
    )
    from pydantic import BaseModel

log: logging.Logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT: str = "You are a helpful assistant."
DEFAULT_TIMEOUT_SECONDS: float = 600.0

_JSON_OBJECT_INSTRUCTION = (
    "Respond with a single valid JSON object only, with no surrounding prose "
    "or Markdown code fences."
)
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


class CopilotCompletionError(RuntimeError):
    """Raised when the Copilot CLI fails to produce a completion."""


class _CopilotRuntime:
    """A Copilot SDK client running on a dedicated background event loop.

    The SDK client is bound to the event loop it was started on, while
    benchmark-qed may call the model from several loops (one per
    ``asyncio.run``) or from worker threads. Running the client on its own
    loop thread lets every caller share a single Copilot CLI process.
    """

    def __init__(self, *, cli_path: str | None, github_token: str | None) -> None:
        self._cli_path = cli_path
        self._github_token = github_token
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="copilot-cli-runtime",
            daemon=True,
        )
        self._thread.start()
        self._client: Any = None
        self._models: set[str] | None = None
        self._start_lock: asyncio.Lock | None = None

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future[Any]:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def client(self) -> Any:
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            if self._client is None:
                try:
                    from copilot import CopilotClient, RuntimeConnection
                except ImportError as e:  # pragma: no cover - dependency is required
                    msg = (
                        "The 'copilot.chat' provider requires the 'github-copilot-sdk' "
                        "package. Install it with `pip install github-copilot-sdk`."
                    )
                    raise ImportError(msg) from e

                kwargs: dict[str, Any] = {}
                if self._cli_path:
                    kwargs["connection"] = RuntimeConnection.for_stdio(
                        path=self._cli_path
                    )
                if self._github_token:
                    kwargs["github_token"] = self._github_token
                client = CopilotClient(**kwargs)
                await client.start()
                self._client = client
        return self._client

    async def available_models(self) -> set[str]:
        if self._models is None:
            client = await self.client()
            self._models = {m.id for m in await client.list_models()}
        return self._models

    def close(self) -> None:
        if self._client is not None:
            try:
                self.submit(self._client.stop()).result(timeout=15)
            except Exception:
                log.debug("Error stopping Copilot client", exc_info=True)
            self._client = None
        self._loop.call_soon_threadsafe(self._loop.stop)


_runtimes: dict[tuple[str | None, str | None], _CopilotRuntime] = {}
_runtimes_lock = threading.Lock()


def _get_runtime(cli_path: str | None, github_token: str | None) -> _CopilotRuntime:
    key = (cli_path, github_token)
    with _runtimes_lock:
        runtime = _runtimes.get(key)
        if runtime is None:
            runtime = _CopilotRuntime(cli_path=cli_path, github_token=github_token)
            _runtimes[key] = runtime
        return runtime


def _shutdown_runtimes() -> None:
    with _runtimes_lock:
        runtimes = list(_runtimes.values())
        _runtimes.clear()
    for runtime in runtimes:
        runtime.close()


# The SDK uses the loop's default executor, which refuses new work once
# interpreter shutdown begins, so stop the CLI before executors shut down.
# ``threading._register_atexit`` hooks run in reverse order, ahead of the
# executor teardown registered by ``concurrent.futures``.
try:
    threading._register_atexit(_shutdown_runtimes)  # type: ignore[attr-defined]  # ruff: ignore[private-member-access]
except AttributeError:  # pragma: no cover
    atexit.register(_shutdown_runtimes)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            else:
                msg = "The copilot.chat provider only supports text message content."
                raise ValueError(msg)
        return "\n".join(parts)
    return str(content)


def _split_messages(messages: Any) -> tuple[str | None, str]:
    """Convert chat messages into a (system prompt, user prompt) pair.

    System/developer messages become the session's system prompt. A single
    user message is sent verbatim; multi-turn conversations are rendered as a
    labelled transcript because each completion runs in a fresh session.
    """
    if isinstance(messages, str):
        return None, messages

    system_parts: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in messages:
        msg = message if isinstance(message, dict) else message.model_dump()
        role = str(msg.get("role", "user"))
        text = _content_to_text(msg.get("content"))
        if role in ("system", "developer"):
            system_parts.append(text)
        else:
            turns.append((role, text))

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    if len(turns) == 1 and turns[0][0] == "user":
        return system_prompt, turns[0][1]

    transcript = "\n\n".join(f"[{role.upper()}]\n{text}" for role, text in turns)
    prompt = (
        "The following is the conversation so far. Write the next ASSISTANT "
        f"message only.\n\n{transcript}\n\n[ASSISTANT]\n"
    )
    return system_prompt, prompt


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text)
    return match.group(1).strip() if match else text.strip()


def _parse_structured(text: str, response_format: type[BaseModel]) -> BaseModel:
    cleaned = _strip_code_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = json_repair.loads(cleaned)
    return response_format.model_validate(data)


def _deny_permission(_request: Any, _invocation: Any) -> Any:
    from copilot.generated.rpc import PermissionDecisionDeniedInteractivelyByUser

    return PermissionDecisionDeniedInteractivelyByUser(
        feedback="Tools are disabled for benchmark-qed completions."
    )


class CopilotCLICompletion(LLMCompletion):
    """Chat completion through the GitHub Copilot CLI.

    Supported ``init_args`` (from ``LLMConfig.init_args``):

    - ``cli_path``: path to a Copilot CLI executable. Defaults to the runtime
      bundled with the SDK (or ``COPILOT_CLI_PATH``).
    - ``github_token``: GitHub token to authenticate with. Defaults to the
      Copilot CLI's logged-in user / ``COPILOT_GITHUB_TOKEN`` / ``GH_TOKEN``.
    - ``timeout``: seconds to wait for each completion (default 600).
    - ``reasoning_effort``: optional reasoning effort for supported models.
    - ``validate_model``: verify the model is available before the first
      request (default ``True``). Copilot otherwise silently falls back to
      its default model.
    - ``delete_sessions``: delete each Copilot session from disk after it
      completes (default ``True``).

    Sampling parameters in ``call_args`` (e.g. ``temperature``, ``seed``) are
    not supported by the Copilot CLI and are ignored.
    """

    def __init__(
        self,
        *,
        model_id: str,
        model_config: ModelConfig,
        tokenizer: Tokenizer,
        metrics_store: MetricsStore,
        metrics_processor: MetricsProcessor | None = None,
        rate_limiter: RateLimiter | None = None,
        retrier: Retry | None = None,
        cache: Cache | None = None,
        cache_key_creator: CacheKeyCreator,
        cli_path: str | None = None,
        github_token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        reasoning_effort: str | None = None,
        validate_model: bool = True,
        delete_sessions: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Copilot CLI completion."""
        if kwargs:
            log.warning(
                "Ignoring unsupported init_args for copilot.chat: %s",
                ", ".join(sorted(kwargs)),
            )
        self._model_id = model_id
        self._model = model_config.model
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store
        self._track_metrics = metrics_processor is not None
        self._timeout = float(timeout)
        self._reasoning_effort = reasoning_effort
        self._validate_model = validate_model
        self._model_validated = False
        self._delete_sessions = delete_sessions
        self._runtime = _get_runtime(cli_path, github_token)

        self._completion, self._completion_async = with_middleware_pipeline(
            model_config=model_config,
            model_fn=self._base_completion,
            async_model_fn=self._base_completion_async,
            request_type="chat",
            cache=cache,
            cache_key_creator=cache_key_creator,
            tokenizer=tokenizer,
            metrics_processor=metrics_processor,
            rate_limiter=rate_limiter,
            retrier=retrier,
        )

    async def _ensure_model(self) -> None:
        if not self._validate_model or self._model_validated:
            return
        models = await self._runtime.available_models()
        if self._model not in models:
            msg = (
                f"Model '{self._model}' is not available through GitHub Copilot. "
                f"Available models: {', '.join(sorted(models))}"
            )
            raise ValueError(msg)
        self._model_validated = True

    async def _run(
        self,
        *,
        messages: Any,
        response_format: type[BaseModel] | None,
        json_object: bool,
    ) -> LLMCompletionResponse:
        await self._ensure_model()
        client = await self._runtime.client()

        system_prompt, prompt = _split_messages(messages)
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        if response_format is not None:
            schema = json.dumps(response_format.model_json_schema())
            system_prompt += (
                f"\n\n{_JSON_OBJECT_INSTRUCTION} The JSON object must conform "
                f"to this JSON schema:\n{schema}"
            )
        elif json_object:
            system_prompt += f"\n\n{_JSON_OBJECT_INSTRUCTION}"

        from copilot.session_events import AssistantUsageData

        session_kwargs: dict[str, Any] = {
            "model": self._model,
            "system_message": {"mode": "replace", "content": system_prompt},
            "available_tools": [],
            "on_permission_request": _deny_permission,
            "skip_custom_instructions": True,
            "enable_config_discovery": False,
            "enable_skills": False,
        }
        if self._reasoning_effort:
            session_kwargs["reasoning_effort"] = self._reasoning_effort

        usage: list[Any] = []
        session = await client.create_session(**session_kwargs)
        try:
            session.on(
                lambda event: (
                    usage.append(event.data)
                    if isinstance(event.data, AssistantUsageData) and not event.agent_id
                    else None
                )
            )
            event = await session.send_and_wait(prompt, timeout=self._timeout)
        finally:
            await session.disconnect()
            if self._delete_sessions:
                try:
                    await client.delete_session(session.session_id)
                except Exception:
                    log.debug("Failed to delete Copilot session", exc_info=True)

        if event is None or not getattr(event.data, "content", None):
            msg = "The Copilot CLI returned no assistant message."
            raise CopilotCompletionError(msg)

        content: str = event.data.content
        formatted = None
        if response_format is not None:
            formatted = _parse_structured(content, response_format)
            content = formatted.model_dump_json()
        elif json_object:
            content = _strip_code_fence(content)

        prompt_tokens = sum(u.input_tokens or 0 for u in usage)
        completion_tokens = sum(u.output_tokens or 0 for u in usage)
        model = next((u.model for u in reversed(usage) if u.model), self._model)
        return LLMCompletionResponse(
            id=f"copilot-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                LLMChoice(
                    index=0,
                    message=LLMCompletionMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=LLMCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            formatted_response=formatted,
        )

    def _make_request(self, kwargs: dict[str, Any]) -> Coroutine[Any, Any, Any]:
        kwargs.pop("metrics", None)
        if kwargs.pop("stream", False):
            msg = "The copilot.chat provider does not support streaming completions."
            raise ValueError(msg)
        return self._run(
            messages=kwargs.pop("messages"),
            response_format=kwargs.pop("response_format", None),
            json_object=bool(kwargs.pop("response_format_json_object", False)),
        )

    def _base_completion(self, **kwargs: Any) -> LLMCompletionResponse:
        return self._runtime.submit(self._make_request(kwargs)).result()

    async def _base_completion_async(self, **kwargs: Any) -> LLMCompletionResponse:
        future = self._runtime.submit(self._make_request(kwargs))
        return await asyncio.wrap_future(future)

    def _prepare(self, kwargs: dict[str, Any]) -> Metrics | None:
        if kwargs.get("stream"):
            msg = "The copilot.chat provider does not support streaming completions."
            raise ValueError(msg)
        request_metrics: Metrics | None = kwargs.pop("metrics", None) or {}
        return request_metrics if self._track_metrics else None

    def completion(
        self,
        /,
        **kwargs: Unpack[LLMCompletionArgs[ResponseFormat]],
    ) -> LLMCompletionResponse[ResponseFormat] | Iterator[LLMCompletionChunk]:
        """Sync completion method."""
        args: dict[str, Any] = dict(kwargs)
        request_metrics = self._prepare(args)
        try:
            return self._completion(metrics=request_metrics, **args)
        finally:
            if request_metrics is not None:
                self._metrics_store.update_metrics(metrics=request_metrics)

    async def completion_async(
        self,
        /,
        **kwargs: Unpack[LLMCompletionArgs[ResponseFormat]],
    ) -> LLMCompletionResponse[ResponseFormat] | AsyncIterator[LLMCompletionChunk]:
        """Async completion method."""
        args: dict[str, Any] = dict(kwargs)
        request_metrics = self._prepare(args)
        try:
            return await self._completion_async(metrics=request_metrics, **args)
        finally:
            if request_metrics is not None:
                self._metrics_store.update_metrics(metrics=request_metrics)

    @property
    def metrics_store(self) -> MetricsStore:
        """The metrics store."""
        return self._metrics_store

    @property
    def tokenizer(self) -> Tokenizer:
        """The tokenizer."""
        return self._tokenizer
