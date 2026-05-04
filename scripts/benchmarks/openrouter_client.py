from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, cast

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True, slots=True)
class OpenRouterConfig:
    api_key: str
    app_name: str | None = None
    http_referer: str | None = None
    base_url: str = OPENROUTER_URL
    timeout_seconds: float = 120.0
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> OpenRouterConfig:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key is None or not api_key:
            raise RuntimeError("OPENROUTER_API_KEY must be set to run the benchmark")
        return cls(
            api_key=api_key,
            app_name=os.environ.get("OPENROUTER_APP_NAME"),
            http_referer=os.environ.get("OPENROUTER_HTTP_REFERER"),
        )


@dataclass(frozen=True, slots=True)
class CompletionResult:
    content: str
    usage: dict[str, object] | None


class OpenRouterClient:
    def __init__(
        self,
        config: OpenRouterConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._owns_client = client is None

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()

    def complete(
        self,
        *,
        model_slug: str,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        response_format: dict[str, object] | None = None,
    ) -> CompletionResult:
        payload: dict[str, object] = {
            "model": model_slug,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "reasoning": {"enabled": False}
        }
        if response_format is not None:
            payload["response_format"] = response_format
        response = self._post_with_retry(payload)
        response.raise_for_status()
        return _parse_completion_response(response.json())

    def _post_with_retry(self, payload: dict[str, object]) -> httpx.Response:
        client = self._client
        if client is None:
            client = httpx.Client(timeout=self._config.timeout_seconds)
            self._client = client

        headers = self._headers()
        for attempt in range(self._config.max_retries + 1):
            try:
                response = client.post(
                    self._config.base_url,
                    headers=headers,
                    json=payload,
                )
            except httpx.TransportError:
                if attempt < self._config.max_retries:
                    time.sleep(self._config.retry_backoff_seconds * (attempt + 1))
                    continue
                raise
            if (
                response.status_code in RETRY_STATUS_CODES
                and attempt < self._config.max_retries
            ):
                time.sleep(self._config.retry_backoff_seconds * (attempt + 1))
                continue
            return response
        raise RuntimeError("unreachable retry state")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        if self._config.http_referer is not None:
            headers["HTTP-Referer"] = self._config.http_referer
        if self._config.app_name is not None:
            headers["X-Title"] = self._config.app_name
        return headers


def _parse_completion_response(payload: Any) -> CompletionResult:
    if not isinstance(payload, dict):
        raise ValueError("OpenRouter response must be a JSON object")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response is missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("OpenRouter choice must be an object")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter choice is missing message")

    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("OpenRouter message is missing string content")

    usage_obj = payload.get("usage")
    usage = cast(dict[str, object], usage_obj) if isinstance(usage_obj, dict) else None
    return CompletionResult(content=content, usage=usage)
