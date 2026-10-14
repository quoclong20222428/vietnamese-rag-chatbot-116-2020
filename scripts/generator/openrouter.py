"""OpenRouter Chat Completions API wrapper for the legal RAG chatbot.

Reads ``OPENROUTER_API_KEY`` and ``OPENROUTER_MODEL`` from the environment
(or the ``.env`` file at the repository root) using the project's shared
:func:`~scripts.environment.load_dotenv` loader.

Responsibility scope
--------------------
This module handles **only** OpenRouter API communication.  Prompt
construction, retrieval orchestration, and evidence grounding live in
``scripts/rag_pipeline.py``.

Environment variables
---------------------
OPENROUTER_API_KEY
    OpenRouter API key.  Required when this provider is selected.
    Obtain one at: https://openrouter.ai/settings/keys
OPENROUTER_MODEL
    OpenRouter model identifier.  Optional; defaults to
    ``google/gemma-4-31b-it``.
    Full list: https://openrouter.ai/models

API reference
-------------
https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "google/gemma-4-31b-it"

# HTTP status codes where a single retry with exponential back-off is appropriate.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 502})

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_env() -> None:
    """Load .env into the process environment (no-op if already set)."""
    if __package__ and __package__.startswith("scripts."):
        from ..environment import load_dotenv  # noqa: PLC0415
    else:
        from environment import load_dotenv  # noqa: PLC0415

    load_dotenv(_ENV_PATH)


def _require_env(key: str) -> str:
    """Return the value of *key* or raise a clear :exc:`RuntimeError`."""
    value = os.environ.get(key, "").strip()
    if not value:
        raise RuntimeError(
            f"{key} is required. Set it in the environment or in .env"
        )
    return value


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    """Return a safe, human-readable description of an HTTP error."""
    status = exc.code
    descriptions: dict[int, str] = {
        400: "400 Bad Request - invalid request or unsupported parameter.",
        401: "401 Unauthorized - missing or invalid OPENROUTER_API_KEY.",
        402: "402 Payment Required - insufficient OpenRouter credits.",
        403: "403 Forbidden - access denied, disabled key, or spend limit reached.",
        404: "404 Not Found - unknown model or unavailable provider.",
        429: "429 Too Many Requests - rate limit exceeded; please retry later.",
        502: "502 Bad Gateway - upstream generation failure; please retry later.",
    }
    return descriptions.get(status, f"HTTP {status}: {exc.reason}")


# ---------------------------------------------------------------------------
# OpenRouterClient
# ---------------------------------------------------------------------------


class OpenRouterClient:
    """Thin wrapper around the OpenRouter Chat Completions API.

    Uses only Python's stdlib ``urllib`` — no additional dependencies.

    Parameters
    ----------
    api_key:
        OpenRouter API key.  If ``None``, read from ``OPENROUTER_API_KEY``
        (loaded from ``.env`` if absent from the environment).
    model:
        OpenRouter model identifier.  If ``None``, read from
        ``OPENROUTER_MODEL``; falls back to ``google/gemma-4-31b-it``.

    Raises
    ------
    RuntimeError
        If ``OPENROUTER_API_KEY`` is not configured.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        _load_env()
        self._api_key = api_key or _require_env("OPENROUTER_API_KEY")
        # OPENROUTER_MODEL is optional; fall back to the default model.
        self._model = (
            model
            or os.environ.get("OPENROUTER_MODEL", "").strip()
            or _DEFAULT_MODEL
        )
        LOGGER.debug("OpenRouterClient ready (model=%s)", self._model)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """OpenRouter model identifier used by this client."""
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> str:
        """Generate text from a system instruction and a user prompt.

        Constructs a Chat Completions request compatible with the
        OpenRouter API and extracts ``choices[0].message.content`` from
        the response.

        Parameters
        ----------
        system_instruction:
            The system-level instruction (e.g. the Vietnamese legal QA prompt).
        user_prompt:
            The user-visible prompt combining the question and retrieved context.
        temperature:
            Sampling temperature.  Default: ``0.2`` (deterministic, factual).
        max_output_tokens:
            Maximum tokens in the generated response.  Default: ``2048``.

        Returns
        -------
        str
            Generated text with leading/trailing whitespace stripped.

        Raises
        ------
        ValueError
            If ``system_instruction`` or ``user_prompt`` is empty.
        RuntimeError
            On API authentication failures, rate-limit errors, malformed
            responses, or empty model outputs.
        """
        if not system_instruction or not system_instruction.strip():
            raise ValueError("system_instruction must not be empty")
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt must not be empty")

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        LOGGER.debug(
            "Sending generation request to OpenRouter (model=%s)", self._model
        )

        return self._send_with_retry(body, headers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_with_retry(self, body: bytes, headers: dict[str, str]) -> str:
        """Send the request, retrying once on transient failures (429, 502)."""
        last_exc: Exception | None = None

        for attempt in range(2):  # at most one retry
            try:
                return self._send(body, headers)
            except RuntimeError as exc:
                msg = str(exc)
                is_retryable = any(
                    f"HTTP {code}" in msg or f"{code} " in msg
                    for code in _RETRYABLE_STATUS_CODES
                )
                if not is_retryable or attempt == 1:
                    raise
                last_exc = exc
                wait = 2 ** attempt  # 1s, then 2s
                LOGGER.warning(
                    "OpenRouter transient error (%s); retrying in %ds.",
                    msg[:80],
                    wait,
                )
                time.sleep(wait)

        # Should be unreachable, but satisfies type checkers.
        raise last_exc  # type: ignore[misc]

    def _send(self, body: bytes, headers: dict[str, str]) -> str:
        """Perform a single HTTP POST and return the extracted answer text."""
        request = urllib.request.Request(
            _OPENROUTER_API_URL,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"OpenRouter API request failed: {_http_error_message(exc)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"OpenRouter API request failed (network error): {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                "OpenRouter API request timed out after 60 s."
            ) from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"OpenRouter returned a non-JSON response: {exc}"
            ) from exc

        text = self._extract_text(data)
        if not text:
            raise RuntimeError(
                "OpenRouter returned an empty response. "
                "The model may have refused to answer or the response was filtered."
            )
        return text

    @staticmethod
    def _extract_text(data: dict) -> str:  # type: ignore[type-arg]
        """Extract the answer string from a Chat Completions response dict."""
        try:
            content = data["choices"][0]["message"]["content"]
            if content:
                return str(content).strip()
        except (KeyError, IndexError, TypeError):
            pass
        return ""
