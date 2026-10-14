"""Google Gemini API wrapper for the legal RAG chatbot.

Reads ``GEMINI_API_KEY`` and ``GEMINI_MODEL`` from the environment (or the
``.env`` file at the repository root) using the project's shared
:func:`~scripts.environment.load_dotenv` loader.

Responsibility scope
--------------------
This module handles **only** Gemini API communication.  Prompt construction,
retrieval orchestration, and evidence grounding live in
``scripts/rag_pipeline.py``.

Environment variables
---------------------
GEMINI_API_KEY
    Google Generative AI API key.  Required.
GEMINI_MODEL
    Gemini model identifier (e.g. ``gemini-2.0-flash``).  Required.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _load_env() -> None:
    """Load .env into the process environment (no-op if already set)."""
    # Use the project's shared loader to avoid duplicating parsing logic.
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


# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------


class GeminiClient:
    """Thin wrapper around the Google Gen AI SDK for text generation.

    Parameters
    ----------
    api_key:
        Google Generative AI API key.  If ``None``, read from
        ``GEMINI_API_KEY`` (loaded from ``.env`` if absent from the environment).
    model:
        Gemini model identifier.  If ``None``, read from ``GEMINI_MODEL``.

    Raises
    ------
    RuntimeError
        If ``GEMINI_API_KEY`` or ``GEMINI_MODEL`` is not configured.
    ImportError
        If ``google-genai`` is not installed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        _load_env()
        self._api_key = api_key or _require_env("GEMINI_API_KEY")
        self._model = model or _require_env("GEMINI_MODEL")
        self._client = self._build_client()
        LOGGER.debug("GeminiClient ready (model=%s)", self._model)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Gemini model identifier used by this client."""
        return self._model

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_client(self):
        """Instantiate the google-genai client.  Fails early if not installed."""
        try:
            import google.genai as genai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed. "
                "Run:  pip install google-genai>=1.0.0"
            ) from exc
        return genai.Client(api_key=self._api_key)

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

        Parameters
        ----------
        system_instruction:
            The system-level instruction (e.g. the Vietnamese legal QA prompt).
        user_prompt:
            The user-visible prompt combining the question and retrieved context.
        temperature:
            Sampling temperature.  Lower values produce more deterministic,
            factual responses suitable for legal QA.  Default: ``0.2``.
        max_output_tokens:
            Maximum tokens in the generated response.  Default: ``2048``.

        Returns
        -------
        str
            Generated text, with leading/trailing whitespace stripped.

        Raises
        ------
        ValueError
            If ``system_instruction`` or ``user_prompt`` is empty.
        RuntimeError
            On API authentication failures, rate-limit errors, or empty
            responses from the model.
        """
        if not system_instruction or not system_instruction.strip():
            raise ValueError("system_instruction must not be empty")
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt must not be empty")

        try:
            from google.genai import types as genai_types  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai>=1.0.0"
            ) from exc

        LOGGER.debug("Sending generation request to Gemini (model=%s)", self._model)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
        except Exception as exc:
            # Surface the original exception type in the message so callers
            # can distinguish auth failures from quota errors without
            # catching overly broad exception types.
            raise RuntimeError(
                f"Gemini API request failed ({type(exc).__name__}): {exc}"
            ) from exc

        text = self._extract_text(response)
        if not text:
            raise RuntimeError(
                "Gemini returned an empty response. "
                "The model may have refused to answer or the response was filtered."
            )
        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response) -> str:
        """Extract the text string from a Gemini API response object."""
        # google-genai SDK: response.text is the primary accessor.
        try:
            text = response.text
            if text:
                return text.strip()
        except (AttributeError, ValueError):
            pass

        # Fallback: walk candidates → content → parts.
        try:
            for candidate in response.candidates or []:
                for part in (candidate.content.parts or []):
                    if hasattr(part, "text") and part.text:
                        return part.text.strip()
        except (AttributeError, TypeError):
            pass

        return ""
