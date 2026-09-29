"""Source classification helper for retrieval evaluation."""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Legal source classification
# ---------------------------------------------------------------------------


def _is_legal_source(meta: dict[str, Any]) -> bool:
    """Return True if the chunk originates from a legal-document source.

    Classification is based on ``content_type``, which is the cleanest
    discriminator confirmed by the actual dataset:

    - ``content_type == 'legal_text'`` - legal document (core primary,
      core amendment, and reference/supporting legal documents such as
      Luật Giáo dục 2019 all carry this value).
    - ``content_type == 'qa'`` - QA/supporting text source.

    This intentionally treats ``reference/supporting`` chunks (e.g. Luật
    Giáo dục 2019) as legal sources, because they are formally-structured
    legal texts with full structural metadata (chapter, article, clause,
    point).  Only ``qa`` chunks are excluded from structural matching.

    Parameters
    ----------
    meta:
        The ``metadata`` dict from a ``RetrievalResult``.

    Returns
    -------
    bool
        ``True`` when the chunk is a legal-document source, ``False`` when
        it is a QA or other non-legal supporting text.
    """
    content_type = (meta.get("content_type") or "").strip().lower()
    return content_type == "legal_text"


# ---------------------------------------------------------------------------
