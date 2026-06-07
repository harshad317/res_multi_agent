from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def parse_jsonish(text: str) -> Any:
    """Parse JSON that may be surrounded by Markdown fences or prose."""

    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    starts = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx >= 0]
    if not starts:
        raise ValueError("No JSON object or array found in model output.")

    start = min(starts)
    end_obj = cleaned.rfind("}")
    end_arr = cleaned.rfind("]")
    end = max(end_obj, end_arr)
    if end <= start:
        raise ValueError("Incomplete JSON found in model output.")
    return json.loads(cleaned[start : end + 1])


def clamp_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."

