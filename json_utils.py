"""Robust JSON extraction from LLM responses."""
import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json_response(text: Any) -> Any:
    """Parse JSON from an LLM reply.

    Handles markdown code fences, leading/trailing prose ("Here is the JSON: ...")
    and trailing commentary. Raises ValueError if no valid JSON is found.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()

    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue

    raise ValueError(f"No valid JSON found in model response: {text[:200]!r}")
