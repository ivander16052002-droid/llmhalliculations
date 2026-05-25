from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


NULL_LIKE = {"", "null", "none", "nan", "нет", "не указано", "n/a"}


@dataclass
class ParsedModelResponse:
    status: str
    answer: str
    source_id: str
    source_authors: str
    source_title: str
    source_year: str
    source_container: str
    source_locator: str
    quote_id: str
    quote_text: str
    limitations: str
    parse_error: str = ""


def normalize_value(value: Any) -> str:
    """Convert model values to normalized strings."""
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    text = str(value).strip()

    if text.lower() in NULL_LIKE:
        return ""

    return text


def strip_markdown_fences(text: str) -> str:
    """Remove common markdown code fences around JSON."""
    text = text.strip()

    # ```json ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def extract_fenced_json_candidates(raw_response: str) -> list[str]:
    """
    Extract JSON blocks from markdown fences.

    Some models produce broken text first and then a valid ```json {...}``` block.
    In that case we prefer the fenced JSON block over the outer broken object.
    """
    pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    return re.findall(pattern, raw_response, flags=re.IGNORECASE | re.DOTALL)


def extract_json_candidate(raw_response: str) -> str:
    """
    Extract JSON object from raw model response.

    The model may return:
    - pure JSON
    - markdown fenced JSON
    - explanatory text before/after JSON
    - almost complete JSON with missing closing braces
    """
    text = strip_markdown_fences(raw_response)

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in raw_response")

    text = text[start:].strip()

    # If there is at least one closing brace, cut to the last one.
    # If not, keep the candidate and try to repair missing braces later.
    end = text.rfind("}")
    if end != -1:
        text = text[: end + 1].strip()

    return text


def balance_json_braces(text: str) -> str:
    """
    Add missing closing braces for common truncated JSON outputs.

    This is a light repair: it only appends closing braces if there are more
    opening braces than closing braces.
    """
    open_braces = text.count("{")
    close_braces = text.count("}")

    if open_braces > close_braces:
        text = text + ("}" * (open_braces - close_braces))

    return text


def repair_unquoted_simple_values(text: str) -> str:
    """
    Repair common LLM JSON errors like:
    "id": Q1
    "id": M1

    It only quotes simple alphanumeric identifiers after JSON keys.
    """
    return re.sub(
        r'(:\s*)([A-Za-zА-Яа-яЁё_][A-Za-zА-Яа-яЁё0-9_-]*)(\s*[,}\]])',
        r'\1"\2"\3',
        text,
    )


def repair_unescaped_quotes_in_string_values(text: str) -> str:
    """
    Repair a common LLM JSON error:
    "container": "Издательство "Языки славянской культуры""

    The function works line by line and escapes inner quotes
    inside known string fields.
    """
    known_fields = {
        "status",
        "answer",
        "id",
        "authors",
        "title",
        "year",
        "container",
        "locator",
        "text",
        "limitations",
    }

    repaired_lines = []

    for line in text.splitlines():
        match = re.match(
            r'^(\s*"([^"]+)"\s*:\s*")(.*)(")(,?\s*)$',
            line,
        )

        if not match:
            repaired_lines.append(line)
            continue

        prefix, field_name, value, closing_quote, suffix = match.groups()

        if field_name not in known_fields:
            repaired_lines.append(line)
            continue

        # Escape quotes inside the value, but do not touch already escaped quotes.
        value = re.sub(r'(?<!\\)"', r'\\"', value)

        repaired_lines.append(f"{prefix}{value}{closing_quote}{suffix}")

    return "\n".join(repaired_lines)


def cleanup_json_candidate(candidate: str) -> str:
    """Apply light, deterministic cleanup rules before JSON parsing."""
    candidate = candidate.strip()

    # Sometimes models produce backslash-newline artifacts.
    candidate = candidate.replace("\\\n", "\n")

    # Repair a common LLM typo:
    # "title": "Some value"),
    # -> "title": "Some value",
    candidate = re.sub(r'(":\s*"[^"\n]*")\s*\)(\s*[,}])', r"\1\2", candidate)

    # Remove invalid control characters except \n, \r, \t.
    candidate = "".join(
        ch for ch in candidate
        if ch == "\n" or ch == "\r" or ch == "\t" or ord(ch) >= 32
    )

    # Remove trailing commas before } or ].
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    # Quote bare values such as Q1 or M1.
    candidate = repair_unquoted_simple_values(candidate)

    # Add missing closing braces if the JSON was truncated.
    candidate = balance_json_braces(candidate)

    return candidate


def parse_json_safely(raw_response: str) -> dict[str, Any]:
    """Parse JSON from raw response with light cleanup."""

    # Attempt 0: if the model produced fenced JSON blocks,
    # try them first, from last to first.
    fenced_candidates = extract_fenced_json_candidates(raw_response)

    for fenced_candidate in reversed(fenced_candidates):
        cleaned_fenced = cleanup_json_candidate(fenced_candidate)

        try:
            return json.loads(cleaned_fenced)
        except json.JSONDecodeError:
            repaired_fenced = repair_unescaped_quotes_in_string_values(cleaned_fenced)

            try:
                return json.loads(repaired_fenced)
            except json.JSONDecodeError:
                continue

    candidate = extract_json_candidate(raw_response)

    # Attempt 1: as is.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Attempt 2: deterministic cleanup.
    cleaned = cleanup_json_candidate(candidate)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 3: repair unescaped quotes inside known string fields.
    repaired = repair_unescaped_quotes_in_string_values(cleaned)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as final_error:
        raise ValueError(f"JSON parse error: {final_error}") from final_error


def parse_malformed_json_fallback(raw_response: str) -> dict[str, Any]:
    """
    Salvage heavily malformed JSON.

    Used when the model starts a JSON object but breaks it before source/quote.
    We extract status and answer when possible; source and quote remain empty.
    """
    status_match = re.search(
        r'"status"\s*:\s*"([^"]*)"',
        raw_response,
        flags=re.DOTALL,
    )

    answer_match = re.search(
        r'"answer"\s*:\s*"(.*)',
        raw_response,
        flags=re.DOTALL,
    )

    status = status_match.group(1).strip() if status_match else "partial"
    answer = ""

    if answer_match:
        answer = answer_match.group(1)

        # If source block starts later, cut answer before it.
        answer = re.split(r'"\s*,\s*"source"\s*:', answer, maxsplit=1)[0]

        # If a fenced JSON block starts later, cut answer before it.
        answer = re.split(r"```(?:json)?", answer, maxsplit=1, flags=re.IGNORECASE)[0]

        # Clean obvious JSON tails.
        answer = answer.strip()
        answer = answer.rstrip()
        answer = answer.rstrip('",')
        answer = answer.replace('\\"', '"')

    return {
        "status": status,
        "answer": answer,
        "source": {},
        "quote": {},
        "limitations": "",
    }


def parse_model_response(raw_response: str) -> ParsedModelResponse:
    """
    Parse model JSON response into flat fields.

    Never use this to judge correctness of citation.
    It only extracts fields for later annotation.
    """
    try:
        data = parse_json_safely(raw_response)

        source = data.get("source") or {}
        quote = data.get("quote") or {}

        if not isinstance(source, dict):
            source = {}

        if not isinstance(quote, dict):
            quote = {}

        return ParsedModelResponse(
            status=normalize_value(data.get("status")),
            answer=normalize_value(data.get("answer")),
            source_id=normalize_value(source.get("id")),
            source_authors=normalize_value(source.get("authors")),
            source_title=normalize_value(source.get("title")),
            source_year=normalize_value(source.get("year")),
            source_container=normalize_value(source.get("container")),
            source_locator=normalize_value(source.get("locator")),
            quote_id=normalize_value(quote.get("id")),
            quote_text=normalize_value(quote.get("text")),
            limitations=normalize_value(data.get("limitations")),
            parse_error="",
        )

    except Exception as exc:
        try:
            data = parse_malformed_json_fallback(raw_response)

            source = data.get("source") or {}
            quote = data.get("quote") or {}

            return ParsedModelResponse(
                status=normalize_value(data.get("status")),
                answer=normalize_value(data.get("answer")),
                source_id=normalize_value(source.get("id")),
                source_authors=normalize_value(source.get("authors")),
                source_title=normalize_value(source.get("title")),
                source_year=normalize_value(source.get("year")),
                source_container=normalize_value(source.get("container")),
                source_locator=normalize_value(source.get("locator")),
                quote_id=normalize_value(quote.get("id")),
                quote_text=normalize_value(quote.get("text")),
                limitations=normalize_value(data.get("limitations")),
                parse_error="",
            )

        except Exception:
            return ParsedModelResponse(
                status="",
                answer="",
                source_id="",
                source_authors="",
                source_title="",
                source_year="",
                source_container="",
                source_locator="",
                quote_id="",
                quote_text="",
                limitations="",
                parse_error=repr(exc),
            )