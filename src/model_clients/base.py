from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    provider: str
    model_name: str
    raw_status: str | None = None
    raw_metadata: dict | None = None


class BaseModelClient:
    provider: str
    model_name: str

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
        raise NotImplementedError