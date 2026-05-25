from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class OllamaResponse:
    text: str
    raw_status: str
    raw_metadata: dict


class OllamaClient:
    def __init__(self, model_name: str = "llama3") -> None:
        self.model_name = model_name

    def generate(self, prompt: str) -> OllamaResponse:
        """
        Generate response using local Ollama model.

        Requires Ollama installed and model pulled locally:
        ollama pull llama3
        """
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
            )

            if result.returncode != 0:
                return OllamaResponse(
                    text="",
                    raw_status="ERROR",
                    raw_metadata={
                        "returncode": result.returncode,
                        "stderr": result.stderr,
                    },
                )

            return OllamaResponse(
                text=result.stdout.strip(),
                raw_status="FINAL",
                raw_metadata={
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                    "model_name": self.model_name,
                },
            )

        except subprocess.TimeoutExpired as exc:
            return OllamaResponse(
                text="",
                raw_status="TIMEOUT",
                raw_metadata={
                    "error": repr(exc),
                    "model_name": self.model_name,
                },
            )

        except Exception as exc:
            return OllamaResponse(
                text="",
                raw_status="ERROR",
                raw_metadata={
                    "error": repr(exc),
                    "model_name": self.model_name,
                },
            )