from __future__ import annotations

from pathlib import Path

from src.config import PROJECT_ROOT, load_yaml


def load_prompts(path: str | Path = "configs/prompts.yaml") -> dict[str, str]:
    prompts = load_yaml(path)

    if not prompts:
        raise ValueError("Prompts config is empty")

    for prompt_type, prompt_text in prompts.items():
        if not isinstance(prompt_text, str):
            raise ValueError(f"Prompt must be a string: {prompt_type}")

        if "{question}" not in prompt_text:
            raise ValueError(f"Prompt does not contain {{question}} placeholder: {prompt_type}")

    return prompts


def build_prompt(question: str, prompt_type: str, prompts: dict[str, str]) -> str:
    if prompt_type not in prompts:
        available = ", ".join(prompts.keys())
        raise ValueError(f"Unknown prompt_type={prompt_type}. Available: {available}")

    # Important: use replace(), not str.format(), because the prompt contains JSON braces.
    return prompts[prompt_type].replace("{question}", question)