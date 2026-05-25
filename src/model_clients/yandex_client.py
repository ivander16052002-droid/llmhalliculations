from __future__ import annotations

import os
from typing import Any

from yandex_ai_studio_sdk import AIStudio

try:
    from yandex_ai_studio_sdk.auth import APIKeyAuth, IAMTokenAuth, OAuthTokenAuth
except Exception:  # pragma: no cover
    APIKeyAuth = None
    IAMTokenAuth = None
    OAuthTokenAuth = None

from src.model_clients.base import BaseModelClient, GenerationResult


class YandexClient(BaseModelClient):
    provider = "yandex"

    def __init__(self, model_name: str = "yandexgpt") -> None:
        self.model_name = model_name

        folder_id = os.getenv("YC_FOLDER_ID")
        if not folder_id:
            raise RuntimeError(
                "YC_FOLDER_ID is not set. Add it to .env or environment variables."
            )

        auth = self._build_auth()

        self.sdk = AIStudio(
            folder_id=folder_id,
            auth=auth,
        )

    def _build_auth(self) -> Any:
        api_key = os.getenv("YC_API_KEY")
        iam_token = os.getenv("YC_IAM_TOKEN")
        oauth_token = os.getenv("YC_OAUTH_TOKEN")

        if api_key:
            if APIKeyAuth is not None:
                return APIKeyAuth(api_key)
            return api_key

        if iam_token:
            if IAMTokenAuth is not None:
                return IAMTokenAuth(iam_token)
            return iam_token

        if oauth_token:
            if OAuthTokenAuth is not None:
                return OAuthTokenAuth(oauth_token)
            return oauth_token

        raise RuntimeError(
            "No Yandex auth found. Set YC_API_KEY, YC_IAM_TOKEN, or YC_OAUTH_TOKEN."
        )

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.2) -> GenerationResult:
        model = self.sdk.models.completions(self.model_name)

        # SDK versions may differ, so keep this defensive.
        try:
            configured_model = model.configure(
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            configured_model = model.configure(
                temperature=temperature,
            )

        result = configured_model.run(prompt)

        alternative = result.alternatives[0]
        text = alternative.text
        status = str(getattr(alternative, "status", ""))

        metadata = {
            "usage": str(getattr(result, "usage", "")),
            "model_version": str(getattr(result, "model_version", "")),
        }

        return GenerationResult(
            text=text,
            provider=self.provider,
            model_name=self.model_name,
            raw_status=status,
            raw_metadata=metadata,
        )