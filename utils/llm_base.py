from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from openai import OpenAI

from utils.cache_manager import CacheManager


class LLMBase:
    SUPPORTED_PROVIDERS = ("openai", "openrouter", "google")

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        *,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        prov = (provider or "").lower()
        if prov not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider '{provider}'. Supported providers: {self.SUPPORTED_PROVIDERS}")
        self.provider = prov
        self.model = model
        self.temperature = temperature
        self.max_retries = 10
        self.backoff_base = 0.5  # seconds
        self._cache_spec: Optional[Dict[str, Any]] = None
        self._uses_openai_client = prov in {"openai", "openrouter"}

        if prov in {"openai", "openrouter"}:
            base = base_url
            headers = dict(default_headers or {})
            if prov == "openrouter":
                base = base or "https://openrouter.ai/api/v1"
                if "HTTP-Referer" not in headers:
                    referer = os.getenv("OPENROUTER_SITE_URL")
                    if referer:
                        headers["HTTP-Referer"] = referer
                if "X-Title" not in headers:
                    title = os.getenv("OPENROUTER_APP_NAME")
                    if title:
                        headers["X-Title"] = title
            self.client = OpenAI(api_key=api_key, base_url=base, default_headers=headers or None)
        elif prov == "google":
            self.client = _GoogleClient(api_key=api_key, model_name=model, temperature=temperature)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported provider '{provider}'.")

    def configure_cache(
        self,
        *,
        module: str,
        method: str,
        dataset_root: Optional[Union[str, Path]] = None,
        base_dir: Optional[Union[str, Path]] = None,
        variant: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        base_dirs = [base_dir] if base_dir is not None else []
        self._cache_spec = {
            "module": module,
            "method": method,
            "dataset": dataset_root,
            "base_dirs": base_dirs,
            "variant": variant,
            "model_name": model_name or self.model,
        }

    def set_cache_dir(self, cache_dir: Union[Path, str]) -> None:
        try:
            dataset_root = CacheManager.infer_dataset_root(cache_dir)
            self.configure_cache(
                module="llm_base",
                method="legacy",
                dataset_root=dataset_root,
                base_dir=cache_dir,
                variant=None,
                model_name=self.model,
            )
        except Exception as e:
            print(f"[LLMBase] Failed to configure legacy cache {cache_dir}: {e}", file=sys.stderr)
            self._cache_spec = None

    def _parse_retry_after_seconds(self, message: str) -> Optional[float]:
        # e.g., "Please try again in 562ms"
        try:
            m = re.search(r"try again in\s+(\d+)ms", message)
            if m:
                return max(0.05, int(m.group(1)) / 1000.0)
        except Exception:
            pass
        return None

    def chat_completion_with_cache(self, namespace: str, prompt: str, temperature: Optional[float] = None, components: Optional[Dict[str, str]] = None, file_tag: Optional[str] = None) -> str:
        """Single-prompt chat completion with optional persistent caching.

        - namespace: logical grouping (e.g., "qe", "judge")
        - components: additional key parts (e.g., {"field":"description","version":"v1"})
        """
        key_obj = {
            "model": self.model,
            "prompt": prompt,
            **(components or {}),
        }
        # Try cache
        tag = file_tag or namespace
        cache_spec = self._cache_spec
        if cache_spec is not None:
            cached = CacheManager.get_llm_record(
                module=cache_spec["module"],
                method=cache_spec["method"],
                model_name=cache_spec.get("model_name", self.model),
                namespace=tag,
                key=key_obj,
                dataset=cache_spec.get("dataset"),
                base_dirs=cache_spec.get("base_dirs", []),
                variant=cache_spec.get("variant"),
            )
            if cached is not None:
                rec = cached.get("record", {})
                meta = rec.get("meta", {}) if isinstance(rec.get("meta"), dict) else {}
                if meta.get("model") == self.model:
                    if components:
                        ok = all(
                            meta.get(k) == components.get(k)
                            for k in ("used_by", "method", "step")
                            if k in components
                        )
                        if ok:
                            return str(rec.get("content", ""))
                    else:
                        return str(rec.get("content", ""))

        # Invoke
        resp = self._create_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        try:
            content = resp.choices[0].message.content  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[LLMBase] Unexpected chat response format: {e}", file=sys.stderr)
            raise

        # Store
        if cache_spec is not None and self._uses_openai_client:
            try:
                meta_info = {"model": self.model}
                if components:
                    for k in ("used_by", "method", "step"):
                        if k in components:
                            meta_info[k] = components[k]
                CacheManager.append_llm_record(
                    module=cache_spec["module"],
                    method=cache_spec["method"],
                    model_name=cache_spec.get("model_name", self.model),
                    namespace=tag,
                    key=key_obj,
                    content=str(content),
                    record_meta=meta_info,
                    dataset=cache_spec.get("dataset"),
                    base_dirs=cache_spec.get("base_dirs", []),
                    variant=cache_spec.get("variant"),
                )
            except Exception as e:
                print(f"[LLMBase] Failed writing cache: {e}", file=sys.stderr)
        return str(content)

    def generate_chat_completion(self, *, messages: Sequence[Dict[str, Any]], temperature: Optional[float] = None, **kwargs):
        return self._create_completion(messages=messages, temperature=temperature, extra_kwargs=kwargs)

    def _create_completion(
        self,
        *,
        messages: Sequence[Dict[str, Any]],
        temperature: Optional[float] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if self.provider == "google":
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature if temperature is None else temperature,
                **(extra_kwargs or {}),
            )

        def _invoke():
            return self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature if temperature is None else temperature,
                messages=messages,
                **(extra_kwargs or {}),
            )

        return self.call_with_retry(_invoke)

    def call_with_retry(self, fn, *args, **kwargs):
        """Invoke OpenAI API with rate-limit aware retries."""
        # Lazy import exceptions to avoid hard dependency at import time
        try:
            import openai  # type: ignore
            RateLimitError = getattr(openai, 'RateLimitError', Exception)
            APIError = getattr(openai, 'APIError', Exception)
        except Exception:  # pragma: no cover
            RateLimitError = Exception  # type: ignore
            APIError = Exception  # type: ignore

        attempt = 0
        if not self._uses_openai_client:
            return fn(*args, **kwargs)

        while True:
            try:
                return fn(*args, **kwargs)
            except RateLimitError as e:  # tokens per minute or RPM exceeded
                print(f"[LLM] Rate limit error: {e}, current attempt: {attempt}", file=sys.stderr)
                attempt += 1
                if attempt > self.max_retries:
                    raise
                wait_s = self._parse_retry_after_seconds(str(e))
                if wait_s is None:
                    wait_s = self.backoff_base * (2 ** (attempt - 1))
                # small jitter
                wait_s += random.uniform(0, 0.25)
                time.sleep(60)
                continue
            except APIError as e:
                # If status suggests throttling (429), retry; else raise
                attempt += 1
                status = getattr(e, 'status_code', None)
                if status == 429 and attempt <= self.max_retries:
                    print(f"[LLM] API error: {e}, current attempt: {attempt}", file=sys.stderr)
                    time.sleep(60)
                    continue
                raise

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[\u0000-\u001F]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def try_parse_json(text: str) -> Optional[Any]:
        # Attempt direct parse
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try fenced code blocks ```json ... ``` or ``` ... ```
        code_blocks = re.findall(r"```(?:json)?\n([\s\S]*?)```", text, re.IGNORECASE)
        for block in code_blocks:
            try:
                return json.loads(block.strip())
            except Exception:
                continue
        # Balanced brace extraction
        start = text.find("{")
        if start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:i+1]
                            try:
                                return json.loads(candidate)
                            except Exception:
                                break
        return None


def _parse_data_url(url: str) -> Optional[Dict[str, str]]:
    if not url or not url.startswith("data:"):
        return None
    try:
        header, data = url.split(",", 1)
        if ";" in header:
            mime = header.split(";")[0].split(":", 1)[1]
        else:
            mime = header.split(":", 1)[1]
        return {"mime_type": mime, "data": data}
    except Exception:
        return None


class _GoogleChatCompletions:
    def __init__(self, api_key: str, model_name: str, temperature: float) -> None:
        self._default_temp = temperature
        self._model_name = model_name
        from google import genai  # type: ignore
        self._client = genai.Client(api_key=api_key)
        self._use_new_client = True

    def create(self, *, model: str, messages: Sequence[Dict[str, Any]], temperature: float, **_: Any) -> Any:
        system_prompts: List[str] = []
        contents: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role") or "user"
            payload = message.get("content", "")
            if role == "system" and isinstance(payload, str):
                system_prompts.append(payload)
                continue
            parts: List[Dict[str, Any]] = []
            if isinstance(payload, str):
                parts.append({"text": payload})
            elif isinstance(payload, list):
                for chunk in payload:
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("type") == "text":
                        parts.append({"text": chunk.get("text", "")})
                    elif chunk.get("type") == "image_url":
                        data_url = _parse_data_url(str(chunk.get("image_url", {}).get("url")))
                        if data_url:
                            parts.append({"inline_data": data_url})
            else:
                parts.append({"text": str(payload)})
            if not parts:
                parts.append({"text": ""})
            mapped_role = "model" if role == "assistant" else "user"
            contents.append({"role": mapped_role, "parts": parts})

        generation_config = {"temperature": float(temperature if temperature is not None else self._default_temp)}

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config={"temperature": generation_config["temperature"]})
        
        text = getattr(response, "text", None)

        if not text and getattr(response, "candidates", None):
            first = response.candidates[0]
            text_parts: List[str] = []
            for part in getattr(first.content, "parts", []):
                value = getattr(part, "text", None)
                if value:
                    text_parts.append(value)
            text = "\n".join(text_parts)
        text = text or ""
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )


class _GoogleClient:
    def __init__(self, api_key: str, model_name: str, temperature: float) -> None:
        self.chat = SimpleNamespace(completions=_GoogleChatCompletions(api_key, model_name, temperature))