"""
LLM Client with true streaming, progress callbacks, retry, and context window management.
Supports multi-provider routing: DeepSeek and Zhipu AI via OpenAI-compatible APIs.

All LLM calls stream tokens in real-time via an optional callback, so callers (nodes)
can immediately display partial output to users for a responsive experience.
"""
import os
import math
import time
import re
from typing import Callable, Optional
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError
import openai


class LLMClient:
    """LLM client with streaming, retry logic, context window management, and progress callbacks."""

    _instance = None
    _deepseek_client: OpenAI | None = None
    _zhipu_client: OpenAI | None = None

    DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
    DEFAULT_ZHIPU_MODEL = "glm-5.1"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if LLMClient._deepseek_client is not None:
            return

        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        deepseek_base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

        zhipu_api_key = os.getenv("OPENAI_API_KEY")
        zhipu_base_url = os.getenv("OPENAI_API_BASE", "https://open.bigmodel.cn/api/paas/v4")

        if deepseek_api_key:
            LLMClient._deepseek_client = OpenAI(
                api_key=deepseek_api_key,
                base_url=deepseek_base_url,
            )
        if zhipu_api_key:
            LLMClient._zhipu_client = OpenAI(
                api_key=zhipu_api_key,
                base_url=zhipu_base_url,
            )

    def _get_client_for_model(self, model: str) -> tuple[OpenAI, str, bool]:
        """
        Route to the correct client based on model name.
        Returns (client, model_name, is_deepseek).
        """
        if model.startswith("deepseek"):
            if LLMClient._deepseek_client is None:
                raise RuntimeError(
                    "DeepSeek client not initialized. Set DEEPSEEK_API_KEY and DEEPSEEK_API_BASE env vars."
                )
            return LLMClient._deepseek_client, model, True
        else:
            if LLMClient._zhipu_client is None:
                raise RuntimeError(
                    "Zhipu client not initialized. Set OPENAI_API_KEY and OPENAI_API_BASE env vars."
                )
            return LLMClient._zhipu_client, model, False

    def _get_context_window(self, model: str) -> int:
        key = f"MODEL_CONTEXT_WINDOW_{self._model_env_suffix(model)}"
        try:
            return int(os.getenv(key, os.getenv("MODEL_CONTEXT_WINDOW", "128000")))
        except Exception:
            return 128000

    def _model_env_suffix(self, model: str) -> str:
        if model.startswith("deepseek"):
            return "DEEPSEEK"
        return "ZHIPU"

    def _estimate_tokens_from_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 2.5))

    def _estimate_tokens_from_messages(self, messages: list) -> int:
        if not messages:
            return 0
        total = 0
        for m in messages:
            total += self._estimate_tokens_from_text(m.get("content", ""))
            total += 6
        return total

    def _adjust_for_window(
        self, messages: list, requested_max_tokens: int, model: str
    ) -> tuple[list, int]:
        context_window = self._get_context_window(model)
        safety_margin = int(os.getenv("TOKEN_SAFETY_MARGIN", "512"))
        min_completion = int(os.getenv("MIN_COMPLETION_TOKENS", "256"))

        adj_messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        ]

        prompt_tokens = self._estimate_tokens_from_messages(adj_messages)
        tokens_left = context_window - prompt_tokens - safety_margin

        if tokens_left < min_completion:
            user_indices = [
                i for i, m in enumerate(adj_messages) if m.get("role") == "user"
            ]
            if user_indices:
                longest_i = max(
                    user_indices, key=lambda i: len(adj_messages[i].get("content", ""))
                )
                content = adj_messages[longest_i].get("content", "")
                needed = max(0, (min_completion + safety_margin) - (context_window - prompt_tokens))
                chars_to_remove = math.ceil(needed * 2.5)

                ctx_marker = "Context:"
                if ctx_marker in content:
                    pre, post = content.split(ctx_marker, 1)
                    if len(post) > chars_to_remove:
                        post_truncated = post[:-chars_to_remove]
                    else:
                        post_truncated = ""
                    new_content = pre + ctx_marker + post_truncated
                else:
                    if len(content) > chars_to_remove:
                        new_content = content[:-chars_to_remove]
                    else:
                        new_content = ""
                adj_messages[longest_i]["content"] = new_content

                prompt_tokens = self._estimate_tokens_from_messages(adj_messages)
                tokens_left = context_window - prompt_tokens - safety_margin

        adj_max_tokens = max(
            min_completion, min(requested_max_tokens, max(0, tokens_left))
        )
        if adj_max_tokens < min_completion:
            adj_max_tokens = min_completion

        return adj_messages, adj_max_tokens

    def _create_completion_kwargs(
        self,
        model: str,
        messages: list,
        adj_max_tokens: int,
        temperature: float,
        is_deepseek: bool,
    ) -> dict:
        kwargs = {
            "model": model,
            "max_tokens": adj_max_tokens,
            "temperature": temperature,
            "messages": messages,
            "stream": True,
        }
        if is_deepseek:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            kwargs["stop"] = ["<|im_end|>"]
        return kwargs

    def _is_retryable_error(self, error: Exception) -> bool:
        """Return whether the error is likely transient and worth retrying."""
        retryable_types = (
            openai.RateLimitError,
            APIConnectionError,
            APITimeoutError,
            TimeoutError,
            ConnectionError,
        )
        return isinstance(error, retryable_types)

    def _retry_wait_time(self, attempt: int, backoff_factor: float) -> float:
        """Compute exponential backoff delay for retryable errors."""
        return max(backoff_factor * (2 ** attempt), backoff_factor)

    def _stream_and_join(
        self,
        client: OpenAI,
        model: str,
        kwargs: dict,
        is_deepseek: bool,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Execute a streaming LLM call.
        Accumulates tokens and fires `on_chunk` for every delta so callers
        can immediately display partial output.
        """
        reasoning_content = ""
        content = ""

        response = client.chat.completions.create(**kwargs)

        for chunk in response:
            delta = chunk.choices[0].delta

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
                if on_chunk:
                    on_chunk(delta.reasoning_content)

            if hasattr(delta, "content") and delta.content:
                content += delta.content
                if on_chunk:
                    on_chunk(delta.content)

        return content

    def _generate_with_retry(
        self,
        messages: list[dict],
        adj_max_tokens: int,
        temperature: float,
        max_retries: int,
        backoff_factor: float,
        effective_model: str,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        for attempt in range(max_retries):
            try:
                client, model_name, is_deepseek = self._get_client_for_model(effective_model)
                kwargs = self._create_completion_kwargs(
                    model_name, messages, adj_max_tokens, temperature, is_deepseek
                )
                text = self._stream_and_join(
                    client, model_name, kwargs, is_deepseek, on_chunk=on_chunk
                )
                return self._strip_think_blocks(text.strip())
            except Exception as e:
                if not self._is_retryable_error(e) or attempt == max_retries - 1:
                    raise e
                time.sleep(self._retry_wait_time(attempt, backoff_factor))
        raise Exception("Max retries reached for LLM generation")

    def _strip_think_blocks(self, text: str) -> str:
        """Strip <think> blocks from model outputs."""
        if not text:
            return text
        cleaned = re.sub(r"<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"<\s*thinking\s*>[\s\S]*?<\s*/\s*thinking\s*>", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 32768,
        temperature: float = 0.5,
        max_retries: int = 10,
        backoff_factor: float = 1.0,
        model: str | None = None,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Generate a response from the LLM with streaming support.

        Args:
            system_prompt: System prompt.
            user_prompt: User message.
            max_tokens: Max output tokens.
            temperature: Sampling temperature.
            max_retries: Retry attempts for transient errors.
            backoff_factor: Exponential backoff base.
            model: Override the default model.
            on_chunk: Optional callback(chunk: str) fired for every streamed token delta.
                      Pass this to enable real-time UI feedback — each call renders one
                      incremental token so the user sees output appear live.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        effective_model = model or self.DEFAULT_ZHIPU_MODEL
        adj_messages, adj_max_tokens = self._adjust_for_window(messages, max_tokens, effective_model)
        return self._generate_with_retry(
            messages=adj_messages,
            adj_max_tokens=adj_max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            effective_model=effective_model,
            on_chunk=on_chunk,
        )

    def generate_with_messages(
        self,
        messages: list[dict],
        max_tokens: int = 32768,
        temperature: float = 0.5,
        max_retries: int = 10,
        backoff_factor: float = 1.0,
        model: str | None = None,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Generate a response from the LLM using a pre-built messages list, with streaming.

        Args:
            messages: Pre-formatted OpenAI-style message list.
            on_chunk: Optional callback(chunk: str) fired for every streamed token delta.
        """
        effective_model = model or self.DEFAULT_ZHIPU_MODEL
        adj_messages, adj_max_tokens = self._adjust_for_window(messages, max_tokens, effective_model)
        return self._generate_with_retry(
            messages=adj_messages,
            adj_max_tokens=adj_max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            effective_model=effective_model,
            on_chunk=on_chunk,
        )
