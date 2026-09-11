"""
LLM Client — OpenRouter API Wrapper
=====================================
Provides a unified interface to LLM models via OpenRouter.
Uses the OpenAI SDK with custom base_url for compatibility.

Models:
    - google/gemini-2.5-flash: Main pipeline (intent classification, reply generation, escalation)
    - anthropic/claude-sonnet-4-20250514: LLM-as-judge for evaluation

Usage:
    from src.llm_client import LLMClient
    client = LLMClient()
    response = client.classify(prompt)
"""

import os
import json
import time
import logging
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model aliases
MODELS = {
    "pipeline": "google/gemini-2.5-flash",          # Fast, cheap, structured output
    "judge": "anthropic/claude-sonnet-4-20250514",    # Strong judge, different from pipeline model
    "fallback": "openai/gpt-4o-mini",                 # Fallback if primary fails
}

# Rate limiting
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, doubles on each retry


class LLMClient:
    """
    Wrapper around OpenRouter API for LLM calls.
    
    Handles:
        - Model routing (pipeline vs judge)
        - Structured JSON output parsing
        - Rate limiting and retries
        - Cost tracking
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Set it as an environment variable "
                "or pass it to LLMClient(api_key=...)"
            )
        
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.api_key,
        )
        
        # Cost tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0
    
    def call(
        self,
        prompt: str,
        model_type: str = "pipeline",
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        """
        Make a single LLM call.
        
        Args:
            prompt: User message
            model_type: "pipeline" or "judge"
            system_prompt: Optional system message
            temperature: Sampling temperature
            max_tokens: Max response tokens
            json_mode: If True, request JSON response format
            
        Returns:
            Response text string
        """
        model = MODELS.get(model_type, MODELS["pipeline"])
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        # Retry with exponential backoff
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
                
                # Track usage
                if response.usage:
                    self._total_input_tokens += response.usage.prompt_tokens or 0
                    self._total_output_tokens += response.usage.completion_tokens or 0
                self._total_calls += 1
                
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                wait = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"LLM call failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                
                if attempt < MAX_RETRIES - 1:
                    # Try fallback model on second retry
                    if attempt == 1 and model_type == "pipeline":
                        model = MODELS["fallback"]
                        kwargs["model"] = model
                        logger.info(f"Switching to fallback model: {model}")
                    
                    time.sleep(wait)
        
        raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}")
    
    def call_json(
        self,
        prompt: str,
        model_type: str = "pipeline",
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Make an LLM call and parse JSON response.
        
        Returns:
            Parsed JSON dict
        """
        response = self.call(
            prompt=prompt,
            model_type=model_type,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        
        # Try to extract JSON from response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            json_match = _extract_json(response)
            if json_match:
                return json.loads(json_match)
            raise ValueError(f"Could not parse JSON from response: {response[:200]}")
    
    def get_usage_stats(self) -> dict:
        """Return usage statistics."""
        return {
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0


def _extract_json(text: str) -> Optional[str]:
    """Try to extract a JSON object from a text string."""
    # Look for ```json ... ``` blocks
    import re
    json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_block:
        return json_block.group(1)
    
    # Look for raw JSON object
    brace_start = text.find('{')
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[brace_start:i+1]
    
    return None
