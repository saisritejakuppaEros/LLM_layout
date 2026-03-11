"""Thin wrappers around the local LLM and VLM APIs."""

import json
import re
import requests
from stage2.config import TEXT_API, VL_API


def call_llm(prompt: str, max_tokens: int = 2048) -> str:
    resp = requests.post(
        TEXT_API,
        json={"prompt": prompt, "max_tokens": max_tokens},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def call_vlm(image_path: str, prompt: str, max_tokens: int = 1024) -> str:
    resp = requests.post(
        VL_API,
        json={"image_url": image_path, "prompt": prompt, "max_tokens": max_tokens},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def parse_json_response(raw: str) -> dict:
    """Strip markdown fences, thinking blocks, and parse JSON from LLM/VLM output."""
    if not raw or not raw.strip():
        raise json.JSONDecodeError("Empty response", "", 0)

    cleaned = raw.strip()
    # Strip <think>...</think> blocks (Qwen thinking format)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract first complete {...} block (handles nested braces)
    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        for i, c in enumerate(cleaned[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break

    # Regex fallback: extract first {...} block (handles prose wrapping)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found in response", raw[:200], 0)
