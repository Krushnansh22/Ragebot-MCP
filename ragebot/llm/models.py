"""
LLM Model Registries
─────────────────────
Available models for each provider, used by `rage auth login` to offer
an interactive model-selection menu.
"""
from __future__ import annotations

# ── Groq Models ──────────────────────────────────────────────────────────────
GROQ_MODELS: list[dict[str, str]] = [
    # ── OpenAI (GPT-OSS) ─────────────────────────────────────────────────────
    {
        "id":          "openai/gpt-oss-120b",
        "name":        "GPT-OSS 120B",
        "description": "OpenAI's largest open-source model — best quality (default)",
    },
    {
        "id":          "openai/gpt-oss-20b",
        "name":        "GPT-OSS 20B",
        "description": "OpenAI's compact open-source model — fast and capable",
    },
    {
        "id":          "openai/gpt-oss-safeguard-20b",
        "name":        "GPT-OSS Safeguard 20B",
        "description": "OpenAI's safety-focused open-source model",
    },
    # ── Whisper (speech-to-text) ──────────────────────────────────────────────
    {
        "id":          "whisper-large-v3",
        "name":        "Whisper Large V3",
        "description": "OpenAI's speech-to-text model — highest accuracy",
    },
    {
        "id":          "whisper-large-v3-turbo",
        "name":        "Whisper Large V3 Turbo",
        "description": "OpenAI's speech-to-text model — faster variant",
    },
    # ── Meta LLaMA ────────────────────────────────────────────────────────────
    {
        "id":          "llama-3.3-70b-versatile",
        "name":        "LLaMA 3.3 70B Versatile",
        "description": "Meta's large model — great for complex reasoning",
    },
    {
        "id":          "llama-3.1-8b-instant",
        "name":        "LLaMA 3.1 8B Instant",
        "description": "Ultra-fast responses — ideal for quick tasks and chat",
    },
    {
        "id":          "llama-3.2-1b-preview",
        "name":        "LLaMA 3.2 1B Preview",
        "description": "Smallest and fastest — good for simple completions",
    },
    {
        "id":          "llama-3.2-3b-preview",
        "name":        "LLaMA 3.2 3B Preview",
        "description": "Compact model — balance of speed and capability",
    },
    # ── Google Gemma ──────────────────────────────────────────────────────────
    {
        "id":          "gemma2-9b-it",
        "name":        "Gemma 2 9B IT",
        "description": "Google's Gemma 2 — strong instruction following",
    },
    # ── Mistral ───────────────────────────────────────────────────────────────
    {
        "id":          "mixtral-8x7b-32768",
        "name":        "Mixtral 8x7B (32k ctx)",
        "description": "Mistral MoE — large context window, excellent coding",
    },
]

# ── Gemini Models ────────────────────────────────────────────────────────────
GEMINI_MODELS: list[dict[str, str]] = [
    {
        "id":          "gemini-2.0-flash",
        "name":        "Gemini 2.0 Flash",
        "description": "Latest and fastest — best for most tasks (recommended)",
    },
    {
        "id":          "gemini-2.0-flash-lite",
        "name":        "Gemini 2.0 Flash Lite",
        "description": "Cost-efficient — lighter variant of 2.0 Flash",
    },
    {
        "id":          "gemini-1.5-flash",
        "name":        "Gemini 1.5 Flash",
        "description": "Fast and versatile — 1M token context window",
    },
    {
        "id":          "gemini-1.5-pro",
        "name":        "Gemini 1.5 Pro",
        "description": "Highest capability — complex reasoning and long context",
    },
]

# ── Lookup helper ────────────────────────────────────────────────────────────
PROVIDER_MODELS: dict[str, list[dict[str, str]]] = {
    "groq":   GROQ_MODELS,
    "gemini": GEMINI_MODELS,
}

PROVIDER_DEFAULTS: dict[str, str] = {
    "groq":   "openai/gpt-oss-120b",
    "gemini": "gemini-2.0-flash",
}
