"""Meta model wiring (Llama, Prompt Guard, Muse) for gleaner, through graphflow's OpenAIModel adapter."""

from __future__ import annotations

import os
import re
from typing import Any

from graphflow import OpenAIModel

# Default Meta Llama models by role
LLAMA_MODELS: dict[str, str] = {
    "judge": "llama3.2:3b",
    # The writer words what code picked (summary lines); the one Llama pulled locally (user, 2026-10-30)
    "writer": "llama3.2:3b",
    "answer": "muse-spark-1.3",
    "digest": "llama3.1:8b",
    "trend": "llama3.1:8b",
    # The one Llama installed here; 8b wasn't, so every Ask fell back to a plain search (2026-11-03)
    "ask": "llama3.2:3b",
    "prompt_guard": "prompt-guard",
}

DEFAULT_BASE_URL = "http://localhost:11434/v1"
# Meta Model API (OpenAI-compatible): hosted Muse Spark; its key is MODEL_API_KEY
META_MODEL_API_URL = "https://api.meta.ai/v1"
_HOSTED_MUSE = re.compile(r"^muse-spark(?!-open)", re.IGNORECASE)


# Ask is where answer quality shows most: Muse Spark when the Meta Model API key is set
MUSE_ASK_MODEL = "muse-spark-1.3"


def ask_model_names() -> list[str | None]:
    """List the models Ask tries, in order.

    Muse Spark first when `MODEL_API_KEY` is set, then the ask role's own model (`None`
    means that role's usual resolution). `LLAMA_ASK_MODEL` pins Ask to one model.

    Returns:
        Model names to pass to `get_llama_model("ask", model_name=...)`.
    """
    if os.environ.get("MODEL_API_KEY") and not os.environ.get("LLAMA_ASK_MODEL"):
        return [MUSE_ASK_MODEL, None]
    return [None]


def is_meta_model(name: str) -> bool:
    """Tell whether a model name is one of Meta's models.

    Meta models are Llama (incl. Llama Guard), Prompt Guard, and the hackathon's Muse
    Spark and Muse Glimmer.

    Args:
        name: Model name, optionally with an `org/` prefix and a `:tag` suffix.

    Returns:
        True if the name belongs to a Meta model family.
    """
    # the model's own name (after any org/ and before a :tag) must *start* with a Meta family:
    # "deepseek-r1-distill-llama-70b" is DeepSeek's model, distilled from Llama
    model = name.lower().rsplit("/", 1)[-1].split(":", 1)[0]
    return re.match(r"(code)?llama(\d|[-_.]|$)|prompt-guard|muse-(spark|glimmer)", model) is not None


def get_llama_model(
    role: str = "judge",
    model_name: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> OpenAIModel:
    """Return an OpenAIModel for a Meta model, local (Ollama) or hosted.

    Hosted means the Meta Model API or another OpenAI-compatible endpoint.
    The model is `model_name`, else the role's `LLAMA_<ROLE>_MODEL`, else the global
    `GLEANER_MODEL` / `OLLAMA_MODEL` / `LLAMA_DEFAULT_MODEL`, else the role's default.
    Hosted Muse Spark (`muse-spark-*`, not `-open`) goes to the Meta Model API with
    `MODEL_API_KEY`; everything else to `OLLAMA_BASE_URL` / `OPENAI_BASE_URL`.

    Args:
        role: Role whose default model and `LLAMA_<ROLE>_MODEL` variable apply.
        model_name: Explicit model name; overrides the environment and role default.
        base_url: Explicit endpoint; overrides the environment and built-in URLs.
        temperature: Sampling temperature.
        **kwargs: Extra options passed through to `OpenAIModel`.

    Returns:
        The configured `OpenAIModel`.

    Raises:
        ValueError: The resolved name isn't a Meta model (hackathon rule).
    """
    env_vars = (f"LLAMA_{role.upper()}_MODEL", "GLEANER_MODEL", "OLLAMA_MODEL", "LLAMA_DEFAULT_MODEL")
    candidates = [
        ("model_name", model_name),
        *((var, os.environ.get(var)) for var in env_vars),
        ("the role default", LLAMA_MODELS.get(role, "llama3.2:3b")),
    ]
    origin, resolved_model = next((src, name) for src, name in candidates if name)
    if not is_meta_model(resolved_model):
        raise ValueError(
            f"Model '{resolved_model}' (from {origin}) is not a Meta model; gleaner only runs Meta's models: "
            "Muse Spark, Muse Glimmer, Llama, Llama Guard or Prompt Guard (e.g. 'muse-spark-1.1', 'llama3.2:3b')."
        )
    if _HOSTED_MUSE.match(resolved_model):
        # Hosted Muse Spark runs on the Meta Model API, not on the local Ollama the judge uses
        resolved_base_url = base_url or META_MODEL_API_URL
        api_key = os.environ.get("MODEL_API_KEY") or "missing-MODEL_API_KEY"
        # Its reasoning tokens count against max_tokens: at 4096 they used it all and cut the JSON off
        kwargs.setdefault("max_tokens", 8192)
        kwargs.setdefault("reasoning_effort", "low")
    else:
        resolved_base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL
        )
        api_key = os.environ.get("OPENAI_API_KEY") or "ollama"

    return OpenAIModel(
        model_name=resolved_model,
        base_url=resolved_base_url,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )
