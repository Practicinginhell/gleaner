"""gleaner: personal information discovery with Llama models, built on graphflow."""

__version__ = "0.1.0"

# Lazy (PEP 810): `import gleaner` and `gleaner --version` don't load the app stack;
# each name is imported on first use.
lazy from gleaner.db import Database
lazy from gleaner.feed import generate_feed
lazy from gleaner.llama import LLAMA_MODELS, get_llama_model
lazy from gleaner.security import PromptGuardResult, screen_prompt_guard

__all__ = [
    "LLAMA_MODELS",
    "Database",
    "PromptGuardResult",
    "__version__",
    "generate_feed",
    "get_llama_model",
    "screen_prompt_guard",
]
