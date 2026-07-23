"""Groq LLM integration.

Uses Lua (via prompt_builder.lua) for prompt construction, so the
prompt logic can be modified without redeployment.
"""

from groq import Groq

from app.config import settings
from app.lua_runtime import execute as lua_execute


_client: Groq | None = None

# Free tier TPM limit: 12,000. We reserve ~2,000 for the prompt overhead
# (system message, question, formatting) and 1,024 for the answer.
# That leaves ~9,000 tokens for context.
_MAX_CONTEXT_TOKENS = 9000


def _get_client() -> Groq | None:
    global _client
    if _client is None:
        if settings.groq_api_key == "your-groq-api-key-here":
            return None
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def ask(question: str, contexts: list[dict]) -> str:
    """Send question + retrieved contexts to Groq and return the answer.

    Each context dict: {"method": str, "content": str, ...}

    Prompt construction is delegated to Lua (prompt_builder.lua) so the
    system prompt, context formatting, and truncation logic can be
    modified at runtime without redeployment.
    """
    client = _get_client()
    if client is None:
        return (
            "⚠️ Groq API key not configured. "
            "Set GROQ_API_KEY in your .env file and restart."
        )

    # Build messages via Lua
    try:
        messages = lua_execute(
            "prompt_builder.lua",
            "build_messages",
            question,
            contexts,
            _MAX_CONTEXT_TOKENS,
        )
    except Exception as exc:
        # Fallback: simple inline prompt if Lua fails
        print(f"[llm] WARNING: Lua prompt_builder failed: {exc}")
        context_str = "\n\n".join(
            f"[Source: {c.get('method', 'unknown')} - {c.get('source', c.get('document', 'unknown'))}]\n{c.get('content', '')[:500]}"
            for c in contexts
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful research assistant. Answer based on the provided context.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:",
            },
        ]

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()

