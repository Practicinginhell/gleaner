"""Voice control: Meta's Muse Voice Transcribe turns speech into text, a fixed grammar into actions.

The page records a short clip, sends it to `POST /api/voice`, and gets back the words and the
action they mean. Speech is transcribed by Muse Voice Transcribe on the Meta Model API (the
one-shot HTTPS endpoint, `PUSH_TO_TALK` mode). Commands are matched by a small fixed grammar,
not a model: it is instant, free and testable, and anything it doesn't match is a question for
Ask. Replies are spoken by the browser's own speech synthesis; Meta has no text-to-speech model.
"""

from __future__ import annotations

import io
import json
import re
import wave
from typing import Any

import httpx

TRANSCRIBE_URL = "https://api.meta.ai/v1/asr/transcribe"
TRANSCRIBE_MODEL = "muse-voice-transcribe-1.0"
SAMPLE_RATES = (16000, 24000)  # what Muse accepts; the page records 16 kHz
MAX_SECONDS = 60  # a spoken command or question, not a dictation
MAX_BYTES = MAX_SECONDS * max(SAMPLE_RATES) * 2 + 1024  # 16-bit mono plus header room

TABS = {
    "my questions": ("brief", "My questions"),
    "questions": ("brief", "My questions"),
    "for you": ("feed", "For you"),
    "feed": ("feed", "For you"),
    "top 5": ("digest", "Top 5"),
    "top five": ("digest", "Top 5"),
    "trending": ("trends", "Trending"),
    "trends": ("trends", "Trending"),
    "deadlines": ("opportunities", "Deadlines"),
    "ask": ("ask", "Ask"),
    "websites": ("sources", "Websites"),
    "sources": ("sources", "Websites"),
    "blocked": ("quarantine", "Blocked"),
}

_TAB = re.compile(r"(?:show|open|go to|switch to)(?: me)?(?: the| my)? (?P<tab>.+?)(?: tab| page)?")
_COLLECT = re.compile(r"(?:collect|refresh|check (?:my |all my )?websites|read (?:my |all my )?websites)(?: now)?")
_READ_DIGEST = re.compile(r"read(?: me)?(?: the)? (?:top (?:5|five)|today'?s top(?: 5| five)?)")
_NEXT_STEPS = re.compile(r"(?:what should i do next|what next|next steps)")
_ADD_QUESTION = re.compile(r"(?:add|track|follow)(?: a)? question[:,]? (?P<text>.+)")
_HELP = re.compile(r"help|what can i say")


class VoiceError(Exception):
    """A recording or transcription problem, worded for the person who spoke."""


def parse_command(text: str) -> dict[str, Any]:
    """Map what was said to an app action.

    Args:
        text: The transcript.

    Returns:
        `{"action": ...}` plus its arguments: `tab` (with `tab` and `label`), `collect`,
        `read_digest`, `next_steps`, `add_question` (with `text`), `help`, or `ask` (with `text`)
        for anything else.
    """
    said = text.strip()
    plain = re.sub(r"\s+", " ", said.lower().strip(" .!?"))
    if _COLLECT.fullmatch(plain):
        return {"action": "collect"}
    if _READ_DIGEST.fullmatch(plain):
        return {"action": "read_digest"}
    if _NEXT_STEPS.fullmatch(plain):
        return {"action": "next_steps"}
    if _HELP.fullmatch(plain):
        return {"action": "help"}
    if (m := _TAB.fullmatch(plain)) and m["tab"] in TABS:
        tab, label = TABS[m["tab"]]
        return {"action": "tab", "tab": tab, "label": label}
    if _ADD_QUESTION.fullmatch(plain):
        # keep the speaker's own casing for the question itself
        question = re.split(r"question[:,]?\s+", said, maxsplit=1, flags=re.IGNORECASE)[-1].strip(" .?")
        return {"action": "add_question", "text": question[:1].upper() + question[1:] + "?"}
    return {"action": "ask", "text": said}


def check_wav(audio: bytes) -> None:
    """Refuse a recording Muse Voice Transcribe would refuse, before paying for the request.

    Args:
        audio: The uploaded bytes.

    Raises:
        VoiceError: Not a WAV, not mono 16-bit, an unsupported rate, or too long.
    """
    try:
        with wave.open(io.BytesIO(audio)) as w:
            channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
    except (wave.Error, EOFError) as exc:
        raise VoiceError("The recording isn't a WAV file.") from exc
    if channels != 1:
        raise VoiceError("The recording must be mono.")
    if width != 2:
        raise VoiceError("The recording must be 16-bit.")
    if rate not in SAMPLE_RATES:
        raise VoiceError("The recording must be 16 kHz or 24 kHz.")
    if frames > MAX_SECONDS * rate:
        raise VoiceError(f"The recording is longer than {MAX_SECONDS} seconds.")


def transcribe(audio: bytes, api_key: str, *, client: httpx.Client | None = None) -> str:
    """Turn a WAV recording into text with Muse Voice Transcribe.

    Args:
        audio: A mono 16-bit WAV at 16 or 24 kHz.
        api_key: The Meta Model API key.
        client: HTTP client to use; tests pass one with a mock transport.

    Returns:
        The transcript, trimmed.

    Raises:
        VoiceError: Meta refused the request or couldn't be reached.
    """
    request = {"mode": "PUSH_TO_TALK", "model": TRANSCRIBE_MODEL, "audioEncoding": "WAV"}
    files = {
        # the settings travel as a JSON part, not as form fields
        "request": (None, json.dumps(request), "application/json"),
        "audio": ("speech.wav", audio, "audio/wav"),
    }
    http = client or httpx.Client(timeout=30)
    try:
        res = http.post(TRANSCRIBE_URL, headers={"Authorization": f"Bearer {api_key}"}, files=files)
    except httpx.HTTPError as exc:
        raise VoiceError("Couldn't reach Meta to transcribe the recording.") from exc
    finally:
        if client is None:
            http.close()
    if res.status_code == 401:
        raise VoiceError("Meta rejected MODEL_API_KEY.")
    if res.status_code == 429:
        raise VoiceError("Too many voice requests to Meta. Wait a moment and try again.")
    if res.status_code != 200:
        raise VoiceError(f"Muse Voice Transcribe failed (HTTP {res.status_code}).")
    try:
        return str(res.json().get("transcript") or "").strip()
    except (ValueError, AttributeError) as exc:
        raise VoiceError("Muse Voice Transcribe sent an unexpected reply. Try again.") from exc
