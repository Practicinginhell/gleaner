"""Tests for voice control: Muse Voice Transcribe client, spoken commands, POST /api/voice."""

import io
import json
import wave

import httpx
import pytest
from fastapi.testclient import TestClient

from gleaner import voice
from gleaner.dashboard import JS
from gleaner.db import Database
from gleaner.server import create_app


def make_wav(seconds: float = 0.5, rate: int = 16000, channels: int = 1, width: int = 2) -> bytes:
    """Build a silent WAV of the given shape."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x00" * int(seconds * rate) * channels * width)
    return buf.getvalue()


@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("Collect.", {"action": "collect"}),
        ("check my websites now", {"action": "collect"}),
        ("Open deadlines", {"action": "tab", "tab": "opportunities", "label": "Deadlines"}),
        ("go to my questions", {"action": "tab", "tab": "brief", "label": "My questions"}),
        ("Show me the top five.", {"action": "tab", "tab": "digest", "label": "Top 5"}),
        ("switch to websites tab", {"action": "tab", "tab": "sources", "label": "Websites"}),
        ("Read me the top 5", {"action": "read_digest"}),
        ("Add a question: how do people evaluate agents", {"action": "add_question", "text": "How do people evaluate agents?"}),
        ("What can I say?", {"action": "help"}),
        ("What changed in LangGraph this week?", {"action": "ask", "text": "What changed in LangGraph this week?"}),
        ("open the pod bay doors", {"action": "ask", "text": "open the pod bay doors"}),
    ],
)
def test_parse_command(said, expected):
    """Fixed phrases become app actions; anything else is a question for Ask."""
    assert voice.parse_command(said) == expected


def test_check_wav_accepts_what_the_page_records():
    """16 kHz mono 16-bit is what the page sends; 24 kHz is Muse's native rate."""
    voice.check_wav(make_wav(rate=16000))
    voice.check_wav(make_wav(rate=24000))


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (b"not audio at all", "isn't a WAV"),
        (make_wav(channels=2), "mono"),
        (make_wav(width=1), "16-bit"),
        (make_wav(rate=44100), "16 kHz or 24 kHz"),
        (make_wav(seconds=voice.MAX_SECONDS + 1), "longer than"),
    ],
)
def test_check_wav_rejects_what_muse_cant_take(audio, message):
    """A recording Muse would refuse is refused before it costs a request."""
    with pytest.raises(voice.VoiceError, match=message):
        voice.check_wav(audio)


def test_transcribe_sends_the_documented_request():
    """Multipart with a JSON `request` part and the WAV as `audio`, Bearer auth."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.read()
        return httpx.Response(200, json={"sessionId": "s", "transcript": " open deadlines ", "audioDurationMs": 500})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert voice.transcribe(make_wav(), api_key="k-test", client=client) == "open deadlines"
    assert seen["url"] == "https://api.meta.ai/v1/asr/transcribe"
    assert seen["auth"] == "Bearer k-test"
    body = seen["body"]
    assert b'name="request"' in body and b'name="audio"' in body
    part = body.split(b'name="request"')[1].split(b"\r\n\r\n", 1)[1].split(b"\r\n--", 1)[0]
    assert json.loads(part) == {"mode": "PUSH_TO_TALK", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}


@pytest.mark.parametrize(("status", "message"), [(401, "rejected MODEL_API_KEY"), (429, "Too many"), (500, "HTTP 500")])
def test_transcribe_explains_failures(status, message):
    """Meta's error codes become sentences a person can act on."""
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status, json={})))
    with pytest.raises(voice.VoiceError, match=message):
        voice.transcribe(make_wav(), api_key="k", client=client)


def test_voice_endpoint_needs_the_key(tmp_path, monkeypatch):
    """Without MODEL_API_KEY the page is told to type instead; Meta is never called."""
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setattr(voice, "transcribe", lambda *a, **k: pytest.fail("called Meta without a key"))
    client = TestClient(create_app(Database(tmp_path / "v.db")))
    res = client.post("/api/voice", content=make_wav(), headers={"Content-Type": "audio/wav"})
    assert res.status_code == 503
    assert "MODEL_API_KEY" in res.json()["detail"]


def test_voice_endpoint_transcribes_and_parses(tmp_path, monkeypatch):
    """The page gets the words it heard and the action they mean."""
    monkeypatch.setenv("MODEL_API_KEY", "k")
    monkeypatch.setattr(voice, "transcribe", lambda wav, api_key: "Open deadlines.")
    client = TestClient(create_app(Database(tmp_path / "v.db")))
    res = client.post("/api/voice", content=make_wav(), headers={"Content-Type": "audio/wav"})
    assert res.status_code == 200
    assert res.json() == {"text": "Open deadlines.",
                          "command": {"action": "tab", "tab": "opportunities", "label": "Deadlines"}}


def test_voice_endpoint_refuses_bad_or_huge_audio(tmp_path, monkeypatch):
    """Bad audio is a 400 and an oversized body a 413, both before any call to Meta."""
    monkeypatch.setenv("MODEL_API_KEY", "k")
    monkeypatch.setattr(voice, "transcribe", lambda *a, **k: pytest.fail("called Meta for bad audio"))
    client = TestClient(create_app(Database(tmp_path / "v.db")))
    assert client.post("/api/voice", content=make_wav(channels=2)).status_code == 400
    assert client.post("/api/voice", content=b"\x00" * (voice.MAX_BYTES + 1)).status_code == 413


def test_voice_endpoint_reports_meta_failures(tmp_path, monkeypatch):
    """A Meta failure is a 502 with the reason, so the page can show it."""
    monkeypatch.setenv("MODEL_API_KEY", "k")

    def fail(wav, api_key):
        raise voice.VoiceError("Meta rejected MODEL_API_KEY.")

    monkeypatch.setattr(voice, "transcribe", fail)
    client = TestClient(create_app(Database(tmp_path / "v.db")))
    res = client.post("/api/voice", content=make_wav())
    assert res.status_code == 502
    assert res.json()["detail"] == "Meta rejected MODEL_API_KEY."


def test_voice_endpoint_refuses_other_websites(tmp_path, monkeypatch):
    """Another site can't spend the user's Meta credits through their browser."""
    monkeypatch.setenv("MODEL_API_KEY", "k")
    monkeypatch.setattr(voice, "transcribe", lambda *a, **k: pytest.fail("called Meta for another site"))
    client = TestClient(create_app(Database(tmp_path / "v.db")))
    res = client.post("/api/voice", content=make_wav(), headers={"Origin": "https://evil.example"})
    assert res.status_code == 403


def test_dashboard_has_a_mic_and_reads_answers_aloud(tmp_path):
    """The page has a Talk button in the bar and on Ask, records 16 kHz WAV and speaks replies."""
    html = TestClient(create_app(Database(tmp_path / "v.db"))).get("/").text
    assert 'id="btn-voice"' in html
    assert 'id="btn-ask-voice"' in html
    assert "/api/voice" in JS
    assert "speechSynthesis" in JS
    assert "16000" in JS
    assert "Talk" in html


@pytest.mark.parametrize("reply", [b"not json", b"[1, 2]"])
def test_transcribe_turns_a_malformed_reply_into_a_clear_error(reply):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=reply)))
    with pytest.raises(voice.VoiceError, match="unexpected"):
        voice.transcribe(make_wav(), api_key="k", client=client)


def test_voice_endpoint_refuses_a_huge_upload_from_its_length(tmp_path, monkeypatch):
    """The declared length is checked before the body is read into memory."""
    monkeypatch.setenv("MODEL_API_KEY", "k")
    client = TestClient(create_app(Database(tmp_path / "v.db")))
    res = client.post("/api/voice", content=make_wav(), headers={"Content-Length": str(voice.MAX_BYTES + 1)})
    assert res.status_code == 413
