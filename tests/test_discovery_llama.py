"""Tests for Llama model wiring and trend naming."""

from datetime import UTC, datetime, timedelta

import pytest
from graphflow import MockModel, ModelResponse, OpenAIModel

from gleaner.db import Database
from gleaner.llama import get_llama_model, is_meta_model
from gleaner.models import Item, SourceRow
from gleaner.trends import detect_trends, name_trend_cluster


def test_get_llama_model_defaults():
    m_judge = get_llama_model("judge")
    assert isinstance(m_judge, OpenAIModel)
    assert m_judge.model_name == "llama3.2:3b"
    assert "11434" in m_judge.base_url

    m_writer = get_llama_model("writer")
    assert isinstance(m_writer, OpenAIModel)
    assert m_writer.model_name == "llama3.2:3b"

    m_ask = get_llama_model("ask")
    assert m_ask.model_name == "llama3.2:3b"  # the one Llama installed; 8b answered nothing (2026-11-03)

    m_answer = get_llama_model("answer")
    assert isinstance(m_answer, OpenAIModel)
    assert m_answer.model_name == "muse-spark-1.3"
    assert m_answer.base_url == "https://api.meta.ai/v1"


def test_get_llama_model_env_override(monkeypatch):
    monkeypatch.setenv("LLAMA_JUDGE_MODEL", "llama3.2:1b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://hosted-llama:8000/v1")
    m = get_llama_model("judge")
    assert m.model_name == "llama3.2:1b"
    assert m.base_url == "http://hosted-llama:8000/v1"


def test_get_llama_model_global_override(monkeypatch):
    monkeypatch.delenv("GLEANER_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.3:70b")
    m = get_llama_model("ask")
    assert m.model_name == "llama3.3:70b"

    # GLEANER_MODEL takes precedence over OLLAMA_MODEL
    monkeypatch.setenv("GLEANER_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    m2 = get_llama_model("writer")
    assert m2.model_name == "meta-llama/Llama-3.1-8B-Instruct"

    # A role's own variable beats the global ones (the judge stays small)
    monkeypatch.setenv("LLAMA_JUDGE_MODEL", "llama3.2:3b")
    assert get_llama_model("judge").model_name == "llama3.2:3b"

    # Explicit model_name argument takes highest precedence
    m3 = get_llama_model("writer", model_name="llama3.1:8b-instruct-q4_K_M")
    assert m3.model_name == "llama3.1:8b-instruct-q4_K_M"


@pytest.mark.parametrize("name", ["llama-guard3:1b", "prompt-guard", "meta-llama/Llama-Prompt-Guard-2-86M"])
def test_meta_safety_models_are_accepted(name):
    assert get_llama_model("prompt_guard", model_name=name).model_name == name


@pytest.mark.parametrize("name", ["muse-spark-1.1", "meta-models/muse-spark-open-1026", "hf.co/unsloth/Muse-Glimmer-30B-GGUF"])
def test_official_hackathon_meta_models_are_accepted(name):
    # The hackathon's official list is the Muse family; Llama and Prompt Guard stay accepted
    assert get_llama_model("writer", model_name=name).model_name == name


def test_muse_spark_goes_to_the_meta_model_api_with_its_own_key(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")  # the local judge keeps Ollama

    hosted = get_llama_model("writer", model_name="muse-spark-1.1")
    local = get_llama_model("judge", model_name="llama3.2:3b")

    assert hosted.base_url == "https://api.meta.ai/v1"
    assert hosted.api_key == "test-key"
    assert local.base_url == "http://localhost:11434/v1"


def test_hosted_muse_reasons_briefly_with_room_left_for_the_answer():
    # Muse Spark's reasoning tokens count against max_tokens: at the 4096 default, a live answer
    # call spent them all and returned truncated JSON (2026-11-01)
    hosted = get_llama_model("answer", model_name="muse-spark-1.1")
    local = get_llama_model("judge", model_name="llama3.2:3b")
    custom = get_llama_model("answer", model_name="muse-spark-1.1", max_tokens=2048, reasoning_effort="high")

    assert hosted.max_tokens == 8192
    assert hosted.extra_kwargs["reasoning_effort"] == "low"
    assert "reasoning_effort" not in local.extra_kwargs
    assert (custom.max_tokens, custom.extra_kwargs["reasoning_effort"]) == (2048, "high")


def test_non_meta_models_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="not a Meta model"):
        get_llama_model("ask", model_name="Qwen/Qwen3.5-0.8B:latest")
    monkeypatch.setenv("OLLAMA_MODEL", "Qwen/Qwen3.5-0.8B:latest")
    monkeypatch.delenv("GLEANER_MODEL", raising=False)
    with pytest.raises(ValueError, match="OLLAMA_MODEL"):
        get_llama_model("writer")


def test_trend_naming_heuristic_and_model():
    titles = [
        "Test-Time Compute Breakthrough from Lab Alpha",
        "Scaling Test-Time Search in Reasoning Models",
        "Test-Time Search Compute Scaling Laws",
    ]

    # Without model: heuristic from first title
    h_label = name_trend_cluster(titles, model=None)
    assert h_label == "Test-Time Compute Breakthrough from Lab Alpha"

    # With model: uses model's generated name
    mock_model = MockModel(
        responses=[ModelResponse(content='"Test-Time Reasoning Scaling"')]
    )
    m_label = name_trend_cluster(titles, model=mock_model)
    assert m_label == "Test-Time Reasoning Scaling"
    assert mock_model.call_count == 1


def test_detect_trends_with_model(tmp_path):
    db = Database(tmp_path / "trends_model.db")
    now = datetime.now(UTC)

    for i in range(1, 4):
        src_id = f"source-{i}"
        db.upsert_source(SourceRow(id=src_id, type="rss", params={"url": f"http://s{i}.com"}, cadence_s=3600))
        item = Item(
            id=f"item-trend-{i}",
            source_id=src_id,
            title=f"Reasoning Scaling Breakthrough from Lab {i}",
            text="Scaling test-time search in reasoning models.",
            published_at=now - timedelta(hours=i * 6),
            story_id="story-reasoning",
        )
        db.upsert_item(item)

    mock_model = MockModel(
        responses=[ModelResponse(content="Inference-Time Search Trends")]
    )
    trends = detect_trends(db, now=now, min_distinct_sources=3, z_threshold=2.0, model=mock_model)
    assert len(trends) == 1
    assert trends[0].label == "Inference-Time Search Trends"
    db.close()


@pytest.mark.parametrize("name", ["deepseek-r1-distill-llama-70b", "qwen-llama-merge", "openchat/llamafied-7b"])
def test_other_vendors_models_built_on_llama_are_rejected(name):
    assert is_meta_model(name) is False
