"""Phase 0: Embedder protocol and FakeEmbedder tests."""



from gleaner.embed import FakeEmbedder, cosine_similarity


def test_fake_embedder_deterministic():
    embedder = FakeEmbedder(dim=64)
    v1 = embedder.embed(["Hello world", "Agent workflows"])
    v2 = embedder.embed(["Hello world", "Agent workflows"])
    assert v1 == v2
    assert len(v1[0]) == 64
    assert len(v1[1]) == 64


def test_fake_embedder_semantic_similarity():
    embedder = FakeEmbedder(dim=128)
    texts = [
        "autonomous agents and langgraph workflows",
        "langgraph agents compiling workflows",
        "delicious chocolate cake recipes and baking",
    ]
    vectors = embedder.embed(texts)
    sim_related = cosine_similarity(vectors[0], vectors[1])
    sim_unrelated = cosine_similarity(vectors[0], vectors[2])
    assert sim_related > sim_unrelated
    assert sim_related > 0.4
    assert sim_unrelated < 0.3


def test_cosine_similarity_edge_cases():
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-6
    assert abs(cosine_similarity([0.0, 0.0], [1.0, 0.0]) - 0.0) < 1e-6


def test_ollama_embedder_posts_batches_to_api_embed(monkeypatch):
    import json

    import httpx

    from gleaner.embed import Embedder, OllamaEmbedder

    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append({"url": str(request.url), **body})
        return httpx.Response(200, json={"embeddings": [[float(len(t)), 0.0, 1.0] for t in body["input"]]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = OllamaEmbedder(base_url="http://ollama.test:11434/v1", dim=3, batch_size=2, client=client)

    vecs = embedder.embed(["a", "bb", "ccc"])

    assert isinstance(embedder, Embedder)
    assert embedder.model_id == "nomic-embed-text"
    assert vecs == [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]
    assert [r["input"] for r in requests] == [["a", "bb"], ["ccc"]]
    assert {r["url"] for r in requests} == {"http://ollama.test:11434/api/embed"}
    assert {r["model"] for r in requests} == {"nomic-embed-text"}


def test_ollama_embedder_rejects_wrong_dimension():
    import httpx
    import pytest

    from gleaner.embed import OllamaEmbedder

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"embeddings": [[0.1, 0.2]]}))
    embedder = OllamaEmbedder(model_id="nomic-embed-text", dim=3, client=httpx.Client(transport=transport))
    with pytest.raises(ValueError, match="dimension"):
        embedder.embed(["text"])


def test_embedder_change_marks_vectors_stale(tmp_path):
    from gleaner.db import Database

    db = Database(tmp_path / "embedder_meta.db", vec_dim=4)
    assert db.sync_embedder("fake-bow") is False  # first model is recorded
    db.upsert_item_vector("it-1", [1.0, 0.0, 0.0, 0.0])
    db.upsert_interest_vector("int-1", [1.0, 0.0, 0.0, 0.0])
    assert db.sync_embedder("fake-bow") is False

    assert db.sync_embedder("nomic-embed-text") is True  # a new vector space: old vectors are dropped
    assert db.get_item_vector("it-1") is None
    assert db.get_interest_vector("int-1") is None
    assert db.sync_embedder("nomic-embed-text") is False
    db.close()


def test_collect_batch_records_the_embedder(tmp_path):
    from gleaner.db import Database
    from gleaner.models import Item
    from gleaner.workflows import process_and_score_items

    db = Database(tmp_path / "embedder_batch.db")
    item = Item(id="it-e", title="Agent workflows", text="Graphflow agents compile workflows.")
    db.upsert_item(item)
    process_and_score_items(db, [item], embedder=FakeEmbedder(dim=db.vec_dim, model_id="fake-a"))

    assert db.sync_embedder("fake-a") is False
    db.close()
