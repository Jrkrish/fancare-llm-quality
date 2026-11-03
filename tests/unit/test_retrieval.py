from dataclasses import replace

import pytest

from app.retrieval import HybridRetriever


def test_corpus_loaded(retriever):
    assert len(retriever.documents) == 19
    assert all(c.chunk_id.startswith(c.doc_id + ":") for c in retriever.chunks)
    assert retriever.documents["DOC-13"].meta["status"] == "superseded"
    assert retriever.documents["DOC-18"].meta["status"] == "community"


@pytest.mark.parametrize(
    "query, expected_doc",
    [
        ("what does error E7 mean", "DOC-07"),
        ("does the fan work on 5 GHz wifi", "DOC-03"),
        ("how many phones can pair over bluetooth", "DOC-04"),
        ("how long is the motor warranty", "DOC-12"),
        ("restocking fee for opened fans", "DOC-14"),
        ("which fans can be installed outdoors", "DOC-16"),
    ],
)
def test_relevant_doc_in_top_k(retriever, query, expected_doc):
    assert expected_doc in {c.doc_id for c in retriever.search(query)}


@pytest.mark.parametrize("query", ["what is the capital of France", "give me a pasta recipe"])
def test_off_topic_returns_no_evidence(retriever, query):
    assert retriever.search(query) == []


@pytest.mark.parametrize("mode", ["bm25", "vector"])
def test_single_retriever_modes(settings, mode):
    r = HybridRetriever(replace(settings, retriever_mode=mode, rerank=False))
    results = r.search("remote control battery type")
    assert len(results) == settings.retrieval_top_k
    assert "DOC-06" in {c.doc_id for c in results}
