"""Hybrid retrieval over data/docs/*.md: BM25 + dense vectors, fused with RRF, then cross-encoder reranking.

The corpus is ~20 small files, so the index is rebuilt in memory at start-up and
can never go stale. Embedding and reranking models run locally via fastembed (ONNX).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi

from app.config import Settings

RRF_K = 60
CANDIDATES_PER_RETRIEVER = 10
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = set("a an and are as at be by can do for from how i if in is it my of on or the to what when with you your".split())


@dataclass
class Chunk:
    chunk_id: str  # "DOC-12:2" = second section of DOC-12
    doc_id: str
    title: str
    section: str
    status: str  # current | superseded | community
    text: str
    score: float | None = None


@dataclass
class Document:
    doc_id: str
    title: str
    meta: dict
    chunks: list[Chunk] = field(default_factory=list)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    match = re.match(r"^---\n(.*?)\n---\n", content, flags=re.DOTALL)
    if not match:
        return {}, content
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, content[match.end():]


def load_document(path: Path) -> Document:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    doc_id, title = meta["id"], meta["title"]
    doc = Document(doc_id=doc_id, title=title, meta=meta)
    # One chunk per "## " section; the "# " title line is dropped.
    sections = [s.strip() for s in re.split(r"(?m)^## ", body) if s.strip() and not s.strip().startswith("# ")]
    for i, section in enumerate(sections, start=1):
        heading, _, text = section.partition("\n")
        doc.chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:{i}",
                doc_id=doc_id,
                title=title,
                section=heading.strip(),
                status=meta.get("status", "current"),
                # Title + heading travel with the passage so both retrievers see the topic.
                text=f"{title} - {heading.strip()}\n{text.strip()}",
            )
        )
    return doc


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class HybridRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.documents = {d.doc_id: d for d in (load_document(p) for p in sorted(settings.docs_dir.glob("*.md")))}
        self.chunks = [c for d in self.documents.values() for c in d.chunks]
        if not self.chunks:
            raise RuntimeError(f"No documents found in {settings.docs_dir}")
        self.by_id = {c.chunk_id: c for c in self.chunks}

        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        cache = str(settings.model_cache_dir)
        self.embedder = TextEmbedding(settings.embedding_model, cache_dir=cache)
        self.vectors = np.array(list(self.embedder.passage_embed([c.text for c in self.chunks])))
        self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.reranker = TextCrossEncoder(settings.rerank_model, cache_dir=cache) if settings.rerank else None

    def _bm25_ranking(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [i for i in np.argsort(scores)[::-1][:CANDIDATES_PER_RETRIEVER] if scores[i] > 0]

    def _vector_ranking(self, query: str) -> list[int]:
        q = np.array(list(self.embedder.query_embed(query))[0])
        sims = self.vectors @ (q / np.linalg.norm(q))
        return list(np.argsort(sims)[::-1][:CANDIDATES_PER_RETRIEVER])

    def search(self, query: str) -> list[Chunk]:
        mode = self.settings.retriever_mode
        rankings = []
        if mode in ("hybrid", "bm25"):
            rankings.append(self._bm25_ranking(query))
        if mode in ("hybrid", "vector"):
            rankings.append(self._vector_ranking(query))
        if not rankings:
            raise ValueError(f"Unknown retriever_mode '{mode}'")

        # Reciprocal rank fusion.
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        order = sorted(fused, key=fused.get, reverse=True)

        if self.reranker is None:
            return [self._with_score(self.chunks[i], fused[i]) for i in order[: self.settings.retrieval_top_k]]

        candidates = [self.chunks[i] for i in order]
        scores = list(self.reranker.rerank(query, [c.text for c in candidates]))
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            self._with_score(c, s)
            for c, s in ranked[: self.settings.retrieval_top_k]
            if s >= self.settings.rerank_min_score
        ]

    @staticmethod
    def _with_score(chunk: Chunk, score: float) -> Chunk:
        return Chunk(**{**chunk.__dict__, "score": round(float(score), 4)})
