from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

from .schema import Evidence


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


class DenseBackend(Protocol):
    def index(self, passages: list[str]) -> None: ...
    def query_scores(self, query: str) -> list[float]: ...
    def rerank_scores(self, query: str, passages: list[str]) -> list[float]: ...


class MockDense:
    """Lexical fixture backend, explicitly selected; never used as a live fallback."""
    def index(self, passages: list[str]) -> None:
        self.passages = passages

    def query_scores(self, query: str) -> list[float]:
        q = set(tokens(query))
        return [len(q & set(tokens(p))) for p in self.passages]

    def rerank_scores(self, query: str, passages: list[str]) -> list[float]:
        q = set(tokens(query))
        return [len(q & set(tokens(p))) for p in passages]


class MedCPTDense:
    """Official query/article encoders, inner product, and cross encoder on CPU."""
    def __init__(self, model_manifest: Path | None = None):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as exc:
            raise RuntimeError("Install the medcpt extra for the live dense backend") from exc
        self.torch = torch
        manifest = json.loads(model_manifest.read_text(encoding="utf-8"))["models"] if model_manifest else {}
        ids = ("ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder", "ncbi/MedCPT-Cross-Encoder")
        locations = [manifest[x]["snapshot_path"] if x in manifest else x for x in ids]
        self.qtok = AutoTokenizer.from_pretrained(locations[0])
        self.qmodel = AutoModel.from_pretrained(locations[0]).eval()
        self.atok = AutoTokenizer.from_pretrained(locations[1])
        self.amodel = AutoModel.from_pretrained(locations[1]).eval()
        self.rtok = AutoTokenizer.from_pretrained(locations[2])
        self.rmodel = AutoModelForSequenceClassification.from_pretrained(locations[2]).eval()
        # 保存模型仓库实际解析出的版本，避免日后同名模型悄悄变化。
        self.revisions = {name: (manifest[model_id]["revision"] if model_id in manifest else
                                 getattr(model.config, "_commit_hash", None))
                          for name, model_id, model in (("query", ids[0], self.qmodel),
                                                        ("article", ids[1], self.amodel),
                                                        ("cross", ids[2], self.rmodel))}
        self.vectors = None

    def load_vectors(self, path: Path) -> None:
        self.vectors = self.torch.load(path, map_location="cpu", weights_only=True)

    def _encode(self, tokenizer, model, texts: list[str], max_length: int):
        batches = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 16):
                encoded = tokenizer(texts[i:i + 16], truncation=True, padding=True,
                                    max_length=max_length, return_tensors="pt")
                batches.append(model(**encoded).last_hidden_state[:, 0, :].cpu())
        return self.torch.cat(batches) if batches else self.torch.empty((0, 768))

    def index_evidence(self, evidence: list[Evidence]) -> None:
        """官方 Article Encoder 的标题、摘要分别作为 tokenizer 的两个输入。"""
        batches = []
        with self.torch.no_grad():
            for i in range(0, len(evidence), 16):
                batch = evidence[i:i + 16]
                encoded = self.atok([e.title for e in batch], [e.text for e in batch],
                                    truncation=True, padding=True, max_length=512, return_tensors="pt")
                batches.append(self.amodel(**encoded).last_hidden_state[:, 0, :].cpu())
        self.vectors = self.torch.cat(batches) if batches else self.torch.empty((0, 768))

    def index(self, passages: list[str]) -> None:
        self.vectors = self._encode(self.atok, self.amodel, passages, 512)

    def query_scores(self, query: str) -> list[float]:
        q = self._encode(self.qtok, self.qmodel, [query], 64)[0]
        return (self.vectors @ q).tolist()

    def rerank_scores(self, query: str, passages: list[str]) -> list[float]:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(passages), 8):
                pair = [[query, p] for p in passages[i:i + 8]]
                encoded = self.rtok(pair, truncation=True, padding=True, max_length=512,
                                    return_tensors="pt")
                logits = self.rmodel(**encoded).logits.squeeze(-1).cpu().tolist()
                out.extend(logits if isinstance(logits, list) else [logits])
        return out


class Retriever:
    def __init__(self, evidence: list[Evidence], dense: DenseBackend):
        ids = [e.chunk_id for e in evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("Corpus chunk_id must be unique; intervention duplicates are appended after retrieval")
        # 只索引允许充当患者问题证据的来源；来源判断用元数据，绝不靠正文关键词。
        self.evidence = [e for e in evidence if allowed_source(e, allow_unknown=isinstance(dense, MockDense))]
        self.dense = dense
        self.docs = [Counter(tokens(e.title + " " + e.text)) for e in self.evidence]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / max(len(self.lengths), 1)
        self.df = Counter(word for doc in self.docs for word in doc)
        if self.evidence and getattr(dense, "vectors", None) is None:
            if hasattr(dense, "index_evidence"):
                dense.index_evidence(self.evidence)
            else:
                dense.index([article_text(e) for e in self.evidence])
        elif getattr(dense, "vectors", None) is not None and len(dense.vectors) != len(self.evidence):
            raise ValueError("Dense index does not match eligible corpus")
        self.last_trace: dict = {}

    def bm25(self, query: str) -> list[float]:
        n = len(self.docs)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            score = 0.0
            for word in set(tokens(query)):
                tf = doc[word]
                if tf:
                    idf = math.log(1 + (n - self.df[word] + 0.5) / (self.df[word] + 0.5))
                    score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / max(self.avgdl, 1)))
            out.append(score)
        return out

    def search(self, query: str, limit: int = 6, exclude: set[str] | None = None) -> list[Evidence]:
        if not self.evidence:
            self.last_trace = {"query": query, "excluded_chunk_ids": sorted(exclude or []),
                               "reason": "empty_corpus", "returned_chunk_ids": []}
            return []
        # 排除已见片段必须发生在两路召回排名之前，否则补检索会丢掉候选名额。
        eligible = [i for i, e in enumerate(self.evidence) if e.chunk_id not in (exclude or set())]
        if not eligible:
            self.last_trace = {"query": query, "excluded_chunk_ids": sorted(exclude or []),
                               "reason": "all_excluded", "returned_chunk_ids": []}
            return []
        b = self.bm25(query)
        d = self.dense.query_scores(query)
        if len(d) != len(self.evidence):
            raise RuntimeError("Dense score count mismatch")
        rankings = [sorted(eligible, key=lambda i: (-b[i], i))[:50],
                    sorted(eligible, key=lambda i: (-d[i], i))[:50]]
        rrf = Counter()
        for ranking in rankings:
            for rank, idx in enumerate(ranking, 1):
                rrf[self.evidence[idx].chunk_id] += 1 / (60 + rank)
        by_id = {e.chunk_id: e for e in self.evidence}
        candidates = [by_id[cid] for cid, _ in rrf.most_common(30)]
        if not candidates:
            return []
        scores = self.dense.rerank_scores(query, [article_text(e) for e in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda p: p[0], reverse=True)
        result = [e for _, e in ranked[:limit]]
        self.last_trace = {"query": query, "excluded_chunk_ids": sorted(exclude or []),
                           "bm25_top50": [{"chunk_id": self.evidence[i].chunk_id, "score": b[i]}
                                          for i in rankings[0]],
                           "dense_top50": [{"chunk_id": self.evidence[i].chunk_id, "score": d[i]}
                                          for i in rankings[1]],
                           "rrf_top30": [{"chunk_id": e.chunk_id, "score": rrf[e.chunk_id]}
                                         for e in candidates],
                           "reranked": [{"chunk_id": e.chunk_id, "score": score}
                                        for score, e in ranked],
                           "returned_chunk_ids": [e.chunk_id for e in result]}
        return result


def load_corpus(path: Path) -> list[Evidence]:
    return [Evidence(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def allowed_source(evidence: Evidence, allow_unknown: bool = False) -> bool:
    """来源白名单；unknown 仅供离线合成测试，真实运行必须有可审计来源。"""
    source = evidence.source.lower().strip()
    if source == "unknown":
        return allow_unknown
    if source in {"local", "pubmed", "pmc"}:
        return True
    if source.startswith("https://pubmed.ncbi.nlm.nih.gov/"):
        return True
    if source.startswith("https://pmc.ncbi.nlm.nih.gov/"):
        return True
    return False


def article_text(evidence: Evidence) -> str:
    """MedCPT Article Encoder 使用标题与摘要；无标题时仅保留正文。"""
    return f"{evidence.title} [SEP] {evidence.text}" if evidence.title else evidence.text
