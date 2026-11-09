"""PubMed relevance 快照和逐字切片；不依赖历史脚本。"""
from __future__ import annotations
import bisect
import json
import re
import urllib.parse
import urllib.request
import urllib.error
import time
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from .state import digest, ConfigurationError, save
sha = digest

def token_count(tokenizer, title: str, text: str) -> int:
    return len(tokenizer(title + " [SEP] " + text, add_special_tokens=False)["input_ids"])


def spans(title: str, text: str, tokenizer, limit: int = 400) -> list[tuple[int, int]]:
    """按句末空白优先切分；词边界和 tokenizer offset 作超长单句后备。"""
    if token_count(tokenizer, title, text) <= limit:
        return [(0, len(text))]
    prefix_tokens = len(tokenizer(title + " [SEP] ", add_special_tokens=False)["input_ids"])
    budget = limit - prefix_tokens - 4  # 给连接处的分词变化留余量，最终仍逐片实测。
    if budget < 1:
        raise ValueError("Title leaves no room for any abstract token")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if not offsets:
        raise ValueError("Long abstract has no token offsets")
    token_ends = [end for _start, end in offsets]
    sentence_ends = [m.end() for m in re.finditer(r"(?<=[.!?。！？])\s+|\n+", text)]
    word_ends = [m.end() for m in re.finditer(r"\s+", text)]
    out = []
    start = 0
    while start < len(text):
        if token_count(tokenizer, title, text[start:]) <= limit:
            out.append((start, len(text)))
            break
        token_index = bisect.bisect_right(token_ends, start)
        candidate_index = min(len(token_ends) - 1, token_index + budget - 1)
        max_end = token_ends[candidate_index]
        if max_end <= start:
            raise ValueError("Tokenizer offsets did not advance")
        # 有足够近的句末就用句末；否则选择最近词边界，最后才在字符边界切。
        floor = start + (max_end - start) // 2
        sentence = [end for end in sentence_ends if floor <= end <= max_end]
        words = [end for end in word_ends if start < end <= max_end]
        end = sentence[-1] if sentence else words[-1] if words else max_end
        while token_count(tokenizer, title, text[start:end]) > limit:
            prior = bisect.bisect_left(token_ends, end) - 1
            if prior < 0 or token_ends[prior] <= start:
                raise ValueError("A single token exceeds the chunk limit")
            end = token_ends[prior]
        out.append((start, end))
        start = end
    if out[0][0] != 0 or out[-1][1] != len(text) or any(a[1] != b[0] for a, b in zip(out, out[1:])):
        raise AssertionError("Chunk spans do not cover original text exactly")
    if "".join(text[a:b] for a, b in out) != text:
        raise AssertionError("Chunk text cannot reconstruct parent")
    return out


def chunk_rows(rows: list[dict[str, Any]], tokenizer, limit: int = 400
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    children = []
    span_records = []
    parent_to_children = {}
    used_ids: set[str] = set()
    for parent in rows:
        parent_id, original = parent["chunk_id"], parent["text"]
        parent_hash = sha(original.encode("utf-8"))
        if parent.get("content_hash") and parent["content_hash"] != parent_hash:
            raise ValueError(f"Parent content hash mismatch: {parent_id}")
        segments = spans(parent.get("title", ""), original, tokenizer, limit)
        child_ids = []
        for start, end in segments:
            piece = original[start:end]
            if len(segments) == 1:
                child = dict(parent)  # 短摘要完全保留原 ID、原元数据和内容。
            else:
                child = dict(parent)
                child_id = "chunk:" + sha(f"{parent_id}:{start}:{end}".encode("utf-8"))[:20]
                child.update(chunk_id=child_id, text=piece,
                             position=f"{parent.get('position', 'abstract')} chars[{start}:{end}]",
                             content_hash=sha(piece.encode("utf-8")), derived_from=None)
            if child["chunk_id"] in used_ids:
                raise ValueError(f"Duplicate child chunk ID: {child['chunk_id']}")
            used_ids.add(child["chunk_id"])
            count = token_count(tokenizer, child.get("title", ""), child["text"])
            if count > limit:
                raise AssertionError(f"Chunk exceeds {limit} tokens")
            children.append(child)
            child_ids.append(child["chunk_id"])
            span_records.append({"parent_chunk_id": parent_id, "child_chunk_id": child["chunk_id"],
                                 "parent_content_sha256": parent_hash, "char_start": start,
                                 "char_end": end, "child_content_sha256": sha(piece.encode("utf-8")),
                                 "title_sep_text_tokens": count})
        if "".join(c["text"] for c in children[-len(segments):]) != original:
            raise AssertionError(f"Parent cannot be reconstructed: {parent_id}")
        parent_to_children[parent_id] = child_ids
    return children, span_records, parent_to_children



_last_request = 0.0


def pubmed_records(root):
    """只识别返回条目本身，不把参考文献中的 PMID 当作下载结果。"""
    for entry in root:
        if entry.tag == "PubmedArticle":
            document = entry.find("MedlineCitation")
            kind = "journal_article"
        elif entry.tag == "PubmedBookArticle":
            document = entry.find("BookDocument")
            kind = "book_article"
        else:
            continue
        if document is not None:
            yield kind, document


def fetch_pubmed(query, cache, max_records=8, timeout=60):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    def request(endpoint, params):
        url = base + endpoint + "?" + urllib.parse.urlencode(params)
        def operation():
            global _last_request
            time.sleep(max(0, 0.4 - (time.monotonic() - _last_request)))
            _last_request = time.monotonic()
            try:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    raw = response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                save(cache.root / "raw" / (digest(url) + ".error.json"),
                     {"url": url, "status": exc.code, "raw": exc.read().decode("utf-8", errors="replace")})
                if exc.code in {400, 404, 422}:
                    raise ValueError(f"PubMed query request rejected: HTTP {exc.code}") from exc
                raise
            save(cache.root / "raw" / (digest(url) + "." + digest(raw.encode()) + ".json"), {"url": url, "raw": raw})
            if endpoint == "esearch.fcgi":
                value = json.loads(raw)
                ids = value["esearchresult"]["idlist"]
                if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
                    raise ValueError("Invalid PubMed idlist")
            else:
                root = ET.fromstring(raw)
                if root.tag != "PubmedArticleSet":
                    raise ValueError("Invalid PubMed article response")
                requested = set(params["id"].split(","))
                returned = {document.findtext("PMID") for _, document in pubmed_records(root)}
                if requested - returned:
                    raise ValueError("EFetch omitted requested PMIDs: " + ",".join(sorted(requested - returned)))
            return {"url": url, "raw": raw}
        return cache.get("pubmed:" + url, {"url": url}, operation)
    search = request("esearch.fcgi", {"db":"pubmed", "term":query, "retmax":max_records,
                                     "retmode":"json", "sort":"relevance"})
    ids = json.loads(search["raw"])["esearchresult"]["idlist"]
    if not ids:
        return {"status":"empty", "ids":[], "evidence":[]}
    fetched = request("efetch.fcgi", {"db":"pubmed", "id":",".join(ids), "retmode":"xml"})
    rows = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    found = set()
    record_types = {}
    for kind, document in pubmed_records(ET.fromstring(fetched["raw"])):
        pmid = document.findtext("PMID")
        if pmid:
            found.add(pmid)
            record_types[pmid] = kind
        # 书籍也只使用 EFetch 给出的摘要，不把目录、参考文献或全文拼成摘要。
        content = document.find("Article") if kind == "journal_article" else document
        if content is None:
            continue
        title_node = content.find("ArticleTitle")
        if title_node is None and kind == "book_article":
            title_node = content.find("Book/BookTitle")
        title = "".join(title_node.itertext()) if title_node is not None else ""
        text = " ".join("".join(n.itertext()) for n in content.findall("Abstract/AbstractText"))
        if pmid and text.strip():
            rows.append({"chunk_id":f"PMID:{pmid}:abstract", "document_id":f"PMID:{pmid}",
                         "text":text, "title":title, "source":f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                         "position":"book_abstract" if kind == "book_article" else "abstract",
                         "retrieved_at":fetched_at, "content_hash":digest(text.encode()),
                         "derived_from":None})
    missing = set(ids) - found
    if missing:
        raise ValueError("EFetch omitted requested PMIDs: " + ",".join(sorted(missing)))
    return {"status":"ok" if rows else "no_abstracts", "ids":ids, "evidence":rows,
            "record_types": record_types,
            "without_abstract": sorted(found - {r["document_id"].split(":")[1] for r in rows})}


def check_lengths(dense, query, evidence):
    """实测三路 tokenizer；现有检索器的 truncation 不能掩盖超长输入。"""
    if not all(hasattr(dense, x) for x in ("qtok", "atok", "rtok")):
        if type(dense).__name__ != "MockDense":
            raise ConfigurationError("Live retrieval requires all three tokenizers")
        return
    if len(dense.qtok(query, truncation=False)["input_ids"]) > 64:
        raise ValueError("MedCPT retrieval query exceeds 64 tokens")
    for item in evidence:
        if len(dense.atok(item.title, item.text, truncation=False)["input_ids"]) > 512:
            raise ValueError("MedCPT article pair exceeds 512 tokens: " + item.chunk_id)
        passage = f"{item.title} [SEP] {item.text}" if item.title else item.text
        if len(dense.rtok(query, passage, truncation=False)["input_ids"]) > 512:
            raise ValueError("MedCPT cross pair exceeds 512 tokens: " + item.chunk_id)
