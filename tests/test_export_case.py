import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_case import build_case, render_html  # noqa: E402


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")


def fixture(tmp_path):
    run = "q:repeat-1:natural:semantic_retrieval"
    result = {"run_id": run, "question_id": "q", "dataset": "MedQA", "group": "semantic_retrieval",
              "repeat_id": 1, "variant": "natural", "status": "ok", "attempt_id": "good",
              "gold": "A", "answer": "B", "roles": ["r1", "r2", "r3", "r4"],
              "supplemental_chunk_ids": ["PMID:12345678:abstract"]}
    turns = []
    for attempt in ("old", "good"):
        turns.append({"run_id": run, "attempt_id": attempt, "type": "evidence",
                      "chunk_id": "PMID:12345678:abstract" if attempt == "good" else "old-chunk",
                      "document_id": "PMID:12345678" if attempt == "good" else "old-doc",
                      "text": "<img src=x onerror=alert(1)>" if attempt == "good" else "OLD SECRET"})
        turns.append({"run_id": run, "attempt_id": attempt, "type": "turn",
                      "turn_id": f"{run}:r1:R0", "role": "r1", "round_id": 0, "order": 0,
                      "answer": "A" if attempt == "good" else "X", "probabilities": {"A": .8, "B": .2},
                      "claims": [], "visible_messages": [], "visible_payload": {
                          "evidence": [{"chunk_id": "PMID:12345678:abstract", "document_id": "PMID:12345678",
                                        "text": "<script>alert(1)</script>", "headers": "SECRET HEADER"}],
                          "messages": [{"id": "peer", "kind": "opinion", "answer": "B",
                                        "opinion": "<b>text</b>", "api_key": "SECRET KEY"}],
                          "wire_request": "SECRET REQUEST"}})
    turns.append({"run_id": run, "attempt_id": "good", "type": "edge", "relation": "cited",
                  "source_id": "PMID:12345678:abstract", "target_id": f"{run}:r1:R0"})
    turns.append({"run_id": run, "attempt_id": "good", "type": "edge", "relation": "exposed",
                  "source_id": "old-chunk", "target_id": f"{run}:r1:R0"})
    write_rows(tmp_path / "results.jsonl", [{**result, "status": "failed", "attempt_id": "old"}, result])
    write_rows(tmp_path / "events.jsonl", turns)
    return result


def test_success_attempt_only_and_explicit_edges_only(tmp_path):
    fixture(tmp_path)
    case = build_case(tmp_path, "q", "semantic_retrieval", 1)
    assert case["attempt_id"] == "good"
    assert {e["source_id"] for e in case["explicit_edges"]} == {"PMID:12345678:abstract"}
    assert "old-chunk" not in case["evidence"]
    assert case["evidence"]["PMID:12345678:abstract"]["pubmed_url"] == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


def test_safe_self_contained_html_excludes_request_metadata(tmp_path):
    fixture(tmp_path)
    html = render_html(build_case(tmp_path, "q", "semantic_retrieval", 1))
    assert "SECRET HEADER" not in html and "SECRET KEY" not in html and "SECRET REQUEST" not in html
    assert "OLD SECRET" not in html
    assert "<script>alert(1)</script>" not in html
    assert "\\u003cscript\\u003ealert(1)\\u003c/script\\u003e" in html
    assert "https://cdn" not in html
    assert "function sourceLabel" in html and "function targetLabel" in html
    assert "原始来源 ID" in html and "原始目标 ID" in html
    assert "每条箭头对应一条日志中的显式引用边" in html
    payload = re.search(r'<script id="case-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(payload)["attempt_id"] == "good"
    class Assets(HTMLParser):
        def __init__(self):
            super().__init__()
            self.external = []
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if (tag == "script" and "src" in attrs) or (tag == "link" and attrs.get("rel") == "stylesheet"):
                self.external.append((tag, attrs))
    parser = Assets()
    parser.feed(html)
    assert parser.external == []


def test_duplicate_success_requires_attempt_selection(tmp_path):
    result = fixture(tmp_path)
    write_rows(tmp_path / "results.jsonl", [result, {**result, "attempt_id": "old"}])
    with pytest.raises(ValueError, match="恰好一条"):
        build_case(tmp_path, "q", "semantic_retrieval", 1)
    assert build_case(tmp_path, "q", "semantic_retrieval", 1, "good")["attempt_id"] == "good"
