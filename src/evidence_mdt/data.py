from __future__ import annotations

import hashlib
import json
import random
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import Evidence, Question


PQA_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
MX_URL = "https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA/resolve/main/Text/test.jsonl"
MEDQA_URL = ("https://datasets-server.huggingface.co/rows?dataset=bigbio%2Fmed_qa"
             "&config=med_qa_en_4options_source&split=train&offset=0&length=100")


def read_json_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    obj = json.loads(raw)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [{"_source_id": key, **value} for key, value in obj.items()]
    raise ValueError("Expected JSON array/object or JSONL")


def normalize_question(item: dict[str, Any], dataset: str, index: int) -> Question:
    did = dataset.lower()
    qid = str(item.get("question_id") or item.get("id") or item.get("_source_id") or f"{dataset}-{index}")
    if did.startswith("pubmedqa"):
        # Official PQA JSON: QUESTION, CONTEXTS, final_decision, LONG_ANSWER.
        stem = item.get("QUESTION") or item.get("question")
        contexts = item.get("CONTEXTS") or item.get("contexts") or []
        context = "\n".join(contexts) if isinstance(contexts, list) else str(contexts)
        gold = item.get("final_decision") or item.get("gold")
        options = {"yes": "yes", "no": "no", "maybe": "maybe"}
    else:
        stem = item.get("question") or item.get("stem")
        options = item.get("options")
        if isinstance(options, list):
            options = {x["key"]: x["value"] for x in options}
        gold = item.get("answer_idx") or item.get("label") or item.get("gold")
        context = item.get("context") or ""
    if not isinstance(stem, str) or not stem.strip():
        raise ValueError(f"Question {qid} missing stem")
    if not isinstance(options, dict) or len(options) < 2:
        raise ValueError(f"Question {qid} missing options")
    if gold is not None and gold not in options:
        raise ValueError(f"Question {qid} has invalid gold label")
    # Deliberately ignores answer, long_answer, and all other source fields.
    return Question(qid, dataset, stem, options, gold, context)


def prepare_pilot(output_dir: Path, seed: int = 20261008) -> dict[str, Any]:
    """固定抽样 30 MedQA、20 PubMedQA，以及单独保留的 20 MedXpertQA。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {"MedQA": MEDQA_URL, "PubMedQA": PQA_URL, "MedXpertQA": MX_URL}
    raw = {}
    # 镜像每页上限为 100；固定前十页形成 1000 行抽样框，并逐页保存原文和哈希。
    medqa_pages = []
    for offset in range(0, 1000, 100):
        url = MEDQA_URL.replace("offset=0", f"offset={offset}")
        with urllib.request.urlopen(url, timeout=120) as response:
            page_raw = response.read()
        page_name = f"MedQA.page{offset // 100:02d}.raw"
        (output_dir / page_name).write_bytes(page_raw)
        page = json.loads(page_raw)
        if len(page["rows"]) != 100:
            raise ValueError(f"MedQA mirror returned fewer than 100 rows at offset {offset}")
        medqa_pages.extend(page["rows"])
    for name, url in sources.items():
        if name == "MedQA":
            continue
        with urllib.request.urlopen(url, timeout=120) as response:
            raw[name] = response.read()
        (output_dir / f"{name}.raw").write_bytes(raw[name])
    medqa_rows = medqa_pages
    pqa = json.loads(raw["PubMedQA"])
    mx_lines = [json.loads(line) for line in raw["MedXpertQA"].splitlines() if line.strip()]
    pools = {"MedQA": [(str(x["row_idx"]), x["row"]) for x in medqa_rows],
             "PubMedQA": [(str(k), v) for k, v in pqa.items()],
             "MedXpertQA": [(str(x["id"]), x) for x in mx_lines]}
    counts = {"MedQA": 30, "PubMedQA": 20, "MedXpertQA": 20}
    selected = {}
    for name, count in counts.items():
        pool = pools[name]
        if len(pool) < count:
            raise ValueError(f"{name} pool too small")
        # 各数据集独立随机流；以后改某一数据源不会改变其他两组的题号。
        rng = random.Random(f"{seed}:{name}")
        chosen = rng.sample(range(len(pool)), count)
        items = []
        for position in chosen:
            source_id, item = pool[position]
            q = normalize_question({"_source_id": source_id, **item}, name, position)
            items.append(q)
        selected[name] = {"questions": items, "source_ids": [pool[i][0] for i in chosen]}
        write_jsonl(output_dir / f"{name}.jsonl", items)
    write_jsonl(output_dir / "pilot50.jsonl",
                selected["MedQA"]["questions"] + selected["PubMedQA"]["questions"])
    medqa_page_manifest = []
    for offset in range(0, 1000, 100):
        page_name = f"MedQA.page{offset // 100:02d}.raw"
        medqa_page_manifest.append({"url": MEDQA_URL.replace("offset=0", f"offset={offset}"),
                                    "raw_file": page_name,
                                    "raw_sha256": hashlib.sha256((output_dir / page_name).read_bytes()).hexdigest()})
    manifest = {"seed": seed, "selection_method": "random.Random.sample without replacement",
                "pilot": "30 MedQA + 20 PubMedQA", "reserve": "20 MedXpertQA Text; separate file",
                "sources": {name: {"url": sources[name],
                                   "raw_sha256": hashlib.sha256(raw[name]).hexdigest() if name in raw else None,
                                   "raw_file": f"{name}.raw" if name in raw else None,
                                   "pages": medqa_page_manifest if name == "MedQA" else None,
                                   "frame_size": len(pools[name]),
                                   "frame_note": ("first 1000 train rows from third-party BigBio mirror"
                                                  if name == "MedQA" else
                                                  "official ori_pqal.json" if name == "PubMedQA" else
                                                  "official Text/test.jsonl; exploratory reserve, not train"),
                                   "selected_source_ids": selected[name]["source_ids"]}
                            for name in sources}}
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def write_jsonl(path: Path, items: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(x) if hasattr(x, "__dataclass_fields__") else x,
                                       ensure_ascii=False) + "\n" for x in items), encoding="utf-8")


def load_questions(path: Path) -> list[Question]:
    return [Question(**item) for item in read_json_records(path)]


def local_text_corpus(path: Path, source: str = "local") -> list[Evidence]:
    texts = sorted([p for p in path.rglob("*.txt") if p.is_file()])
    out = []
    for doc in texts:
        content = doc.read_text(encoding="utf-8")
        chunks = [content[i:i + 1600] for i in range(0, len(content), 1400)]
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                # 相同文件名可位于不同子目录，片段 ID 必须仍然唯一。
                relative_id = hashlib.sha256(str(doc.relative_to(path)).encode("utf-8")).hexdigest()[:16]
                out.append(Evidence(f"{relative_id}:{i}", str(doc), chunk, source,
                                    f"char:{i*1400}", "", digest))
    return out


def pubmed_snapshot(query: str, max_records: int = 20) -> tuple[list[Evidence], dict[str, Any]]:
    """Fetch PubMed abstracts via NCBI E-utilities and return an auditable snapshot."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = base + "esearch.fcgi?" + urllib.parse.urlencode({"db": "pubmed", "term": query,
                                                                   "retmax": max_records, "retmode": "json"})
    with urllib.request.urlopen(search_url, timeout=30) as response:
        search = json.loads(response.read().decode("utf-8"))
    ids = search["esearchresult"]["idlist"]
    fetched_at = datetime.now(timezone.utc).isoformat()
    if not ids:
        return [], {"query": query, "ids": [], "fetched_at": fetched_at, "search_url": search_url}
    fetch_url = base + "efetch.fcgi?" + urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids),
                                                                "retmode": "xml"})
    with urllib.request.urlopen(fetch_url, timeout=60) as response:
        raw = response.read()
    root = ET.fromstring(raw)
    out = []
    article_metadata = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//MedlineCitation/PMID")
        title = "".join(article.find(".//ArticleTitle").itertext()) if article.find(".//ArticleTitle") is not None else ""
        doi = next((x.text for x in article.findall(".//ArticleId") if x.attrib.get("IdType") == "doi"), None)
        abstract = " ".join("".join(x.itertext()) for x in article.findall(".//Abstract/AbstractText"))
        if pmid and abstract:
            out.append(Evidence(f"PMID:{pmid}:abstract", f"PMID:{pmid}", abstract,
                                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "abstract",
                                fetched_at, hashlib.sha256(abstract.encode()).hexdigest(), title=title))
            article_metadata.append({"pmid": pmid, "doi": doi, "title": title})
    return out, {"query": query, "ids": ids, "fetched_at": fetched_at,
                 "search_url": search_url, "fetch_url": fetch_url,
                 "raw_xml_sha256": hashlib.sha256(raw).hexdigest(),
                 "article_metadata": article_metadata, "_raw_xml": raw.decode("utf-8")}
