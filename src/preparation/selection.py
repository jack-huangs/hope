"""显式抽题命令：固定随机种子和排除来源，不隐式扫描历史目录。"""
from dataclasses import asdict
import json
from pathlib import Path
import random

from evidence_mdt.data import normalize_question
from evidence_mdt.interventions import validate_experiment_questions
from evidence_mdt.schema import Question
from .state import ConfigurationError, canonical, digest, jsonl, save


def select_questions(source: Path, output: Path, count: int, seed: int, exclude=()):
    source, output = source.resolve(), output.resolve()
    if count < 1:
        raise ConfigurationError("Question count must be positive")
    raw = source.read_text(encoding="utf-8")
    try:
        records = json.loads(raw)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(records, list):
        raise ConfigurationError("Question source must be a JSON array or JSONL")
    questions = [Question(**r) if "stem" in r and "question_id" in r else
                 normalize_question(r, r.get("dataset", "MedXpertQA"), i)
                 for i, r in enumerate(records)]
    validate_experiment_questions(questions)
    if any(not q.stem.strip() or len(q.options) < 2 or
           (q.gold is not None and q.gold not in q.options) for q in questions):
        raise ConfigurationError("Invalid question content in source")
    if len({q.question_id for q in questions}) != len(questions):
        raise ConfigurationError("Duplicate source question IDs")
    excluded, provenance = set(), []
    for path in exclude:
        path = Path(path).resolve()
        if path.is_dir():
            path = path / "questions.jsonl"
        from . import load_questions
        excluded.update(q.question_id for q in load_questions(path))
        provenance.append({"path": str(path), "sha256": digest(path.read_bytes())})
    pool = [q for q in questions if q.question_id not in excluded]
    if count > len(pool):
        raise ConfigurationError("Not enough eligible questions")
    chosen = random.Random(seed).sample(pool, count)
    manifest = {"source": str(source), "source_sha256": digest(source.read_bytes()),
                "seed": seed, "count": count, "frame_size": len(questions),
                "eligible_count": len(pool), "exclusions": provenance,
                "selected_ids": [q.question_id for q in chosen],
                "questions_sha256": digest([asdict(q) for q in chosen])}
    path = output / "selection_manifest.json"
    if output.exists() and any(output.iterdir()):
        if (not path.is_file() or json.loads(path.read_bytes()) != manifest or
                not (output / "questions.jsonl").is_file() or
                (output / "questions.jsonl").read_bytes() != b"".join(
                    canonical(asdict(q)) + b"\n"
                    for q in chosen)):
            raise ConfigurationError("Selection output differs; use a new output directory")
        return manifest
    jsonl(output / "questions.jsonl", [asdict(q) for q in chosen])
    save(path, manifest)
    return manifest
