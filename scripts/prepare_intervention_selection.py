"""预先保存两种题集分支的 B/C 抽样；不读取模型表现或生成医学标签。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_mdt.data import load_questions
from evidence_mdt.interventions import subset25
from evidence_mdt.query_manifest import public_question_sha256

SEED = 20261008  # 必须与运行器的默认分层抽样种子一致。
BRANCHES = {"medqa_branch": "MedQA.jsonl", "medxpertqa_branch": "MedXpertQA.jsonl"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def branch_record(pilot: Path, name: str, clinical_file: str, seed: int = SEED) -> dict:
    clinical_path = pilot / clinical_file
    pqa_path = pilot / "PubMedQA.jsonl"
    clinical = load_questions(clinical_path)
    pqa = load_questions(pqa_path)
    if any(q.dataset.lower().startswith("pubmedqa") for q in clinical):
        raise ValueError(f"{name}: clinical file contains PubMedQA")
    if any(not q.dataset.lower().startswith("pubmedqa") for q in pqa):
        raise ValueError(f"{name}: PubMedQA file contains another dataset")
    questions = clinical + pqa  # 顺序与正式 CLI 合并题集一致。
    if len({q.question_id for q in questions}) != len(questions):
        raise ValueError(f"{name}: duplicate question ID")
    selected = subset25(questions, seed)
    if len(selected) != 25 or sum(q.dataset.lower().startswith("pubmedqa") for q in selected) != 10:
        raise AssertionError("Selection must contain 15 clinical and 10 PubMedQA questions")
    return {
        "schema_version": 1,
        "branch": name,
        "branch_selected": False,
        "selection_seed": seed,
        "selection_algorithm": "evidence_mdt.interventions.subset25",
        "source_files": {
            "clinical": {"name": clinical_file, "sha256": sha256(clinical_path), "n": len(clinical)},
            "pubmedqa": {"name": pqa_path.name, "sha256": sha256(pqa_path), "n": len(pqa)},
        },
        "selected_question_count": 25,
        "selected_clinical_count": 15,
        "selected_pubmedqa_count": 10,
        "selected": [
            {"question_id": q.question_id, "dataset": q.dataset,
             "public_question_sha256": public_question_sha256(q)}
            for q in selected
        ],
        "note": ("预先冻结的两个候选分支之一；此文件不决定难度门槛、不含金标准答案，"
                 "也不代表医学扰动已核验。"),
    }


def prepare(pilot: Path, output: Path, seed: int = SEED) -> dict[str, dict]:
    records = {name: branch_record(pilot, name, clinical_file, seed)
               for name, clinical_file in BRANCHES.items()}
    # 先比较所有目标，再写任何文件，避免两个分支只有一个完成。
    for name, record in records.items():
        target = output / f"{name}.json"
        if target.exists() and json.loads(target.read_text(encoding="utf-8")) != record:
            raise ValueError(f"Existing branch selection differs: {target}")
    output.mkdir(parents=True, exist_ok=True)
    for name, record in records.items():
        target = output / f"{name}.json"
        if not target.exists():
            target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=ROOT / "data/pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "data/intervention_selection_oct10")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    records = prepare(args.pilot, args.output, args.seed)
    print(json.dumps({name: {"count": len(row["selected"]),
                             "clinical": row["selected_clinical_count"],
                             "pubmedqa": row["selected_pubmedqa_count"]}
                      for name, row in records.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
