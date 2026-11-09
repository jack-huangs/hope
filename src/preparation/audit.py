"""离线检查材料哈希、冻结证据及实验预检；不重新检索或调用模型。"""
from dataclasses import asdict
import json
from pathlib import Path

from evidence_mdt.data import read_json_records
from evidence_mdt.intervention_preflight import prepare_interventions
from evidence_mdt.interventions import B_LEGACY_CONDITIONS
from evidence_mdt.query_manifest import initial_queries
from evidence_mdt.retrieval import INITIAL_EVIDENCE_LIMIT
from evidence_mdt.schema import Evidence
from .materials import B_TARGET_SELECTION_POLICY, b_selection, validate_b_annotations
from .state import ConfigurationError, digest


def verify_preparation(folder: Path):
    from . import load_questions
    folder = folder.resolve()
    state = json.loads((folder / "preparation_manifest.json").read_bytes())
    config = state["config"]
    if state.get("status") != "complete":
        raise ConfigurationError("Preparation is not complete: " + str(folder))
    if config.get("initial_evidence_limit") != INITIAL_EVIDENCE_LIMIT:
        raise ConfigurationError("Initial evidence policy differs; prepare a new snapshot")
    required = {"questions.jsonl", "initial_queries.jsonl", "corpus.jsonl",
                "initial_top10.jsonl", "index_manifest.json"}
    mode = config["interventions"]
    if mode != "none":
        required.add("intervention_spec.jsonl")
    artifacts = state.get("artifacts", {})
    if not required <= artifacts.keys():
        raise ConfigurationError("Missing required frozen artifacts")
    for name, expected in artifacts.items():
        path = folder / name
        if Path(name).name != name or not path.is_file() or digest(path.read_bytes()) != expected:
            raise ConfigurationError("Artifact hash mismatch: " + name)
    questions = load_questions(folder / "questions.jsonl")
    if digest([asdict(q) for q in questions]) != config["questions"]:
        raise ConfigurationError("Question content differs from configuration")
    ids = [q.question_id for q in questions]
    if [r["question_id"] for r in state["records"]] != ids or any(
            r["status"] != "ready" for r in state["records"]):
        raise ConfigurationError("Preparation records differ from questions")
    queries = initial_queries(questions, folder / "initial_queries.jsonl")
    corpus = read_json_records(folder / "corpus.jsonl")
    by_id = {e["chunk_id"]: asdict(Evidence(**e)) for e in corpus}
    if len(by_id) != len(corpus):
        raise ConfigurationError("Duplicate corpus chunk IDs")
    index = json.loads((folder / "index_manifest.json").read_bytes())
    if index["sha256"] != digest((folder / "corpus.jsonl").read_bytes()):
        raise ConfigurationError("Index corpus hash mismatch")
    if index.get("backend") == "medcpt":
        vectors = folder / "index_manifest.vectors.pt"
        if vectors.name not in artifacts or digest(vectors.read_bytes()) != index["vectors_sha256"]:
            raise ConfigurationError("Index vector hash mismatch")
    frozen = read_json_records(folder / "initial_top10.jsonl")
    if [r["question_id"] for r in frozen] != ids:
        raise ConfigurationError("Frozen evidence question order differs")
    lookup, counts = {}, []
    for q, row in zip(questions, frozen):
        evidence = row["evidence"]
        if not 0 < len(evidence) <= INITIAL_EVIDENCE_LIMIT or len({e["chunk_id"] for e in evidence}) != len(evidence):
            raise ConfigurationError("Invalid frozen evidence count: " + q.question_id)
        if any(by_id.get(e["chunk_id"]) != e for e in evidence):
            raise ConfigurationError("Frozen evidence differs from corpus: " + q.question_id)
        query = queries[q.question_id]
        if query in lookup and lookup[query][0] != evidence:
            raise ConfigurationError("Identical queries have different frozen evidence")
        lookup[query] = evidence, row.get("trace") or {}
        counts.append(len(evidence))

    class FrozenRetriever:
        def search(self, query, **kwargs):
            evidence, self.last_trace = lookup[query]
            return [Evidence(**e) for e in evidence]

    if mode != "none":
        *_, intervention_manifest = prepare_interventions(
            questions, FrozenRetriever(), mode, folder / "intervention_spec.jsonl",
            1, tuple(config.get("b_conditions", B_LEGACY_CONDITIONS)), queries, config["seed"],
            allow_legacy_d=mode == "d" and config.get("d_evidence_count", 1) == 1)
    result = {"status": "passed", "mode": mode, "questions": len(questions),
              "corpus_chunks": len(corpus), "initial_evidence_counts": counts,
              "artifact_hashes_verified": True, "intervention_preflight_passed": mode != "none",
              "note": "Offline structural/provenance checks only; no ranking replay or medical verification."}
    if mode == "b":
        specs = read_json_records(folder / "intervention_spec.jsonl")
        policy = config.get("b_target_selection_policy")
        if policy is not None:
            if policy != B_TARGET_SELECTION_POLICY:
                raise ConfigurationError("Unknown B target selection policy")
            specs_by_question = {spec["question_id"]: spec for spec in specs}
            for q, row in zip(questions, frozen):
                spec = specs_by_question[q.question_id]
                try:
                    annotations = validate_b_annotations(
                        {"annotations": spec.get("b_evidence_annotations")},
                        [e["chunk_id"] for e in row["evidence"]], q.options)
                    expected = b_selection(annotations, config["seed"], q.question_id)
                    if any(key not in spec or spec[key] != value for key, value in expected.items()):
                        raise ValueError("B selection differs from annotated candidate pool and seed")
                except ValueError as exc:
                    raise ConfigurationError(f"Invalid B selection for {q.question_id}: {exc}") from exc
            result["b_target_selection_policy"] = policy
            result["all_null_fallback"] = sum(r["b_target_selection"] == "all_null_fallback" for r in specs)
        result["annotated"] = sum(r["target_answer"] is not None for r in specs)
        result["unannotated"] = sum(r["target_answer"] is None for r in specs)
    if mode == "d":
        result["d_evidence_count"] = intervention_manifest["d_evidence_count"]
    return result


def verify_batch(folder: Path):
    """检查整批 B/C/D 及其共用来源，无需加载检索模型。"""
    from . import load_questions
    folder = folder.resolve()
    state = json.loads((folder / "batch_manifest.json").read_bytes())
    config = state["config"]
    if state.get("status") != "complete" or state.get("fingerprint") != digest(config):
        raise ConfigurationError("Batch is incomplete or its configuration hash differs")
    questions = load_questions(folder / "questions.jsonl")
    if digest([asdict(q) for q in questions]) != config["questions"]:
        raise ConfigurationError("Batch questions differ from configuration")
    origin = config["source"]
    source = Path(origin["path"]) if origin else folder / "shared"
    if origin and digest((source / "preparation_manifest.json").read_bytes()) != origin["manifest_sha256"]:
        raise ConfigurationError("Batch source manifest hash differs")
    result = {"status": "passed", "questions": len(questions),
              "shared": verify_preparation(source), "modes": {}}
    for name in ("questions.jsonl", "corpus.jsonl", "initial_queries.jsonl", "initial_top10.jsonl"):
        if name == "questions.jsonl" and load_questions(source / name) != questions:
            raise ConfigurationError("Batch source questions differ")
        for mode in config["modes"]:
            if (source / name).read_bytes() != (folder / mode / name).read_bytes():
                raise ConfigurationError("Batch shared artifacts differ: " + mode + "/" + name)
    for mode in config["modes"]:
        checked = verify_preparation(folder / mode)
        if checked["mode"] != mode:
            raise ConfigurationError("Batch intervention mode differs: " + mode)
        result["modes"][mode] = checked
    return result
