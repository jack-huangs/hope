"""逐题核对后复制历史检索文件；保留字节、哈希和原调用归属。"""
import copy
import json
from pathlib import Path
import prepare_relevance_corpus as base


def reuse_rows(output, manifest, source, *, broad=False):
    source = Path(source).resolve()
    raw = (source / "preparation_manifest.json").read_bytes()
    parent = json.loads(raw)
    if parent["status"] != "complete":
        raise ValueError("Reuse source is incomplete")
    old = {r["question_id"]: r for r in parent["records"]}
    files = set()
    for row in manifest["records"]:
        previous = old.get(row["question_id"])
        if not row["dataset"].lower().startswith("pubmedqa") or previous is None:
            continue
        if row["public_question_sha256"] != previous["public_question_sha256"]:
            raise ValueError("Reuse public question differs")
        if broad and row["status"] == "not_triggered":
            if previous["status"] != "not_triggered":
                raise ValueError("Reuse broad trigger differs")
            continue
        if not broad and row["queries"] != previous["queries"]:
            raise ValueError("Reuse queries differ")
        if previous["status"] != "ready":
            raise ValueError("Reuse row is incomplete")
        row.update(copy.deepcopy(previous))
        row["reused_from"] = {"source": str(source), "manifest_sha256": base.sha(raw)}
        for entry in row["snapshots"]:
            files.update(entry[k] for k in ("snapshot_file", "evidence_file", "raw_xml_file") if entry.get(k))
            files.update(a["raw_file"] for a in entry.get("attempts", []) if a.get("raw_file"))
            snapshot = json.loads((source / entry["snapshot_file"]).read_bytes())
            if snapshot.get("raw_esearch_file"):
                files.add(snapshot["raw_esearch_file"])
    copied = {}
    for filename in sorted(files):
        data = (source / filename).read_bytes()
        destination = output / filename
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError("Reuse filename collision")
        destination.write_bytes(data)
        copied[filename] = base.sha(data)
    manifest["reuse_provenance"] = {"source": str(source), "manifest_sha256": base.sha(raw), "files_sha256": copied}
