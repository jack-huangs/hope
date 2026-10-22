"""只进行离线切片、索引和50题检索检查；不调用答题API。"""
import json
import subprocess
import sys
import random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import prepare_relevance_corpus as base
import audit_corpus_snapshot
import audit_broad_corpus
import audit_chunk_corpus
import chunk_corpus
import audit_initial_retrieval
from evidence_mdt.cli import index_command
from evidence_mdt.data import load_questions, normalize_question


def selection_audit():
    pilot = ROOT / "data/pilot"
    dataset = ROOT / "data/medxpert_pilot_oct12"
    old_manifest = json.loads((pilot / "selection_manifest.json").read_bytes())
    raw = (pilot / "MedXpertQA.raw").read_bytes()
    assert base.sha(raw) == old_manifest["sources"]["MedXpertQA"]["raw_sha256"]
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    lookup = {str(row["id"]): row for row in records}
    old = load_questions(pilot / "MedXpertQA.jsonl")
    assert [q.question_id for q in old] == old_manifest["sources"]["MedXpertQA"]["selected_source_ids"]
    for question in old:
        assert asdict(question) == asdict(normalize_question(lookup[question.question_id], "MedXpertQA", 0))
    selection = json.loads((dataset / "selection_manifest.json").read_bytes())
    pool = [qid for qid in lookup if qid not in {q.question_id for q in old}]
    assert selection["added_ids"] == random.Random(selection["seed"]).sample(pool, 10)
    current = load_questions(dataset / "MedXpertQA.jsonl")
    assert len(current) == 30 and len({q.question_id for q in current}) == 30
    assert [q.question_id for q in current] == selection["retained_ids"] + selection["added_ids"]
    for question in current:
        assert asdict(question) == asdict(normalize_question(lookup[question.question_id], "MedXpertQA", 0))
    assert (dataset / "MedXpertQA.jsonl").read_bytes().startswith((pilot / "MedXpertQA.jsonl").read_bytes())
    assert (dataset / "PubMedQA.jsonl").read_bytes() == (pilot / "PubMedQA.jsonl").read_bytes()
    return {"status": "passed", "official_raw_matches_old_manifest": True,
            "retained20_and_added10_match_official_normalization": True,
            "id_only_sample_recomputed": True, "pubmedqa_byte_preserved": True,
            "official_raw_sha256": base.sha(raw), "old_manifest_sha256": base.sha((pilot / "selection_manifest.json").read_bytes())}


def record_failed_response_hashes(folder):
    raw = (folder / "preparation_manifest.json").read_bytes()
    manifest = json.loads(raw)
    missing = {attempt["raw_file"] for row in manifest["records"] for entry in row["snapshots"]
               for attempt in entry.get("attempts", []) if attempt.get("raw_file") and
               attempt.get("status") == "failed" and not attempt.get("raw_sha256")}
    if missing:
        sidecar = {"manifest_sha256": base.sha(raw),
                   "note": "Failed HTTP bodies preserved during download; missing attempt hashes recorded during acceptance, not at download time.",
                   "files": {name: base.sha((folder / name).read_bytes()) for name in sorted(missing)}}
        path = folder / "failed_response_hashes.json"
        data = json.dumps(sidecar, ensure_ascii=False, indent=2).encode("utf-8")
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Failure-body hash sidecar differs")
        if not path.exists():
            path.write_bytes(data)


def main():
    reports = ROOT / "runs/medxpert_preparation_oct12"
    base.save_json(reports / "selection_audit.json", selection_audit())
    relevance = ROOT / "data/medxpert_relevance_oct12_v2"
    broad = ROOT / "data/medxpert_broad_oct12_v3"
    chunks = ROOT / "data/medxpert_chunks_oct12"
    models = ROOT / "indexes/model_download_manifest.json"
    record_failed_response_hashes(broad)
    for name, result in (("relevance_audit", audit_corpus_snapshot.audit(relevance)),
                         ("broad_audit", audit_broad_corpus.audit(broad))):
        base.save_json(reports / (name + ".json"), result)
        if result["issues"] or result["status"] != "complete":
            raise RuntimeError(name + " failed")
        print(name + " passed", flush=True)
    from transformers import AutoTokenizer
    model = json.loads(models.read_bytes())["models"]["ncbi/MedCPT-Article-Encoder"]
    tokenizer = AutoTokenizer.from_pretrained(model["snapshot_path"], use_fast=True, local_files_only=True)
    child = chunk_corpus.prepare(broad, chunks, tokenizer, model["revision"], limit=399)
    checked = audit_chunk_corpus.audit(broad, chunks, models)
    base.save_json(reports / "chunk_audit.json", checked)
    if checked["issues"]:
        raise RuntimeError("Chunk audit failed")
    print("chunk audit passed: " + str(child["corpus_chunks"]), flush=True)
    index = ROOT / "indexes/medxpert_chunks_oct12.json"
    if index.exists():
        existing = json.loads(index.read_bytes())
        if existing["sha256"] != child["corpus_sha256"] or base.sha(Path(existing["vectors"]).read_bytes()) != existing["vectors_sha256"]:
            raise ValueError("Existing index differs")
    else:
        index_command(SimpleNamespace(corpus=chunks / "corpus.jsonl", backend="medcpt", output=index))
    summary = audit_initial_retrieval.audit(chunks, ROOT / "runs/medxpert_retrieval_qa_oct12", "medcpt", index,
                                             relevance / "initial_queries.jsonl")
    if summary["query_encoder_over_64_ids"] or summary["empty_top6_ids"]:
        raise RuntimeError("Initial retrieval QA failed")
    queries = json.loads((ROOT / "data/medxpert_queries_oct12/preparation_manifest.json").read_bytes())
    supplements = json.loads((broad / "preparation_manifest.json").read_bytes())
    triggered_mx = sum(r["dataset"] == "MedXpertQA" for r in supplements["selected_questions"])
    triggered_pqa = sum(r["dataset"].lower().startswith("pubmedqa") for r in supplements["selected_questions"])
    costs = {"new_query_calls": queries["actual_new_model_calls"],
             "new_query_tokens": queries["known_tokens"] - queries["reused_known_tokens"],
             "historical_reused_query_calls": queries["reused_model_calls"],
             "historical_reused_query_tokens": queries["reused_known_tokens"],
             "new_broad_calls": supplements["actual_new_model_calls"],
             "new_broad_tokens": supplements["known_tokens"] - supplements["reused_known_tokens"],
             "historical_reused_broad_calls": supplements["reused_model_calls"],
             "historical_reused_broad_tokens": supplements["reused_known_tokens"],
             "fee_currency_amount": None, "fee_note": "保存服务端token用量，未读取账单；历史调用不计入本轮新增费用。"}
    base.save_json(reports / "costs.json", costs)
    manifest = {"status": "complete", "questions": 50, "medxpertqa": 30, "pubmedqa": 20,
                "corpus_parent_abstracts": supplements["corpus_chunks"], "corpus_chunks": child["corpus_chunks"],
                "broad_trigger_counts": {"MedXpertQA": triggered_mx, "PubMedQA": triggered_pqa},
                "costs": costs, "retrieval_qa": summary, "completed_at": base.now(),
                "answer_experiments_started": False,
                "artifacts": {"scoring_questions": str(ROOT / "data/medxpert_pilot_oct12/pilot50.jsonl"),
                    "corpus": str(chunks / "corpus.jsonl"), "index": str(index),
                    "query_manifest": str(relevance / "initial_queries.jsonl")}}
    base.save_json(reports / "acceptance.json", manifest)
    report = f"""# MedXpertQA检索材料准备报告

本阶段仅准备材料，未运行普通组或A/B/C答题。固定题集30道MedXpertQA Text加原20道PubMedQA；旧20道MX保留，按seed20261012从其余官方ID无放回追加10道。PubMedQA原文件逐字保留，评分题集保留金标，查询公开副本排除金标和长答案。

初始查询只使用病例和问句，确定性剥除MedXpertQA题干尾部候选选项段；未读取正确答案生成查询或挑选文献。原20道PubMedQA查询及relevance/broad快照按题面哈希核对复用，XML和请求原始体按字节保存并记录原目录及清单哈希。新题直接使用PubMed sort=relevance，每条最多8篇；初始可用摘要少于3篇时统一生成2条宽查询。全库排除20个目标PMID。

宽查询固定规则触发{triggered_mx + triggered_pqa}题：MedXpertQA {triggered_mx}题、PubMedQA {triggered_pqa}题。查询语法失败允许一次有界重试，全部失败调用和token计入费用，最终查询须对应最后成功响应；不根据正确答案重写查询。

四题两次宽查询均返回不含AND的普通字符串，原v2保留失败。新broad_v3对这四题各做一次格式修复，输入只含最后生成的两个查询字符串；每个修复输出按对应query逐词Counter验证除AND/引号/括号外不得增删词，原顺序一一对应。失败与修复调用全部计费留痕；其余27个已完成补充题未重做。

一条历史curl HTTP400失败原始体已保存但原attempt未写hash，验收时另写failed_response_hashes.json并核对清单hash和原文件；该失败体hash是验收时记录，不能声称下载当时已完整登记。成功XML、ESearch、语料hash仍逐一按原清单核对。

首遍urllib出现截断正文和连接断开，保留失败尝试。续跑发现查询准备函数重复写完成时间，导致父清单hash变化；配置错误未发网络。旧目录完整保留，迁移到relevance_v2目录，核对查询、公开题面与全部检索参数一致，仅更新父清单hash及迁移元数据；curl恢复每个请求记录传输和时间并验证JSON/XML完整解析。迁移详情见migration.json。

Text-281原两查询组合69token，超过64限制；本地查询改为原第一条完整查询43token，因为第二条的空格词集完全包含于第一条。该词集检查不证明语义等价。原生成查询、PubMed请求、原组合清单均保留，修订hash和前后token在initial_queries.revision.json。

最终{supplements['corpus_chunks']}个父摘要，{child['corpus_chunks']}个连续不重叠片段，标题加分隔符正文最大{checked['max_tokens']}个token；各片保留parentID、原文区间与hash且可精确拼回原文。BM25与MedCPT各召回50，RRF60取30重排返回6。50题均返回6条，全部300条引用具有URL及hash；查询最长{summary['max_query_encoder_tokens']}token，超过64的题数为0。

新增初始查询{costs['new_query_calls']}次、{costs['new_query_tokens']}token；新增宽查询{costs['new_broad_calls']}次、{costs['new_broad_tokens']}token。复用历史调用初始{costs['historical_reused_query_calls']}次、宽查询{costs['historical_reused_broad_calls']}次，历史用量单列于costs.json，未重新收费。无法由本地记录确认实际货币账单。

检索缺口：top6无同题来源{summary['no_same_question_source_ids']}。技术检查通过只确认输入、来源和长度；不等于医学相关性通过。未用金标进行医学相关性选择。旧低覆盖PubMedQA题目26348845和21823940保留，具体缺口见初始/宽查询审计与逐题检索轨迹。B/C只冻结15MX+10PQA题号，未制作干预或开始运行。

未来运行的评分输入、语料、索引、短查询路径见acceptance.json。后续每条件repeats=1，并显式设置进程环境MDT_MAX_OUTPUT_TOKENS=2000、OMP_NUM_THREADS=4、MKL_NUM_THREADS=4；此报告不授权启动答题。
"""
    (reports / "preparation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
