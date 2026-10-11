# 公开医学摘要快照准备

`scripts/prepare_question_corpus.py` 从固定的 JSONL 题目文件生成 PubMed 摘要语料。它只把 `Question.public()` 交给查询模型；金标准答案、PubMedQA 长答案和源数据的隐藏字段不进入请求。脚本不判断医学答案，也不按答案筛选文献。

示例（会调用 DeepSeek 与 PubMed，执行前自行核对预算和 API 配置）：

```powershell
python scripts/prepare_question_corpus.py --questions data/pilot/pilot50.jsonl --output data/pilot/pubmed_snapshot --max-records 8 --max-calls 100 --max-tokens 100000 --model-backend deepseek
```

`--model-backend mock` 只供离线流程验证。mock 仍会访问 PubMed；完全离线测试在 `tests/test_corpus_preparation.py` 中用 HTTP fixture 代替下载。原五题 `prepare_smoke_corpus.py` 和其产物保持独立。

输出的 `questions.jsonl` 仅有公开题面，`corpus.jsonl` 是按 `chunk_id` 合并的摘要片段，可交给后续 MedCPT 索引流程。`preparation_manifest.json` 保存输入、代码和模型配置哈希，每题的模型请求、原始响应、已知 token 用量与失败，以及每个查询的状态、来源查询和片段 ID。`chunk_sources` 记录一个片段由哪些题和查询取得。每条查询还保存 `*.snapshot.json`（检索 URL、PMID、文章元数据及时间）、`*.raw.xml` 和 `*.evidence.jsonl`；`summary.json` 给出完成数、失败数和语料哈希。没有命中时无 XML 文件。

PubMedQA 入选题的目标 PMID 在整份 union 语料中排除，包括由其他题查询取得的相同 PMID。管线只使用摘要，不导入全文结论。下载失败保留 `fetch_failed`，下次运行会重试该查询；已成功生成的模型查询直接复用。模型失败保留 `query_failed` 及可获知的用量，下次运行允许重试，并累计调用预算。

再次运行必须使用相同输入文件内容、代码、模型配置和 `--max-records`；不一致时须指定新输出目录。已完成快照直接复用，不覆盖。`--max-calls` 与 `--max-tokens` 是整份快照的累计上限，可在续跑时调整。token 只在服务端响应或错误报告用量后才可获知，因此 `--max-tokens` 是**调用前软阈值**，单次调用可能使实际已知用量越过该值；未报告用量的失败调用记为未知、已知 token 记零。下载调用不计入模型调用预算。预算停止后的状态是 `budget_stopped`，可提高预算续跑。为节制 NCBI 请求，每条下载查询后间隔一秒。
