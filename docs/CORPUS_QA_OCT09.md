# 50 题 PubMed 摘要快照审计（2026-10-09）

审计对象：`data/pilot_corpus_oct09`。执行 `python scripts/audit_corpus_snapshot.py data/pilot_corpus_oct09`；工具只读本地文件，不调用模型或 PubMed。当前 `preparation_manifest.json` 为 `complete`，含 50 题、100 条查询、50 次模型调用、27,204 个已知 token、473 个 union 片段；完整性检查返回 `issues: []`。

| 检查 | 结果 |
|---|---:|
| 输入及请求 | 50 个模型请求的题面与 `questions.jsonl` 及每题公开题面哈希一致；请求没有 `gold`、`LONG_ANSWER`、`final_decision` 或 `answer_idx` 字段 |
| 查询 | 每题 2 条，共 100 条；实际响应与清单查询相符；长度 23–112 字符 |
| 检索状态 | 83 条有 PMID，17 条零 PMID；无下载失败 |
| 排除规则 | 20 个 PubMedQA 目标 PMID 均不在 `corpus.jsonl` 或逐查询 evidence 中；20 次命中的目标摘要被排除。其余 6 个目标 PMID 未在本次 ESearch 命中，不代表规则未覆盖 |
| 文件与溯源 | 所有非空查询的 XML SHA-256 与清单相符；逐条摘要文本可从 XML 重建且内容哈希相符；metadata PMID 可在 XML 找到；union `chunk_id`、`chunk_sources` 与逐查询 evidence 一致；`corpus.jsonl` 哈希相符 |
| 数量 | ESearch 共返回 516 个 PMID；过滤、无摘要及跨查询重复后，逐查询保留 485 条摘要，union 为 473 条 |

每题最多 16 条逐查询结果。按题内唯一可用摘要数，10 题达到 16 条，另有 8 题为 0 条、4 题只有 1–2 条。零摘要题是 MedQA `516`、`944`、`314`、`878`，PubMedQA `12419743`、`26348845`、`21823940`、`19468282`。1–2 条题是 MedQA `556`，PubMedQA `7497757`、`22537902`、`26209118`。17 条零 PMID 查询中，MedQA 占 14 条，PubMedQA 占 3 条。零 PMID 是保存的 ESearch 响应，不是解析器丢弃；具体查询与返回数可通过审计脚本的 `per_question` 查看。

现有语料仅导入 PubMed 摘要，没有全文。PubMedQA 目标研究 PMID 在整份语料统一排除，因而目标文章自身的摘要结论未进入模型语料。此机械检查无法判定其他文章是否在语义上直接回答某题；医学内容相关性与潜在间接泄漏须另行盲审，不能把哈希检查解释为内容审定。

## 排序问题与重抓方案

全部 100 条已保存的 `search_url` 均未显式设置 `sort`。NCBI 说明 PubMed ESearch 默认是 most recent，而 Best Match 需 `sort=relevance`；[NCBI E-utilities 参数文档](https://www.ncbi.nlm.nih.gov/sites/books/NBK25499/)列出 PubMed 的 `relevance` 值，[NCBI API 说明](https://ncbiinsights.ncbi.nlm.nih.gov/2022/03/24/test-server-pubmed-api/)明确两种排序及默认值。因此这份快照应标为 **默认时间排序版**，保留作可回放的先期产物；正式 pilot 建议另建 relevance 版，再冻结用于三组方法比较。

最小实现为独立 `scripts/prepare_relevance_corpus.py`，不修改当前 `src` 或原快照：

1. 只读取原 `preparation_manifest.json` 和 `questions.jsonl`，检查原状态为 `complete`、50 题顺序/公开题面哈希一致、每题查询与模型原始响应一致。把原清单 SHA-256、100 条固定查询的 SHA-256、`sort=relevance`、`max_records=8`、新脚本代码哈希做成新快照指纹。**不创建模型对象，不重新调用模型**；新清单 `new_model_calls=0`，另存 `source_model_calls=50`、`source_known_tokens=27204` 作为来源成本。
2. 每条固定查询仅改变 ESearch 参数：`db=pubmed&term=<原查询>&retmax=8&retmode=json&sort=relevance`。保留 EFetch XML 原文、搜索 URL、排序参数、时间、PMID、文章元数据和哈希。抽取摘要的规则与原脚本相同；不下载全文，也不按金标或答案调整查询。若 ESearch 响应缺少 `esearchresult.idlist`，记录 `fetch_failed` 并重试，不把临时缺字段误记成零结果。
3. 每条查询持久化 `fetch_pending/ok/empty/fetch_failed` 与原始下载文件，再按全 20 个 PubMedQA 目标 PMID 排除，按 `chunk_id` 生成新的 `corpus.jsonl`、`chunk_sources` 和 `summary.json`。中断时重用已成功的查询和下载；同指纹完成快照直接复用，配置不同则拒绝覆盖。输出目录例如 `data/pilot_corpus_relevance_oct09`，绝不就地修改 `data/pilot_corpus_oct09`。
4. 对新目录运行 `scripts/audit_corpus_snapshot.py` 的同等检查，并额外要求每个 `search_url` 含 `sort=relevance`。比较两版每题可用摘要数、零结果数及 PMID 重叠率；这些是检索质量诊断，不依据模型答题结果择优。

按当前请求节奏，重抓最多约 100 次 ESearch 加至多 100 次 EFetch，模型调用为 0。网络请求尚未执行；relevance 版的命中量、实际成本和医学内容相关性目前未知。

现已实现 `scripts/prepare_relevance_corpus.py`：`python scripts/prepare_relevance_corpus.py --source data/pilot_corpus_oct09 --output data/pilot_corpus_relevance_oct09 --max-records 8 --retries 2`。每条查询完成后间隔一秒；正常查询的 ESearch 与 EFetch 连续发出，每秒通常不超过 NCBI 公共接口无 API key 的 3 次请求限制。重试采用 1 秒、2 秒退避；每次尝试的原始 ESearch JSON 或 EFetch XML 与错误日志独立编号保存，断点续跑延续编号，不覆盖旧尝试。若同一机器上另有 NCBI 请求并发，应降低本脚本请求频率或错开运行。零摘要题后续可预先制定固定规则的宽查询回退，但本轮重抓不自动放宽或改写任何查询。

## 逐题检索量

表内 `返回/保留` 分别是 ESearch PMID 数和排除目标 PMID、无摘要后的可用摘要数。

| 数据集 | 题号 | 查询 1 返回/保留 | 查询 2 返回/保留 | 题内唯一摘要 |
|---|---:|---:|---:|---:|
| MedQA | 0 | 8/8 | 8/8 | 16 |
| MedQA | 77 | 8/6 | 8/8 | 14 |
| MedQA | 474 | 8/8 | 8/8 | 16 |
| MedQA | 654 | 8/8 | 1/1 | 9 |
| MedQA | 620 | 3/3 | 8/8 | 11 |
| MedQA | 516 | 0/0 | 0/0 | 0 |
| MedQA | 944 | 0/0 | 0/0 | 0 |
| MedQA | 153 | 8/8 | 8/8 | 13 |
| MedQA | 220 | 8/8 | 8/8 | 16 |
| MedQA | 803 | 8/8 | 8/8 | 15 |
| MedQA | 393 | 8/8 | 8/8 | 16 |
| MedQA | 314 | 0/0 | 0/0 | 0 |
| MedQA | 520 | 8/7 | 8/8 | 14 |
| MedQA | 105 | 0/0 | 8/8 | 8 |
| MedQA | 904 | 8/6 | 8/8 | 14 |
| MedQA | 959 | 0/0 | 4/4 | 4 |
| MedQA | 431 | 8/7 | 8/7 | 14 |
| MedQA | 794 | 8/8 | 8/8 | 14 |
| MedQA | 568 | 8/8 | 8/8 | 15 |
| MedQA | 168 | 8/8 | 0/0 | 8 |
| MedQA | 589 | 8/8 | 8/8 | 15 |
| MedQA | 103 | 4/4 | 0/0 | 4 |
| MedQA | 299 | 8/7 | 0/0 | 7 |
| MedQA | 777 | 8/8 | 8/7 | 15 |
| MedQA | 522 | 8/8 | 8/8 | 16 |
| MedQA | 758 | 8/6 | 2/2 | 8 |
| MedQA | 556 | 2/2 | 0/0 | 2 |
| MedQA | 878 | 0/0 | 0/0 | 0 |
| MedQA | 875 | 8/8 | 8/8 | 16 |
| MedQA | 176 | 8/8 | 8/8 | 16 |
| PubMedQA | 21190419 | 8/7 | 7/6 | 13 |
| PubMedQA | 28143468 | 8/8 | 8/8 | 16 |
| PubMedQA | 20073599 | 8/8 | 1/0 | 8 |
| PubMedQA | 11555508 | 3/2 | 8/8 | 10 |
| PubMedQA | 27581329 | 8/8 | 6/5 | 12 |
| PubMedQA | 12419743 | 0/0 | 0/0 | 0 |
| PubMedQA | 25588461 | 8/8 | 8/8 | 16 |
| PubMedQA | 7497757 | 2/1 | 2/1 | 2 |
| PubMedQA | 10375486 | 4/3 | 2/2 | 5 |
| PubMedQA | 10732884 | 8/8 | 8/8 | 15 |
| PubMedQA | 22537902 | 1/0 | 2/1 | 1 |
| PubMedQA | 26209118 | 2/1 | 1/0 | 1 |
| PubMedQA | 26348845 | 1/0 | 1/0 | 0 |
| PubMedQA | 21823940 | 1/0 | 0/0 | 0 |
| PubMedQA | 19468282 | 1/0 | 1/0 | 0 |
| PubMedQA | 18955431 | 8/8 | 1/0 | 8 |
| PubMedQA | 19155657 | 8/8 | 8/8 | 16 |
| PubMedQA | 25274085 | 2/1 | 8/8 | 9 |
| PubMedQA | 23949294 | 8/8 | 8/8 | 15 |
| PubMedQA | 2503176 | 3/2 | 8/8 | 10 |
