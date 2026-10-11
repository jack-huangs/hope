# 宽查询补充语料审计（2026-10-09）

`data/pilot_broad_oct09` 从完整的 `data/pilot_relevance_oct09` 派生。触发规则在运行前固定为“该题原两条查询按 `chunk_id` 合并后，排除所有 PubMedQA 目标 PMID，独立摘要少于 3 条”。该规则选中 12/50 题：MedQA `516`、`944`、`314`、`556`、`878`；PubMedQA `12419743`、`7497757`、`22537902`、`26209118`、`26348845`、`21823940`、`19468282`。题目选择与查询生成均不读取金标准、模型答题结果或 B/C 干预效果。B/C 仍共用预先固定的 25 题。

每道触发题的宽查询模型请求只含 `question_id`、`dataset`、`stem`，不含选项、PubMedQA 背景摘要、金标准或旧检索结果。模型须给出两条短 PubMed Boolean 查询，以 `AND` 连接核心概念、可用 `OR` 连接同义词。生成的原始请求、响应、版本、已知 token、开始时间和耗时均留在清单中；下载仍使用 `sort=relevance`、每条最多 8 个 PMID。原父语料保持不变，新增结果与父语料按 `chunk_id` 合并。

执行 `python scripts/audit_broad_corpus.py data/pilot_broad_oct09` 的只读检查返回 `issues: []`。审计先核对父 relevance 快照，再验证父清单及语料哈希、统一触发集、12 个仅题干模型请求与实际 wire 请求、生成查询与原始响应、全部 HTTP 尝试的原文哈希、ESearch 原始 JSON、EFetch XML、文章元数据、摘要文本及其哈希、全 20 个 PubMedQA 目标 PMID 的排除、union 片段和查询溯源。

| 项目 | 结果 |
|---|---:|
| 触发题 | 12 / 50 |
| 新模型调用／已知 token | 12 / 5,230 |
| 新宽查询 | 24；22 条有 PMID，2 条零 PMID |
| 逐查询保留摘要（含跨查询重复） | 129 |
| 排除目标 PMID 摘要 | 7 |
| 父语料／新增唯一片段／union | 434 / 122 / 556 |

这些是文件完整性、可见输入和来源审计，不代表 12 题的 top6 已具备直接相关医学依据。最终检索质量须以固定的短查询清单 `data/pilot_relevance_oct09/initial_queries.jsonl` 在 556 片段 union 上重新检验，并人工盲看题面、top6 标题和摘要；不得以重排分数阈值代替内容判断。
