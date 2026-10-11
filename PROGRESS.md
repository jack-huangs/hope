# 项目执行记录

更新日期：2026-10-09。总体研究约定以 `EXPERIMENT_PLAN.md` 为准；本文件记录实际完成的工作，方便额度中断后接续。

## 已完成

- Miniconda 环境：`med-consensus`，Python 3.12；依赖版本见 `requirements-lock.txt`。
- 已固定抽样 30 道 MedQA、20 道 PubMedQA；另存 20 道 MedXpertQA 备用题。来源、抽样框和哈希在 `data/pilot/selection_manifest.json`。
- 已下载三个 MedCPT 模型，模型版本及本机位置在 `indexes/model_download_manifest.json`。
- 真实 BM25＋MedCPT 向量＋重排已经通过检索测试；这不等于问答实验已经完成。
- 五题技术试跑的数据已经准备好：`data/smoke/questions.jsonl` 和 `data/smoke/corpus.jsonl`。语料有 63 个摘要片段，保存了十次 PubMed 查询及原始 XML。生成查询消耗 5 次 DeepSeek 调用、3,303 个已报告 token。
- DeepSeek 密钥已放在 Git 忽略的 `.env`；`.env.example` 保留空白模板。Qwen 暂未配置有效值。
- 模拟模型已覆盖四专家、四轮、三种方法和 A/B/C 流程；模拟输出不作为医学结果。

## 正在进行

首次真实试跑的格式问题已修复，失败原始记录仍保留在 `runs/deepseek_smoke_oct09/`（2 次调用、5,204 个已报告 token、无最终结果）。格式修复最多一次，不允许借修复任意改答案或概率。语义比较已改为判断条件的含义，并采用候选分批与簇内完整检查，避免无必要的全量两两比较。

新的五题三组试跑在 `runs/deepseek_smoke_oct09_v2/`，现已 **15/15 组次完成**，共有 561 次调用、2,534,970 个已报告 token，当前版本未出现格式校验错误。第 5 题曾触发 200 万 token 预算停止；将执行上限提高至 400 万后成功接续，旧失败尝试与费用仍保留。预算迁移仅改变执行限制，旧配置与实验指纹不变，审计记录在 `execution_migrations.jsonl` 和 `budget_limits.jsonl`。全部成功结果的 320 次发言、16,853 条图边通过日志完整性检查。最新全套测试为 **82 项通过**；涵盖 B/C 预检、短查询、缓存独立重复隔离、难度门槛和引用回溯。

烟测中 5 道题三组均答对，未观察到共同错误，样本量不足以支持机制结论。补检索实际执行 2 次：一次未加入新片段，另一次加入 4 个片段并完成四专家独立复核。详细结果、成本与限制见 `smoke_acceptance.md`、`report.json`、`audit_run.json` 和 `analysis/common_errors.md`。本轮仅 1 次重复，不能代替计划中每条件 2 次的先导实验。源代码、计划及依赖清单保存在该目录的 `software_snapshot/`，不含密钥。

50 题 PubMed 初始语料 `data/pilot_corpus_oct09/` 已完成：100 条固定查询、473 个去重摘要；生成查询用了 50 次 DeepSeek 调用、27,204 个已报告 token。源文件哈希、查询谱系、XML 和 PubMedQA 目标 PMID 排除检查通过。原快照使用 ESearch 默认排序，另建的 `data/pilot_relevance_oct09/` 已完成，复用全部原查询并显式指定 `sort=relevance`，共 434 个去重摘要，新增模型调用为零。索引为 `indexes/pilot_relevance_oct09.json`。

早期曾直接用完整题目描述检索文献，并检查了全部 50 道题，结果保存在 `runs/pilot_relevance_retrieval_qa_oct09/`。检查发现：把查询文字转成向量的模型（Query Encoder）最多接收 64 个 token（模型处理文字的计数单位），而 50 道题全部超过这个长度，输入会被截断，可能丢失关键线索。另外，有 8 道题自己的 PubMed 查询没有带回可用摘要；从整个文献库选出的前 6 个片段也多数不能直接帮助答题。**因此，当时没有启动正式预筛（30 道 MedQA 各运行两次，共 60 个团队结果）。**

为解决这些问题，计划中已改为：把每道题之前生成的两条简短检索词拼起来，代替完整题目描述。50 道题中最长的组合只有 39 个 token，不会超过模型限制；这些组合已保存为固定版本，清单见 `data/pilot_relevance_oct09/initial_queries.jsonl`。对于自己的查询带回可用摘要少于 3 篇的题目，统一只根据题干生成最多两条范围更宽的检索词，再补搜文献，并把扩充后的文献库另存为一份固定快照。补搜的依据是文献数量不足，不是看到模型答错后再挑选能让它答对的证据。

补足快照 `data/pilot_broad_oct09/` 已完成并通过审计：固定规则触发 12 题、24 条查询，新增 12 次调用、5,230 个 token；原 434 篇加上 122 篇新摘要，最终 **556 篇**。7 次命中的 PubMedQA 目标摘要被排除，所有 20 个目标 PMID 均不在最终语料。PubMedQA `26348845`、`21823940` 没有取得额外可用摘要，保留题目和不足状态。原 434 语料的短查询检查也已完成，50 题均未超出 64 token；556 语料的最终索引和相关性检查正在进行。

发现两条会议汇编摘要达 81,020 和 20,492 字符，因此实际先导采用有长度上限的派生语料 `data/pilot_chunks_oct09/`：556 个父摘要中 136 个需要切分，最终 **782 个片段**，标题＋分隔符＋正文最大 399 token。所有片段连续、不重叠，可逐字拼回原文；保留真实 PMID／文档 ID、字符区间与哈希，不把切片视为独立研究。原快照全部保留。分片语料已完成独立审计，并已成功生成 MedCPT 索引 `indexes/pilot_chunks_oct09.json`（同时生成向量文件）；该本地步骤未调用 DeepSeek/Qwen API。最终 50 题检索检查已完成并记录在 `runs/pilot_chunks_retrieval_qa_oct10/`：50/50 题均返回 6 条结果，300/300 条引用有 URL 和内容哈希，短查询最长 39 个 Query Encoder token、37 个重排 token；仅 PubMedQA `26348845`、`21823940` 没有同题扩展来源，且其 top-6 相关性仍需人工复核。

## 接续顺序

1. 五题烟测已完成；审查新加的共同错误分析及 B/C 预检代码。
2. 完成相关性语料的初始检索质量检查，必要时按公开题面和固定规则补充语料；小型五题语料不直接充当完整先导语料。
3. 跑 30 道 MedQA 的普通组两次独立运行，完整取得 60 个结果后执行 54/60 的难度切换规则。
4. 开展 A；按预先固定的题目子集和干预素材开展 B/C。
5. 生成先导报告，再设计使用新题的确认实验。

用户已确认 **B、C 共用同一批 25 题：15 道当前临床问答＋10 道 PubMedQA**。B 首轮为 `base`、`literal3`、`paraphrase3` 三个条件，5 次呈现留作可选扩展。计划和代码均已同步并通过审查。实际预筛目录为 `runs/medqa_screen_chunks_oct09/`，源码与计划已存入其中的 `software_snapshot/`。早先的 `runs/medqa_screen_oct09/` 仅存分片前的软件快照，未运行问答 API，不应误计为失败实验。

后续协作模型约定：已有子代理保持原模型；新建且确实需要委派的简单子任务使用 GPT-5.6-terra medium。已存在的 `implementation_oct09`、`corpus_oct09`、`natural_analysis_oct09` 继续原 GPT-6 Sol medium，没有中途切换。

### 查看已完成的试跑

烟测已完成，不需再次调用 API。代码与计划后续有修改时，旧实验结果继续保留，不能在旧输出目录混跑新配置。

```powershell
$py = 'C:\Users\Administrator\miniconda3\envs\med-consensus\python.exe'
& $py scripts/summarize_smoke.py runs/deepseek_smoke_oct09_v2 --preparation-manifest data/smoke/preparation_manifest.json
& $py scripts/audit_run.py runs/deepseek_smoke_oct09_v2
```

真实实验评分使用 `data/pilot/` 中保留金标的标准化题目；语料目录中的 `questions.jsonl` 是去掉金标的公开查询输入副本，不能用它替代评分题集。

## 查阅运行记录

- `config.json`：这一批的模型参数、数据与代码哈希。
- `assignments.jsonl`：开始前登记的全部任务，尚未完成的任务也在其中。
- `calls.jsonl`：实际请求、返回、耗时和用量，不含认证请求头。
- `events.jsonl`：证据、专家发言、语义簇和引用关系；按 `attempt_id` 区分重试。
- `results.jsonl`：成功完成的团队结果；`failures.jsonl`：失败记录。
- `report.json`：分数据集统计的结果、配对分析及失败分母。

相同代码与配置续跑可跳过成功任务；改变提示词或方法后应另建输出目录，避免把不同版本混为同一实验。
