# 项目执行记录

## 2026-10-12 概率容差边界修复完成（未重启实验）

用户要求排查实验 A 的概率格式错误，且继续暂不分析结果。两次失败输出的十进制总和均为 0.99；原先 `abs(sum(p)-1)>0.01` 算出偏差 0.010000000000000009，错误地拒绝了原本允许的 ±0.01 边界。答题模型适配文件与失败 A 快照、已完成 MedQA 预筛快照的哈希相同；查询生成与答题走不同分支，这次停机的直接原因是概率校验浮点边界，不是检索词提示词的格式要求。

`schema.py` 新增共享十进制求和校验，`runner.py` 格式修复保护和 `metrics.py` 有效向量检查均复用它；原容差保持不变，不归一化，不改概率或答案，缺项/非有限数/负数/真正超界仍拒绝。两次真实向量用合成题目离线回放通过，未触发修复调用；最终 **116 项测试通过**。诊断记录见 `runs/probability_validation_fix_oct12/diagnosis.md` 和 `reproduction.json`。本次无新增 API 调用，不计算准确率或其他实验效果；旧 A 的 11 次调用、45,056 token、两次失败日志原样保留。代码指纹已变化，后续恢复实验 A 须用新输出目录，当前未重启后台运行。

### 2026-10-12 MedXpertQA更难题集检索材料准备完成

用户授权本阶段跳过错题溯源，仅准备更难题集材料；**未启动普通组或A/B/C答题**。新评分题集 `data/medxpert_pilot_oct12/pilot50.jsonl` 固定30道MedXpertQA Text与原20道PubMedQA。保留原20道MX，以seed=20261012仅按官方剩余ID无放回追加10道；官方raw哈希与旧抽样清单一致，全部30题标准化字段与源题一致，PubMedQA文件逐字保留。公开题面排除金标与长答案，B/C另冻结15MX+10PQA题号；PQA子集与原选择一致，未制作干预材料。

最终相关性初始快照 `data/medxpert_relevance_oct12_v2/` 与宽查询快照 `data/medxpert_broad_oct12_v3/` 完成。PQA原查询/快照逐题核对题面哈希后复用；新MX直接PubMed sort=relevance，每条最多8篇。初始独立摘要少于3篇固定触发31题（24MX+7PQA），最终545父摘要。整个语料排除20个PQA目标PMID，只有26348845和21823940仍无同题来源摘要，其余48题同题独立摘要至少3篇。技术完整性不自动代表医学相关性。

首遍urllib网络截断/连接断开以及查询准备重复改完成时间引发的父哈希错误均保留。新v2只迁移父清单元数据，核对公开题面、查询和检索参数一致，以curl有限恢复，未放宽旧指纹；四题宽查询两次返回不含AND的字符串，新v3仅对已有查询做一次保词格式修复，逐query词Counter完全一致，未添加医学概念。Text-281本地组合查询69token，保留原第一完整查询修订至43token，原两查询及PubMed请求不改。所有变更、原失败、XML、来源哈希和修订侧车保存；一条HTTP400失败体的缺失hash在验收时补算侧车，只固定当时留存文件，不证明下载时未经变化。

离线分片 `data/medxpert_chunks_oct12/` 已完成：545父摘要中148篇切分，共798个连续不重叠片段，标题加分隔符正文最大399token，逐字可拼回原文，父ID/字符区间/hash完整。新索引 `indexes/medxpert_chunks_oct12.json` 及向量哈希核验通过；BM25top50+MedCPTdense50，RRF60取30重排top6。`runs/medxpert_retrieval_qa_oct12/` 完成50题检查：50/50返回6条，300/300引用有URL与hash，查询最长52token、重排查询最长50token，无超64或空top6，仅上述两道PQA无同题来源。准备进程PID24196已正常结束。

本轮新增查询API共62次、31,800个已报告token：初始30次/15,750token，宽查询32次/16,050token（包含8次语法失败与4次格式修复）。历史复用初始20次/12,845token、宽查询7次/2,711token单列，不计新增费用；未读取货币账单。完整测试98项通过；随后新增失败体hash侧车最小测试，专项7项通过（包含该新增1项），未重复全套。最终64文件软件快照和早期快照均保留。

中文报告 `runs/medxpert_preparation_oct12/preparation_report.md`、验收 `acceptance.json`、费用 `costs.json` 与各来源/分片审计完整保存。后续每条件repeats=1，仍须显式设置进程环境MDT_MAX_OUTPUT_TOKENS=2000、OMP_NUM_THREADS=4、MKL_NUM_THREADS=4；本阶段到此停止，不自动进入实验。

更新日期：2026-10-12。总体研究约定以 `EXPERIMENT_PLAN.md` 为准；本文件记录实际完成的工作，方便额度中断后接续。

## 当前授权与配置（2026-10-12）

用户跳过 MedQA 错题专项溯源，授权直接准备 **30 道 MedXpertQA Text＋原 20 道 PubMedQA** 的检索材料。保留原 20 道 MedXpertQA、从剩余题池固定随机追加 10 道；原 PubMedQA 题目与金标保持原样，尽量复用其查询和文献快照。新材料另建目录，不覆盖旧预筛文件。当前阶段包括题目选择、短查询、相关性排序检索、固定规则补搜、分片、索引与检索质量检查；不启动 A/B/C 答题。

后续所有题目每方法/干预条件只运行一次（repeats=1）；CLI 默认值、README 与 EXPERIMENT_PLAN 已同步。四专家、四轮讨论和 B 的重复呈现条件不变；A 自然条件预期 50×3×1=150 团队结果。已完成 MedQA 两次重复的原始结果、门槛及成本均保留历史定义。下面各日期记录描述当时状态，不能替代本段当前设置。

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

后续协作模型约定（2026-10-11 用户更新）：已有子代理保持原模型；新建有一定难度的子任务使用 GPT-6.1 Sol medium，简单子任务使用 GPT-6 Luna medium，不再新建使用 GPT-5.6-terra 的子任务。已存在的 `implementation_oct09`、`corpus_oct09`、`natural_analysis_oct09` 继续原 GPT-6 Sol medium，没有中途切换。

2026-10-11 用户授权执行下一阶段：30 道 MedQA 普通组，每题两次独立运行，共 60 个团队结果。仅完成该批次与结果审计后停止，等待用户确认，不自动进入 A/B/C。执行任务交给新建 GPT-6.1 Sol medium 子代理 `medqa_screen_oct11`，主代理审查。预设执行上限为 2,000 次调用尝试、1,000 万个已报告 token；触顶停止汇报，不自动增加预算。该限额不是预计用量或精确费用承诺。

本次实际输出目录改为 `runs/medqa_screen_chunks_oct11/`：旧 `oct09` 软件快照之后存在 `runner.py` 图日志缓存修复，故保留旧快照并新建本次快照，不在旧目录混用版本。新目录已保存 35 个软件文件与哈希清单、`preflight_notes.txt`、完整启动命令与执行日志。启动前现有全套测试 **91 项通过**；语料与向量哈希验证通过。2026-10-11 16:50（北京时间）启动，初始 PID 4532；模型 `deepseek-flash`，temperature=0.3，最大输出 800 token，60 个任务已登记。是否仍在运行以进程及日志为准，不能仅凭本段认定完成。

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

### 2026-10-11 预筛首次启动后的异常节点

`runs/medqa_screen_chunks_oct11/` 首次作业已退出（PID 4532）。60 个预登记任务中 **0 成功、1 失败、59 未开始**，不是医学准确率 0%。累计 10 次调用、58,330 个已报告 token；其中 2 次 API 解析失败，均属于首题第一次团队运行。第 9 次讨论发言和第 10 次唯一格式修复都达到单次 800 token 输出上限，服务端 finish_reason=length，JSON 被截断。未到总体预算，未自动重试，也未修改代码、题目、提示词或环境参数；A/B/C 未启动。

已完成离线 evaluate、图日志审计、难度门槛检查和共同错误分析。没有成功结果，准确率/Brier/共同错误均不可评估；54/60 切换规则尚不能执行。详情与处理建议见该目录 `screen_acceptance.md`，失败、部分轨迹与用量全部保留。下一步建议须另经用户确认：新目录固定单次输出 2000 token，先完成两个团队技术验收。

### 2026-10-11 两团队输出上限技术验收完成

用户随后确认下一小步，实际在新目录 `runs/medqa_output2000_check_oct11/` 仅跑第一道 MedQA（question_id=0）两次独立团队：max_questions=1、ordinary、repeats=2。通过进程环境将单次输出上限固定为 2000 token，未改 `.env`、源码或计划；原总预算不变。**2/2 成功，0 失败，44 次 API 全部 finish_reason=stop，最长输出 805 token**。两个 repeat 各自实际招募并执行 4 次 R0，无跨 repeat 或旧批次回放。总用量 **322,842 token**（输入 293,603、输出 29,239），全部有用量记录。离线图审计 42 次发言、1,239 条图边，0 问题和提醒；evaluate 确认2分母完整，答案及概率向量有效。

两次均答对同一道题，仅用于技术验收，不能推断泛化或执行54/60难度切换。PID21492已退出，本次已停止，尚未开始全60或A/B/C。软件快照、完整命令、日志、technical_checks.json及screen_acceptance.md均保存在新目录；旧800输出失败和用量单独保留。后续完整预筛须另经用户确认并用新目录固定2000输出配置，本次不导入作完整预筛成功任务。

### 2026-10-11 完整MedQA预筛完成

用户随后授权30题全部完成，每题两次。完整新目录 `runs/medqa_screen_output2000_full_oct11/` 已取得 **60/60成功结果，56/60正确（93.33%）；repeat1和repeat2各28/30**。28道两次均正确，2道两次均错误；29/30题两次答案一致。错题803两次选D/B、金标C；105两次选A、金标C。成功轨迹核实四专家和主持者共同错4团队/2题，高置信共同错2团队/1题；221条R0正确专家轨迹中无最终转错。Brier=0.138783。题目金标和选项核对无差异，60 run_id唯一。

本批单次输出上限固定2000，科学配置未改变。首进程30428完成45结果后，299repeat2因answer和repair均2000截断失败；用户授权同配置重做未完成团队一次、总token上限提高1500万，续跑PID25900跳过成功结果并完成剩余。最终1327次调用、10545707个已报告token（输入9495212、输出1050495），含历史失败成本；2次API解析失败、1条失败attempt保留，最终未完成/失败任务为0。第60团队另有一次有效格式修复。无网络重试、无R0回放，旧两团队技术结果未移植。

离线evaluate/audit/check_difficulty/common_errors均完成；图审计1260发言、44260图边，0问题/提醒。难度检查ready，56≥54触发既定后续临床题集切换为MedXpertQA Text，但未启动MedXpert或A/B/C。两个执行进程均已退出，本阶段已停止。60结果是30道题的相关重复，不是60独立病例，不证明临床有效性。完整中文报告见本目录 `screen_acceptance.md`，调用/事件/失败/预算记录均保留；旧800失败与2团队技术检查成本单列。

### 2026-10-12 实验A已启动（暂不分析结果）

用户授权实验A运行：50题=30道MedXpertQA加20道PubMedQA，三组all、repeats=1、interventions=none，共150个唯一任务（每组50）。新目录 runs/experiment_a_single_oct12；2026-10-12 02:44:48北京时间启动，Python PID32932、独立监督PID5012。实际config确认deepseek-flash、temperature0.3、输出上限2000；OMP/MKL各4线程。执行硬上限20000次调用尝试、100000000已报告token，这是限制而非预估/已发生用量。

题集/公开题面/语料/索引/查询/向量哈希已与抽样和准备验收记录核对；新software_snapshot保存42文件，不含密钥。未修改科学代码、提示词、数据或旧预筛记录。执行参数、PID和开始时间见execution_attempt1.json，stdout/stderr独立落盘。监督程序只保存退出码、分配/完成/历史失败行数及是否150完成，不解析答案、金标或概率，不执行evaluate、难度检查、共同错误分析或任何准确率/Brier计算。格式失败自动停止并保留日志，不自动无限重试；本阶段只运行，最终状态以completion_status.json和进程/日志为准，不能凭本段认定全部完成。

实验A首进程32932于02:45:17退出，150已登记、0完成、1历史失败；首题R0出现probabilities must sum to one，5次调用/15250个已报告token。未分析答案或效果。原completion_status已保存为completion_status_attempt1.json。根据随后明确授权，同科学配置重做未完成团队一次（不归一化或修改概率/提示词），第二次执行记录在execution_attempt2.json及独立stdout/stderr；旧失败/成本完整承接。若再次格式失败立即停，不无限重试。

实验A一次授权重试也已停止：PythonPID9820于02:46:31退出，exit_code=1，完成0/150。同一首题第二个attempt通过R0后，在第一轮讨论再次出现probabilities must sum to one。累计11调用/45056已报告token，2条历史失败attempt（同一团队），没有完成答案可分析。按约定未无限重试、不归一化概率、不改提示或参数；无后台问答进程。completion_status_attempt1/2.json、stderr_attempt1/2.log与全部调用失败记录均保留，analysis_performed=false。

### 2026-10-12 修复版实验A重新启动（仅运行）

用户确认重启后，以已通过116项离线测试的Decimal概率边界修复版代码在新目录 runs/experiment_a_single_fixed_oct12 重新运行，不移植旧失败记录。2026-10-12 03:59:52北京时间启动：PythonPID10892，独立监督PID30236。已确认150个唯一任务，ordinary/semantic/semantic_retrieval各50；50题为30MedXpertQA和20PubMedQA，all、repeats1、interventionsnone，deepseek-flash、temperature0.3、maxout2000，OMP/MKL4。有限执行硬上限20000调用/100000000已报告token（限制而非预计用量）。

题集/798语料/索引/固定查询/向量hash与准备记录一致；新software_snapshot含43文件（源码、计划、依赖、脚本、新回归测试），不含密钥。启动后API调用日志已前进、尚无失败；作业后台独立运行，实际完成状态以后续completion_status为准。监督仅记exit和行数，不读答案/金标/概率、不运行evaluate、难度检查或任何效果分析；若格式失败自动停并保留日志。旧experiment_a_single_oct12两失败attempt和45056token完整保留，未混入新批次。

### 2026-10-12 实验A重启恢复

电脑重启后确认实验A目录JSONL均可解析：assignments150唯一、results6成功且同一fingerprint、calls203均ok、无截断文件；source/计划/输入与当前配置及新快照哈希一致。旧进程不存活，无失败文件。保存restarts.jsonl记录重启、已完成6和203已知调用成本；未知在途用量不臆测。

按原命令同指纹续跑，跳过6个成功任务，监督PID23580、PythonPID1996于04:17:15北京时间启动；配置all/repeats1/none/maxout2000、20000calls/1亿tokens不变。续跑仅记录JSONL完整性/退出状态，不解析答案、gold、概率，不执行任何分析；以completion_status_attempt2.json和进程状态为准。
