# 医疗多智能体证据重复实验

C 的人工错误意见现为指定轮次开始时的公开发言，不指定 `receiver`。`message_error` 由一位专家署名，`repeated_message` 由两位不同专家各表达一次相同观点；两条件共享首位说话者、文本、错误选项和轮次。消息留在公共历史，之后所有专家及主持人均可见、可引用；注入之前的发言看不到它。

2026-11-07 引用协议更新（v2.10）：删除逐次 `visible_ids` 清单和引用 ID 的清单匹配校验。专家和主持人直接使用已提供证据、讨论历史及 `self_previous` 中的编号引用，提示词要求 `Use the IDs attached to the supplied evidence and discussion history when citing them. Do not invent IDs.`。历史判断保留原有 `citations`、命题 `citation_ids` 和公开命题编号。引用字段仍须为非空 ID 字符串组成的列表（允许空列表）；程序不再据清单确认编号是否存在或触发修复。日志继续记录实际输入、暴露及引用关系；`visible_evidence`、`visible_messages` 是输入审计字段，不是引用允许清单。R0 独立、讨论只包含已发生的发言、隐藏金标与内部实验标签的规则保持原流程。旧运行的冻结代码与结果保留，新旧提示词不混算；本次代码修改不恢复已暂停实验。


查看逐题模型对话：运行 `python scripts/log_viewer.py`，打开 `http://127.0.0.1:8765`，按题号或 `run_id` 查找实际请求和原始返回。详见 [本地日志查看器](docs/LOG_VIEWER.md)。

本项目按 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) 实现探索性实验。四名专家先独立作答，再讨论四轮，由主持者汇总。
## 环境与数据

项目使用 Python 3.12。可以用 `conda env create -f environment.yml` 重建环境，再运行 `conda activate med-consensus`。当前验证环境是 `C:\Users\Administrator\miniconda3\envs\med-consensus\python.exe`；`requirements-lock.txt` 记录了当前测试环境的包版本。安装本地包后，命令可用 `evidence-mdt`，下例用 `python -m evidence_mdt.cli`，避免命令路径差异。

```powershell
$py = 'C:\Users\Administrator\miniconda3\envs\med-consensus\python.exe'
```

## 统一材料准备

新批次统一使用 `src/preparation/`，换题只需更换输入题目文件和输出目录。默认准备命令不抽题，也不会自动启动专家讨论；需要抽样时单独使用显式 `sample` 命令。`scripts/` 中旧日期作业和旧准备脚本保留作历史参考，新入口不依赖它们。

同一批题同时准备 B/C/D，使用固定的批量入口，不需要在 `data/` 或 `runs/` 编写驱动脚本：

```powershell
# 已有明确的题目文件：只改输入路径与批次输出路径，题量不限。
& $py -m preparation batch --questions data/new_questions.jsonl --output data/new_preparation --model deepseek --modes b c d

# 离线复查整批材料，也可把 --preparation 指向单个 b/c/d 目录。
& $py -m preparation verify --preparation data/new_preparation --report runs/new_preparation_acceptance.json
```

`batch` 先在 `shared/` 准备一次真实文献、索引和每题最多10条初始证据，然后 B/C/D 复用它，只分别调用材料生成。输出 `batch_manifest.json`、`questions.jsonl`、`preparation_report.md`，以及 `shared/`、`b/`、`c/`、`d/` 数据目录。各阶段仍有自己的准备清单与缓存；根目录汇总逐阶段状态、失败题、结构验收和本目录保存响应的已知 token，不重复计算外部来源的历史用量。不会生成 Python 文件或自动运行答题。

相同命令可续跑；B 的普通技术失败不阻止 C/D，重跑仅补做未成功步骤，配置或认证错误停止。`verify` 不访问网络、不加载 MedCPT、不调用模型，只验证文件哈希、证据与语料的一致性、共用材料一致性及现有实验预检；不代表重放检索排序或核实医学真假。报告应写在冻结材料目录之外。

需要从本地官方源抽取20道新题时，先执行独立抽样。下面仅为命令示例，不会自动扫描历史题；`--exclude` 可重复提供，指定要排除的旧题文件或含 `questions.jsonl` 的目录：

```powershell
& $py -m preparation sample --source data/pilot/MedXpertQA.raw --count 20 --seed 20261028 --exclude data/preparation_test10_oct27_v3 --output data/new20_preparation
& $py -m preparation batch --questions data/new20_preparation/questions.jsonl --output data/new20_preparation --model deepseek --modes b c d
```

抽样仅写 `questions.jsonl` 和 `selection_manifest.json`，保存源文件哈希、排除来源及种子；不会调用模型或检索。题目确定后再次使用同一命令不会重新抽题，变更抽样配置需换目录。这里排除旧10题并不等于排除了所有历史使用题；如需全部排除，应显式列出其他题目来源。

`batch --source-preparation <目录>` 可让同一批题直接复用已完成的10条协议材料，不再检索；外部来源必须位于新批次目录之外。不同题集不能直接复用整套冻结材料。历史 `data/preparation_test10_oct27_v3/` 是旧10题的数据，仍为6条协议，可用于查看材料或作为抽题排除来源，不能直接作为新20题的准备来源。通用代码、已下载的本地 MedCPT 模型和本批成功缓存可以反复使用。

可复用模块职责：`selection.py` 抽题与排除；`batch.py` 编排共用检索及 B/C/D、断点续跑和报告；`audit.py` 离线验收；`__init__.py` 单模式准备；`corpus.py` 文献与切片；`materials.py` 干预材料；`reuse.py` 复用冻结材料；`state.py` 缓存与原子保存；`prompts.py` 提示词；`__main__.py` 命令入口。以下单模式命令仍可使用：

```powershell
# 示例：只准备材料，不运行答题。路径需替换成将来确认的新题文件。
& $py -m preparation --questions data/new_questions.jsonl --output data/new_preparation --model deepseek --interventions b

# 随后准备 C 时，复用同一批真实文献与初始 top-10，只生成 C 的材料。
& $py -m preparation --questions data/new_questions.jsonl --output data/new_preparation_c --model deepseek --interventions c --source-preparation data/new_preparation
```

`--model` 可选 `deepseek` 或 `qwen`；`--interventions` 仅支持 `none/b/c/d`。`none` 只准备真实检索材料，`b/c/d` 在真实材料基础上准备对应干预。普通检索词只依据题干；C/D 的错误目标与解释只供材料制作和审计，不加入普通检索词。

流程为：短查询 → PubMed 相关性检索 → 摘要不足时有限补搜 → 去重切片 → BM25＋MedCPT 向量＋重排 → 固定初始证据 → 对应干预材料。提示词集中在 `src/preparation/prompts.py`，方便直接查看和修改。

PubMed 期刊论文及书籍条目均可提供摘要证据。书籍仅提取 EFetch 返回的摘要，位置标为 `book_abstract`，检索快照保留条目类型。初始短查询生成并接受 2–3 条，逐条检索。全部查询成功后，按文档 ID 去重的非空摘要不足 3 条，才生成一批 2–3 条更宽查询补搜。下载失败记为技术失败，补搜最多一批。

`--source-preparation` 必须指向同一批题的已完成准备目录；程序校验题目与产物哈希后复用，不重新生成查询、下载文献或重排。准备 D 时同样使用原始准备目录，保证 B/C/D 共用真实证据。首次准备默认读取本地 `indexes/model_download_manifest.json`；可用 `--model-manifest` 指定其他本地 MedCPT 清单。

完成后主要产物为 `questions.jsonl`、`initial_queries.jsonl`、`corpus.jsonl`、`index_manifest.json`、`initial_top10.jsonl`，以及所选模式的 `intervention_spec.jsonl`（`none` 无此文件）。`preparation_manifest.json` 记录状态和逐题进度；`cache/` 保存原始调用、文献响应和可复用结果。只有整批准备完成才导出可供实验使用的查询清单和干预规格。当前 B 准备入口生成10份改写，供主实验 literal11/paraphrase11 使用；B 配置明确记录 `b_conditions=["base","literal11","paraphrase11"]`。D 生成5段不同错误证据，配置记录 `d_evidence_count=5` 和固定的5个角度。准备入口包含 B 时默认模型输出上限为12000 token；不含 B 但包含 D 时为6000，其他模式为3000，可通过 `--max-output-tokens` 显式调整。

初始证据上限统一为10条：BM25与向量各召回50条、RRF融合后30条重排的规则不变，最终取前10条供专家使用；不足10条时保留实际数量，不复制凑数。B先逐条标注这份初始快照中各证据直接支持的选项，再从有明确支持选项的证据中等概率随机选一条；全部无明确支持时才从全部初始证据中等概率随机选一条。literal11/paraphrase11共用这个目标，在全部初始证据上另加10份目标原文副本或10份完整改写：目标总计呈现11次，初始满10条时共呈现20条材料。新产物使用`initial_top10.jsonl`和`top10`字段，配置明确要求`initial_evidence_limit=10`；不提供旧命名或旧上限的兼容分支。历史材料和运行记录保留，须在新目录按当前协议准备。

每步完成后保存缓存。中断后使用相同命令续跑，已成功的调用和下载可复用；单题失败会记录并继续其他题，整个批次仍标为未完成。认证或配置错误直接停止。更换题目、模型或准备规则时使用新目录；原始模型响应、文献快照和内容哈希保留，时间戳变化不作为材料改变的依据。材料生成不等同于已核实医学真假，B 的转述及 C/D 的错误点仍需在正式实验前核对。

## 检索与运行

同一批10道题的普通 MDT-RAG 实验可以用以下命令。已有题单时直接从第二步开始，把 `--questions` 指向题单文件。

```powershell
& $py -m preparation sample --source data/pilot/MedXpertQA.raw --count 10 --seed 20261108 --output data/bcd10_new
& $py -m preparation batch --questions data/bcd10_new/questions.jsonl --output data/bcd10_new --model deepseek --modes b c d
& $py -m evidence_mdt.run_batch --preparation data/bcd10_new --output runs/bcd10_new --model deepseek --model-name deepseek-flash
```

`run_batch` 默认调度 B、C、D，每题登记一次共用基线和五个干预条件，10道题共60个团队条件。C 的两个条件复用基线 R0，B 的重复与改写及 D 的错误证据条件各自独立初答。可用 `--modes b c` 等参数选择已准备的模式。抽样、材料准备和运行代码都在 `src/`，`tests/` 的模型桩和题目样例用于离线软件测试。换一批题时更换题单及输出目录即可。

配对运行同批B/C且只使用普通MDT-RAG时，使用可续跑入口：

```powershell
& $py -m evidence_mdt.run_bc --preparation data/bc_preparation --output runs/bc_ordinary --model deepseek --model-name deepseek-flash --workers 4 --max-output-tokens 6000
```

两个入口都验证共用冻结材料。初始证据来自准备快照，按普通 MDT-RAG 配置执行。按题并行，每题的原始调用、事件和失败记录保存在 `questions/qNNNN/`，根目录 `assignments.jsonl`、`results.jsonl`、`summary.json`、`report.json` 汇总全批。启动时复制当时的 `src` 代码到 `software_snapshot/`，续跑使用该快照和已成功结果。每个未完成条件默认最多尝试两次，认证错误或全批预算耗尽会停止。默认全批上限30000调用及300000000已报告 token，执行预算可在续跑时调整。令牌上限在下次调用前检查，并发在途调用的用量可能随后计入。



新实验入口仅接受 MedXpertQA（包括 MedXpertQA Text）题集。输入多少题，就对这些题逐一运行；没有固定 25 题、15/10 分层或隐式子集抽样。2026-10-28用户授权全新100题的普通组B/C；当前进度以对应准备清单和运行summary.json为准。通用数据准备与旧日志读取仍保留历史数据集支持。

`--interventions none` 仅运行共用的 `B:base`；`b` 追加 `B:literal11`、`B:paraphrase11`，`c` 追加 `C:message_error`、`C:repeated_message`，`d` 追加 `D:evidence_error`。仅支持 `none/b/c/d` 四种选择。`--b-extended` 保留为显式追加历史 literal5/paraphrase5 条件的可选参数，新默认不调度3次或5次条件。每题每次独立重复只运行一次基线，所有选择的干预作用于全部输入题。普通组可用 `--groups ordinary` 单独运行；`all` 比较三组。

启用干预时，`--intervention-spec` 必须逐题恰好一行；只需提供启用条件所需的字段。示例（人工材料应在实验前核对并固定）：

```json
{"question_id":"题号","b_target_id":"初始快照chunk_id","target_answer":"B","b_paraphrases":["完整改写1","完整改写2","完整改写3","完整改写4","完整改写5","完整改写6","完整改写7","完整改写8","完整改写9","完整改写10"],"c_message_text":"同伴声称{wrong_text}正确","c_wrong_option":"C"}
```

B 的目标片段必须出现在该题初始检索快照，转述保留人工派生关系。自 2026-11-07 起，准备时先用一次调用对全部冻结初始证据分别标注直接支持的选项或 `null`，不读取金标，也不借其他证据补足支持。然后在有明确支持选项的证据中等概率随机选一条；只有全部为 `null` 时才从全部证据中等概率随机选一条。按证据条目抽样，不先抽选项；支持任何合法选项的证据均可入池。保存 `b_evidence_annotations`、`b_target_candidate_ids`、`b_target_selection`（`supported` 或 `all_null_fallback`）及种子所确定的目标，续跑不会重新标注已保存的证据或更换目标。新 B 配置记录 `b_target_selection_policy=supported-options-uniform-v1`，离线验收会核对标注覆盖、候选池及种子对应的抽取结果。

选中证据的标注写入 `target_answer` 和 `b_target_reason`，改写调用只接收选中原文，不再标注选项。全部无支持时仍保留 `target_answer=null` 和原因，生成改写并运行全部 B 条件，纳入整体结果分析。目标选项数值变化只计算标注明确且有效配对的子集，报告 `n_valid_pairs`、`n_intervention_trials` 和 `n_excluded_no_target_annotation`；子集不能代表全部题目。遗漏证据、重复或未知编号、非法选项、缺失标注或原因均作为材料错误处理，不当作全部无支持。此规则改变了目标抽样总体，新材料使用新目录，旧批次保留其原抽样规则和结果。

自 2026-11-07 起，B 提示词要求10份改写各自保留整段原文的信息，不以摘要、节选或多份分担原文代替。按用户确认，改写副本以独立虚构文献呈现：chunk_id、document_id、来源链接和标题均独立生成，正文保持原文含义；.example 链接只作呈现标识。真实原文保留在初始证据中，派生关系通过后台 derived_from 追踪，模型不接收该字段；原文重复条件仍保留真实来源。此设计同时操作改写和表观来源身份。不新增改写语义 check 或长度硬门槛，旧批次不改写，新批次另用输出目录。旧材料的离线验收按其记录的条件执行，未记录条件的历史 B 材料按原共3次协议检查；新运行入口要求10份改写。

C 只干预公共讨论历史：消息错误由一位专家公开表达预设错误观点；重复消息由两位不同专家表达同一错误观点，均在指定讨论轮开始插入，后续专家与主持人可见。C 复用同题 `B:base` 的 R0。

D 在 R0 前追加5段不同错误证据，初始10条时共15条，单独生成 R0。准备时先等概率随机固定一个非金标选项，再用一次调用统筹编写5段，分别从机制、临床表现或人群、检查解读、鉴别比较、病程或反应角度独立支持同一个错误选项。每段必须有不同的实质错误主张，不复制、改写或拆分同一个错误来凑数，也不依赖其他段完成论证；角度按题型调整。正文建议每段100–160英文词，不编造真实 PMID、DOI、期刊或指南权威，不写选项字母或实验标签。生成后固定整组材料，三种方法共用，续跑不更换错误选项。

D 规格使用 `d_wrong_option` 和 `d_evidence_passages`。生成提示词要求5项，每项包含 `angle`、`text`、`false_claim`、`error_explanation`。程序按模型返回顺序保存并追加正文，直接记录实际段数。审计说明随返回结果保存，可以缺省。以下结构中的正文和错误点需替换成实际材料。

```json
{"question_id":"题号","d_wrong_option":"D","d_evidence_passages":[
  {"angle":"mechanism","text":"机制角度的证据正文","false_claim":"机制方面的错误主张","error_explanation":"指出机制具体错在哪里"},
  {"angle":"clinical_pattern","text":"表现或人群角度的证据正文","false_claim":"表现关联方面的错误主张","error_explanation":"指出关联具体错在哪里"},
  {"angle":"diagnostic_interpretation","text":"检查解读角度的证据正文","false_claim":"检查解释方面的错误主张","error_explanation":"指出解读具体错在哪里"},
  {"angle":"differential_comparison","text":"鉴别比较角度的证据正文","false_claim":"选项区分方面的错误主张","error_explanation":"指出比较具体错在哪里"},
  {"angle":"course_or_response","text":"病程或反应角度的证据正文","false_claim":"病程预测方面的错误主张","error_explanation":"指出预测具体错在哪里"}
]}
```

5段分别获得独立的 chunk_id、document_id 和正文哈希，以普通 local-document 证据呈现。`intervention_manifest.json` 与结果的 `d_evidence_audit` 逐段关联编号、角度、错误主张及说明；结果保留汇总的 `d_error_explanation` 便于查看。目标、角度、错误说明和人工标记均不发给答题模型，模型只见5段正文和常规引用字段。旧单段 `d_evidence_text`/`d_error_explanation` 及旧 `c_evidence_text` 不能用于新运行，必须在新目录重新准备；历史单段 D 材料仍可离线验收，旧结果不重写。

运行前完成全部素材预检；`intervention_manifest.json` 记录全部输入题、条件、初始片段、C 公共注入位置以及 D 的目标和错误点。不要把这些审计字段填进模型材料。

运行前一次写入 `assignments.jsonl`，包含尚未开始的分配。`results.jsonl` 只记录成功完成的题组；`failures.jsonl` 记失败的 `run_id`；`calls.jsonl` 和 `events.jsonl` 留存调用及逐轮事件。`config.json` 的指纹覆盖数据、代码、提示词、实验计划、干预规格和科学参数；执行预算另记于 `budget_limits.jsonl`。科学配置变更后用新输出目录；同指纹重跑会跳过已成功组。`evaluate` 在 `baseline` 字段按数据集、方法报告全部题的 B 基线，并在 `all_variants` 按变体统计，保留失败和未开始的分母。旧日志仍可读取：同一题、独立重复、组同时含 `B:base` 与 `natural` 时优先前者，仅前者缺失时回退后者，不重写原始结果。`b_target_probability_change` 和 `d_target_probability_change` 分别统计与同题、同重复、同组实际 B 基线有效配对的目标数值变化，未设置目标的基线不会计作干预。原始概率值只校验键完整、有限数值及 [0,1] 范围，不要求总和为 1，也不归一化。报告保留旧 `brier` 字段计算平方误差公式，但这些数值不保证是归一化分布，不据此声称校准。

语义关系缓存只在同一道题、同一次独立重复内共享；同次两组复用时，事件日志记录命中关系、命题对哈希、来源组、attempt、调用序号及批中位置。不同 repeat 重新判定。因此组间实际 token 成本包含共享预处理，不能直接解释为各组单独部署的成本。

`--max-calls` 是调用尝试次数的硬上限，网络失败也计数。`--max-tokens` 依据服务端已返回的 token 用量在**下一次调用前**检查；某次调用可能使累计值越过上限，服务端未返回用量时会在日志中单列未知用量。这两个限制控制运行规模，不能保证精确费用上限。

代码阅读顺序及各模块职责见 [CODE_GUIDE.md](CODE_GUIDE.md)。
