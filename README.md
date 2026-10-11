# 医疗多智能体证据重复实验

本项目按 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) 实现探索性实验。四名专家先独立作答，再讨论四轮，由主持者汇总。首轮主比较为普通 MDT、语义合并、语义合并加定向补检索。输出仅用于研究，不是临床建议。五题真实模型烟测已用于软件与成本验收，尚无正式先导结果；`mock` 只检验软件流程，不能作为医学结果。

## 环境与数据

项目使用 Python 3.12。可以用 `conda env create -f environment.yml` 重建环境，再运行 `conda activate med-consensus`。当前验证环境是 `C:\Users\Administrator\miniconda3\envs\med-consensus\python.exe`；`requirements-lock.txt` 记录了当前测试环境的包版本。安装本地包后，命令可用 `evidence-mdt`，下例用 `python -m evidence_mdt.cli`，避免命令路径差异。

```powershell
$py = 'C:\Users\Administrator\miniconda3\envs\med-consensus\python.exe'
& $py -m evidence_mdt.cli prepare --source pilot --output data/pilot --seed 20261008
```

`data/pilot/pilot50.jsonl` 是 30 道 MedQA 加 20 道 PubMedQA。`data/pilot/MedXpertQA.jsonl` 是单独的 20 道备用题，只有满足计划中的 MedQA 难度门槛才纳入后续主比较。数据文件和运行记录位于 `.gitignore` 排除的 `data/`、`runs/`、`indexes/`。

`selection_manifest.json` 保存原始下载 URL、SHA-256、随机种子、抽样框大小和选中的源 ID。MedQA 使用**第三方 BigBio 镜像训练集前 1000 行**作为固定抽样框，从中随机抽 30 道；这不是从全部 MedQA 训练题随机抽样。PubMedQA 来自官方 `ori_pqal.json`；模型可见题干和 `CONTEXTS`，看不到 `LONG_ANSWER`、金标或预测答案。MedXpertQA Text 来自官方测试集，属于探索备用题，不称为训练集。题目只用于探索，正式确认需另取未参与探索的新题。

## 检索与运行

先准备可审计的本地语料。纯文本导入用于自己有权使用的材料；PubMed 快照会保存摘要、标题、PMID、DOI 元数据和原始 XML。论文源码、README 或说明网页不得当作患者问题证据。

```powershell
& $py -m evidence_mdt.cli prepare --source pubmed --query 'diabetes metformin trial' --max-records 20 --output data/pubmed.jsonl
& $py -m evidence_mdt.cli index --corpus data/pubmed.jsonl --backend medcpt --output indexes/pubmed.json
```

MedCPT 使用固定版本的 Query、Article、Cross Encoder；本机可读取 `indexes/model_download_manifest.json` 的本地快照和 revision。Query 最长 64 token，Article 最长 512 token。BM25 和向量召回各取前 50，同 `chunk_id` 做 RRF 融合，再取前 30 交给 Cross Encoder 重排。`mock` 检索只供流程测试，不能当真实检索报告。

五题真实烟测可先运行 `& $py scripts/prepare_smoke_corpus.py`。它固定取已抽出的前三道 MedQA、前两道 PubMedQA，调用 DeepSeek 生成不作答的简短 PubMed 查询，下载每条查询最多 8 篇摘要，并保存请求、返回、空结果、原始 XML 与哈希到 `data/smoke/`。该命令会调用付费 API；运行前先检查脚本和预算。其小语料只用于软件与成本烟测，不能用于正式 50 题比较。

30 道 MedQA 的普通组难度预筛可单独跑；两次独立运行共 60 个主持者答案，若正确数至少 54，再切换临床题集。主比较再使用 `--groups all`。真实 DeepSeek 运行从项目根目录的 `.env` 读取 `DEEPSEEK_API_KEY`；当前已配置密钥，`.env` 被 Git 忽略，`.env.example` 只列变量名和空白占位。实际 API 调用会产生费用。

```powershell
& $py -m evidence_mdt.cli run --questions data/pilot/MedQA.jsonl --corpus data/pubmed.jsonl --output runs/medqa-screen --model-backend deepseek --retrieval-backend medcpt --index-manifest indexes/pubmed.json --query-manifest data/pilot/short_queries.jsonl --groups ordinary --repeats 2
& $py -m evidence_mdt.cli run --questions data/pilot/pilot50.jsonl --corpus data/pubmed.jsonl --output runs/pilot --model-backend deepseek --retrieval-backend medcpt --index-manifest indexes/pubmed.json --query-manifest data/pilot/short_queries.jsonl --groups all --repeats 2
& $py -m evidence_mdt.cli evaluate --results runs/pilot/results.jsonl --output runs/pilot/report.json
```

正式运行使用人工核对的短查询清单。每行只含 question_id、dataset、public_question_sha256、retrieval_query 四个字段。公开题面哈希按 question.public() 的 JSON 排序键、UTF-8 编码、紧凑分隔符计算 SHA-256；完整 50 题查询文件可直接用于 30 题 MedQA 预筛。程序在模型调用前检查当前运行的每道题，拒绝重复 ID、隐藏字段和空查询。显式短查询超过 MedCPT 64 token（含特殊 token）会报错。未提供清单时沿用旧版题干加背景检索；补检索不受此选项影响。MedCPT 索引的模型 revision 必须与本次加载的模型一致。

B/C 探索须另传 `--interventions both --intervention-spec data/intervention_spec.jsonl`。规格文件每行对应一道预抽题，例如：

```json
{"question_id":"固定题号","b_target_id":"初始快照中的chunk_id","b_paraphrases":["研究者核对的转述1","研究者核对的转述2"],"target_answer":"A","c_evidence_text":"研究者预先规定并标记为人工扰动的文本","c_message_text":"同伴声称选项{wrong_label}正确"}
```

`b_target_id` 必须实际出现在该题初始检索快照；改写文本须由研究者核对、在实验前固定，并保留人工派生关系。`c_message_text` 用 `{wrong_label}` 或 `{wrong_text}` 插入程序预先抽取的错误选项；也可明确写 `c_wrong_option`，它必须不是金标。人工扰动不自动构成医学真假金标。B/C 共用预抽的 15 道当前临床题和 10 道 PubMedQA 题，首轮 B 仅运行 `base`、`literal3`、`paraphrase3`；后续显式传 `--b-extended` 才加入两种 5 次暴露。运行前完成全部规格、目标片段和 C 注入配置预检；`intervention_manifest.json` 冻结抽样、条件、初始片段及随机注入位置，哈希与规格文件 SHA-256 均纳入运行指纹。

运行前一次写入 `assignments.jsonl`，包含尚未开始的分配。`results.jsonl` 只记录成功完成的题组；`failures.jsonl` 记失败的 `run_id`；`calls.jsonl` 和 `events.jsonl` 留存调用及逐轮事件。`config.json` 的指纹覆盖数据、代码、提示词、实验计划、干预规格和科学参数；执行预算另记于 `budget_limits.jsonl`。科学配置变更后用新输出目录；同指纹重跑会跳过已成功组。`evaluate` 按数据集、方法、变体分开统计，并保留失败和未开始的分母。

语义关系缓存只在同一道题、同一次独立重复内共享；同次两组复用时，事件日志记录命中关系、命题对哈希、来源组、attempt、调用序号及批中位置。不同 repeat 重新判定。因此组间实际 token 成本包含共享预处理，不能直接解释为各组单独部署的成本。

`--max-calls` 是调用尝试次数的硬上限，网络失败也计数。`--max-tokens` 依据服务端已返回的 token 用量在**下一次调用前**检查；某次调用可能使累计值越过上限，服务端未返回用量时会在日志中单列未知用量。这两个限制控制运行规模，不能保证精确费用上限。

代码阅读顺序及各模块职责见 [CODE_GUIDE.md](CODE_GUIDE.md)。
