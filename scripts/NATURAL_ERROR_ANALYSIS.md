# 自然共同错误审计

运行 `python scripts/analyze_common_errors.py runs/<运行目录>`，生成 `analysis/common_errors.json` 和简短中文报告。脚本只读取成功的 `natural` 结果，按 `run_id + attempt_id` 找对应原始事件；失败旧 attempt 不进入本次图。`replayed_from` 明确记录来源 attempt，不能凭同名 turn 跨 attempt 推断。

每个案例保存题集、组别、题号、repeat、成功 attempt、R0／最终专家／主持者 turn ID，便于回到 `results.jsonl` 和 `events.jsonl` 核查。补检索且确有新片段时，最终专家取 review；其余取 R4。共同错误要求四专家和主持者同选非 gold 答案；高置信再要求主持者对该答案自报概率至少 0.8。没有完整日志时相关判断为 `null`，同时列出缺失对象。

证据集中度只沿外显引用、消息展示映射和已有簇成员引用回溯到原始 chunk，再用登记的 document ID 汇总；无可追溯引用的集中度为 `null`。同一簇 ID 的成员可能随讨论增加，因此按引用发言前的事件快照解析，后来的成员不会倒灌到早期引用。案例单独列出可见消息和显式引用，前者不表示实际采纳。语义组仅汇总日志中最终主持者所见的已有簇；普通组未做该聚类，字段为 `null`。同义簇也不表示真实文献同源。本审计不新增模型聚类或医学正确性判断。

难度门槛使用 `python scripts/check_difficulty.py runs/<运行目录>`。题目默认取 `config.json` 记录的路径，也可用 `--questions` 指定迁移后的文件；两种方式都校验配置中的 SHA-256。它核对原题 ID、四个选项、gold、所有分配和结果的配置指纹，只在原定 30 道 MedQA 的普通组、natural、repeat 1 与 2 共 60 条成功结果且无重复时给出 `≥54/60` 切换判断；未完成返回 `pending`，日志或来源冲突返回 `invalid`。PubMedQA 独立统计。
