# 可复用的 B/C/D 材料准备

这里保存通用准备程序。`data/<批次名>/` 只保存题目、材料、缓存和报告，不再为每次实验编写新的 Python 驱动。

## 准备新 20 题

在项目根目录运行，使用已安装本项目的 Python 环境：

```powershell
# 1. 如果还没选题：显式抽取 20 题，排除旧 10 题。
python -m preparation sample --source data/pilot/MedXpertQA.raw --count 20 --seed 20261028 --exclude data/preparation_test10_oct27_v3 --output data/new20_preparation

# 2. 同批 B/C/D 只准备一次共用检索材料。
python -m preparation batch --questions data/new20_preparation/questions.jsonl --output data/new20_preparation --model deepseek --modes b c d

# 3. 离线检查完整性；报告放在冻结材料目录外。
python -m preparation verify --preparation data/new20_preparation --report runs/new20_preparation_audit.json
```

已有题目文件时直接执行第 2 步。输入文件有多少题就准备多少题，不限定 20；不需要的模式可以从 `--modes` 去掉。抽样只排除明确提供的来源，若要排除其他历史题，继续添加 `--exclude <题目文件或目录>`。

首次真实准备需要 `.env` 中的模型凭据及本地 `indexes/model_download_manifest.json` 指向的 MedCPT 模型；`sample` 和 `verify` 不调用模型 API。以上命令只准备材料，不启动专家答题。

## 程序分工

| 文件 | 职责 |
| --- | --- |
| `__main__.py` | 命令入口：抽题、批量准备、单模式准备、验收 |
| `selection.py` | 固定种子抽题、排除旧题、保存抽样来源 |
| `batch.py` | 编排共用检索与 B/C/D，汇总状态和报告 |
| `__init__.py` | 单模式准备流程、逐题状态及冻结检索快照 |
| `corpus.py` | PubMed 文献下载、摘要解析、去重和切片 |
| `materials.py` | B 随机证据与改写、C 错误意见、D 错误证据 |
| `prompts.py` | 准备阶段的提示词 |
| `reuse.py` | 验证并复用同一批题的冻结材料 |
| `state.py` | 缓存、内容哈希、原子保存和错误处理 |
| `audit.py` | 离线检查产物与共用材料的一致性 |

模型调用、检索排序等底层能力继续使用 `src/evidence_mdt`，不复制到批次目录。

## 输出目录

```text
data/new20_preparation/
  questions.jsonl              # 本批题目
  selection_manifest.json     # 使用 sample 时生成
  batch_manifest.json         # 总状态、阶段进度、配置指纹
  preparation_report.md       # 准备结果报告
  shared/                     # 共用文献、切片、索引、查询和初始最多 10 条证据
  b/                          # 共用材料副本与 B 干预规格
  c/                          # 共用材料副本与 C 干预规格
  d/                          # 共用材料副本与 D 干预规格
```

各阶段目录中的 `preparation_manifest.json` 记录逐题状态，`cache/` 保存已成功的调用和下载。重复执行相同命令会复用成功步骤；普通技术失败保留记录，修复后续跑。题目、模型或准备代码改变时应换新的输出目录。

## 哪些可以复用

- **任何新批次**：这里的程序、`src/evidence_mdt` 的底层能力和已下载的本地 MedCPT 模型。
- **同一批题续跑**：原输出目录里的成功缓存，不用重做成功调用。
- **同一批题增加其他模式**：新输出目录配合 `batch --source-preparation <原批次/shared>`，复用已完成的当前 10 条协议快照，只生成干预材料。也可传入完成的单模式准备目录；不能传批次根目录。
- **不同题目**：重新生成对应查询、检索快照和干预材料；不直接套用旧题的完整产物。

`data/preparation_test10_oct27_v3` 是旧 10 题的历史材料，并且仍为 6 条证据协议。它可以用于查看历史记录或排除旧题，不能直接传给当前 `--source-preparation`。本入口不会自动把旧材料升级、混入新批次，也不会跨批次自动导入旧缓存。

`verify` 检查文件哈希、结构和材料一致性，不调用模型，不重放真实检索排名，也不证明干预文本在医学上正确或错误。
