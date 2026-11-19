# DSPN：动态稀疏可塑性神经网络

自研的新架构：固定内存预算下持续预测、在线学习、保留旧规律；文本模式下检验能否学会语言。

## 去哪找什么

每类信息只有一个地方。别处只放链接，不复制内容。

| 想知道 | 唯一来源 |
|---|---|
| 文档总入口（按问题找） | [docs/README.md](docs/README.md) |
| 架构现在长什么样、为什么这样设计 | [docs/design/README.md](docs/design/README.md) |
| 下一步做什么、做到哪了、过关条件 | [docs/plan/README.md](docs/plan/README.md) |
| 查过哪些文献、对项目的结论 | [docs/literature/README.md](docs/literature/README.md) |
| 跑过什么实验、结果如何 | [docs/experiments/README.md](docs/experiments/README.md) |
| 每次运行的原始结果 | [runs/](runs)（JSON，在实验记录里登记） |
| 当前代码 | [dspn_text.py](dspn_text.py)（模型与训练）、[hub.py](hub.py)（全局记忆中枢）、[evals.py](evals.py)（记忆评测）、[online.py](online.py)（在线学习与睡眠巩固）、[tests/](tests) |
| 旧代码（v0.12–v0.14，只读） | [legacy/](legacy) |
| 旧版设计文档（只读快照） | [docs/archive/](docs/archive) |

## 常用命令

```
conda activate dspn
python dspn_text.py train dspn --size S --data tiny --iters 3000   # gru / tf 自动对齐同档参数
python dspn_text.py sample dspn --size S --prompt "Once upon a time"
python -m pytest -q tests
python tools/check_docs.py
```

数据放在 data/（不入库）：TinyStoriesV2-GPT4 训练集与验证集；`input.txt` 为 tiny Shakespeare。
