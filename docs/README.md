# 文档总入口

每类信息只有一个地方。按问题找入口，进去后每个文件夹的 README 是下一级目录。

| 我想知道 | 去哪 |
|---|---|
| 现在在做什么、下一步、正在跑什么 | [plan/README.md](plan/README.md) 的“当前状态” |
| demo 各阶段过关没有 | [plan/stages.md](plan/stages.md) |
| 某篇文献 / 某个想法决定采不采纳、排在什么时候 | [plan/adoption.md](plan/adoption.md) |
| demo 之后的研究线 | [plan/research-lines.md](plan/research-lines.md) |
| 验收项编号是什么意思 | [plan/acceptance.md](plan/acceptance.md) |
| 架构现在长什么样、为什么 | [design/README.md](design/README.md) → current/ |
| 有哪些候选新方案（第 9–19 章，均未实验；可借鉴的机制见 plan/v2.md） | [design/README.md](design/README.md) → candidates/ |
| 某个问题做过什么实验、结论 | [experiments/README.md](experiments/README.md) 的“按问题查” |
| 查过哪些文献、对项目的结论 | [literature/README.md](literature/README.md) |
| 某个方法具体怎么实现、代码在哪 | [literature/README.md](literature/README.md) → cards/ |
| 旧版设计（只读） | [archive/](archive) |

## 文件夹结构

```
docs/
  plan/          排期（唯一施工入口）：README=当前状态与下一步；stages / adoption / research-lines / acceptance
  design/        设计：current/ 现行架构，basis/ 依据与讨论，candidates/ 候选方案，changelog.md 版本改动
  experiments/   实验记录：按编号分段，README 有“按问题查”
  literature/    文献：topics/ 按主题，cards/ 实现卡
  archive/       旧版设计快照（只读）
```

写文档的规矩见仓库根目录 CLAUDE.md；`python tools/check_docs.py` 查断链、漏登记、过长文件和过长段落。
