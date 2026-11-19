@README.md

## 文档规矩（给代理）

- 文档总入口 docs/README.md；每个文件夹的 README.md 是它的目录，新增、移动文件时同一次改动里更新目录。
- 设计改动只写 docs/design/ 对应章节（current/ 现行架构，basis/ 依据与讨论，candidates/ 候选方案；版本改动记在 changelog.md）；顺序、状态、门槛只写 docs/plan/（当前状态与下一步只在 plan/README.md）；结果只写 docs/experiments/（新结果追加到最新分段文件，并在 README 的“按问题查”登记编号）；文献条目写 docs/literature/topics/，实现卡写 docs/literature/cards/。不另建总结、笔记或新版本文件。
- 文档变长（检查脚本报 WARN long）就按章节拆成多个文件，并在所在文件夹的 README.md 目录里登记；一段写成超长一行（WARN line）就改成要点。
- 设计有大改时，先把 docs/design/ 整个复制到 docs/archive/design_v<旧版本>/，再改并更新 docs/design/README.md 标题里的版本号。
- 每次运行的 JSON 存进 runs/，同一次改动里在 docs/experiments/ 最新分段文件登记一行并链接它。
- 设计文档引用实验用编号（E01…），不复制数字。
- docs/archive/ 和 legacy/ 只读。
- 改完文档或登记实验后跑 `python tools/check_docs.py`，红了不算完成。
- Python 用 conda 环境 dspn：Z:\miniconda\envs\dspn\python.exe。GPU 同一时间只跑一个训练任务。
