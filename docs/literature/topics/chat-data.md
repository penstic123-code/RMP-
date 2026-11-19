# 小型对话模型与对话数据

“已核实”= 代理打开原文或 arXiv 摘要页确认；“未核实”= 只看了搜索片段或二手来源。检索日期 2026-11-19。目录与总结论见 [README](../README.md)。

### 小型对话模型

- [TinyDialogues](https://arxiv.org/abs/2408.03617)（EMNLP 2024）：合成儿童导向对话；对话内顺序重要，全局课程无用。
- TinyChat15M（社区博客，[链接](https://nikhilr.io/posts/TinyChat15M/)）：GPT-4o-mini 按 BASIC English 生成 100 万段短对话训 15M 模型，能简单闲聊；必须标出轮次边界，否则会替用户续写。
- [Dialogue Is Not Enough](https://arxiv.org/abs/2510.20358)（BabyLM 2025）：只用对话预训练，对话续写好、通用差；DPO 有帮助。
- [SmolLM2 / smol-smoltalk](https://arxiv.org/abs/2502.02737)：给小模型删减过的指令数据，对 ≤10M 仍偏难。
- 含义：本项目对话数据（SODA + 模板记忆对话）语言偏成人，后续可补受限词汇合成对话；轮次已用“User:/Bot:”与 SEP 标出。
