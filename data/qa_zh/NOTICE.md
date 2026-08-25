# -*- coding: utf-8 -*-
"""数据来源说明（data/qa_zh/）

1. **MycoGuard 项目中文文案**（本作品集另一项目，MIT）：
   规则引擎 `projects/mycoguard/src/engine/mushroomEngine.ts` 的风险信号标签/详情、
   推理与处置建议（mushroomEngine.ts / expert.ts / scenarios.ts 的中文文本），
   作为中文蘑菇安全知识的种子语料。

2. **本仓库原创中文转写**（scripts/zh_content_*.py）：
   作者根据 `scripts/kb.py`（基于公开资料整理的英文事实，见 data/corpus/NOTICE.md）
   与公开蘑菇安全资料转写/撰写的中文物种事实卡、中毒综合征、安全规则、问答、
   急救要点。这些是**原创中文表述**，事实来自公开参考材料。

3. 数据用途：
   - `corpus_zh.txt`：中文继续预训练语料（可选步骤，scripts/build_zh.py 生成）。
   - `qa_train_zh.jsonl` / `qa_val_zh.jsonl`：中文指令微调问答对（150 条，135/15 拆分）。
   - `qa_eval_zh.jsonl`（由 scripts/build_zh_eval.py 生成）：30 条留出中文评测题，
     与训练问答对程序化确认不重合。

## 免责声明
本语料为蘑菇安全教育内容，**不是医疗建议，也不是采食建议**。蘑菇中毒可致命；
请务必以专家鉴定与中毒控制中心/医生的意见为准。据此训练的模型（NanoLM 中文分支）
仅作技术演示，切勿用于真实蘑菇辨识或医疗决策。
"""
