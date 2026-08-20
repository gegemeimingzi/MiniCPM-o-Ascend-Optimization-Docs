
Sprint 7 B-lite 单变量实验（910，2026-08-15）完成，因果链全部验证：

**改动**：token2wav-impl.cpp 删 14 处冗余 `ggml_cont`（G4 concat 后 12 处 + G3 linear 后 2 处条件式）。1121 是最大单项（DiT 16块×10步×2conv = -320 节点）。字节 456929→456666。

**结果（受控 4 样本 A/B，5 次背靠背）**：
- 图 14320→13980 节点（-340），**只有 CONT 3061→2721 变**
- t2m.compute 244.3→225.4ms（**-7.7%**）；launch wall 9352→8598ms/run（-8.1%）
- SPEAK→WAV RTF **0.605→0.564（-6.8%/-0.041）**
- **正确性：4/4 样本 WAV bit-identical（同 MD5）** → WER/ASV 必然无回归
- 跨会话 20 样本：0.714(Sprint5)→0.702（-1.7%，受负载漂移污染）

**ROI**：340 节点 → -18.9ms/compute；每 1000 节点 ≈ -56ms/compute、RTF -0.12（线性外推）。**实测 ~82μs/op（含 copy kernel+分配），是 Sprint 6 预测 12.8μs 的 4.3 倍** — CONT 在 CANN 上是 aclnnCopy（非纯 dispatch）。

**结论**：
- A. B-lite 收益：t2m -7.7%、RTF -6.8%（受控）；正确性逐位不变
- B. 每 1000 节点：-56ms/compute、RTF -0.03~-0.12（边际递减）
- C. 下一步 **B-mid**（conv 前 permute+cont 消除，~640 节点，同款低风险模式）先于 cache in-place（高难）
- D. 0.71→0.60：需 B-lite + B-mid + cache in-place（或 vocoder）组合；仅 B-lite 到 ~0.69-0.70

报告：`/workspace/competition/sprint7_b_lite_report.md`。相关：[[sprint6-t2m-analysis]] [[sprint5-tts-flash-attn-result]] [[minicpm-competition-progress]]
