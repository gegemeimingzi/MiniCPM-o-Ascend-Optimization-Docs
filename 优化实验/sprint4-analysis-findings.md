
Sprint 4（纯分析，910C，2026-08-15）四问结论：

**Q1 TTS FlashAttention**：**值得做，首选**。
- 根因：`llama-context.cpp:3400` 仅当 `flash_attn_type==AUTO` 且 CANN 后端时强制 DISABLED（数值稳定性顾虑 for long/multi-image shapes）。
- 主 LLM 被显式 ENABLED（omni.cpp:4418），**TTS context 继承 AUTO 从未覆盖**（omni.cpp:4452 用 `common_context_params_to_llama(*params)`）。
- CANN FA kernel 存在且能跑：t2w flow-matching DiT（19 层 FLASH_ATTN_EXT 图）在 CANN 上成功运行。
- TTS decode 实测是 vanilla（203/242 节点 SOFT_MAX 图，每 token 1 个）。
- **修复只需 1 行**（TTS context 设 ENABLED）；预计 decode -30~-40%（6.5→~4ms/token）→ RTF -0.04~-0.05。
- 风险：llama-context 注释的数值问题，但 TTS 短序列单 token 非 long/multi-image；需精度验证。
- Sprint 1 "TTS FA 噪声" 结论可疑（很可能当时根本没给 TTS context 开 FA）。

**Q2 SET_ROWS**：20 次/step（n_layer×2），0.264ms/op（load），idle ~1.3ms/step（decode 20%）。根因：per-layer×head 循环多次 aclnn。批量化 -0.02~-0.03 RTF。**FA（Q1）一次消除 SET_ROWS+ROPE+SOFT_MAX 更好**。

**Q3 headcode V2**：**NOT WORTH IT**。省 d2h(0.25)+h2d(0.05)≈0.3ms/token（4%），需改 llama.cpp embeddings 物化路径（中-大规模）。

**Q4 t2m.compute**：双态。短输入 **launch-bound**（14K op 图 dispatch ~213ms ≈ compute 224ms）；长样本 **compute-bound**（752ms = dispatch ~215 + kernel ~537，kernel 71%）。IM2COL 41%（conv encoder）+ CONT/CONCAT（内存 op）。下一步：减少 t2m 图 op 数（CONCAT/CONT 融合），launch+compute 双收益。

**执行顺序**：#1 TTS FA → 验证数值；若不行退 #2 SET_ROWS 批量化；T2W 融合后续。相关：[[sprint3-headcode-npu-result]] [[minicpm-competition-progress]]
