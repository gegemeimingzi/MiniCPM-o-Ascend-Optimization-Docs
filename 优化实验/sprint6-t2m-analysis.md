
Sprint 6（纯分析，910，2026-08-15）t2m.compute 深度拆解结论：

**核心发现**：t2m 图固定 ~14320 个 NPU 执行节点，**每 chunk（28 token）compute ≈ 225ms，dispatch 恒定 ~183ms 主导**（与输入长度无关）。T_chunk_token 固定 28 → 每 chunk kernel 近似固定 → **短/长输入都最受益于「减少 op 数」**（降 dispatch），非 kernel 融合。修正 Sprint 2「长输入 compute-bound」：那是长文本单 chunk，本管线 chunk 固定。

**op 统计**（稳态 14320 节点，每 compute ~225ms）：
- **IM2COL 323（77-94ms，37%，0.26ms/op kernel 主导）** — 必要 conv compute（A），无 native conv kernel（CANN 只有 aclnnIm2col），周围 permute+cont 是浪费
- CONT 3061（26-31ms，~10μs/op dispatch 主导）；CONCAT 1681（17-20ms）；ADD 3422；MUL_MAT 1597；NORM 993
- launch-prof：dispatch ~12.8μs/op（GetWorkspace 4.9 + launch 7.9）

**CONCAT**：K/V 增长(624/625)×16块×10步 + cache打包(446) + conv cache(1105/1120/1219) + step-pack tree(983/1018, 已O(NlogN)) + CFG(933) + 通道(1532)。**无 mel concat**（mel 用 ggml_add 就地积分 + 主机 memcpy append）。

**CONT**（G1-G7）：**G4 concat后（14处）+ G3 linear后（4处）是纯拷贝可删**（concat/mul_mat 输出必连续，同 1.C fix at line 446 已验证）；G1 注意力 head permute ~480 节点布局必须；G6 conv permute ~320-640 需 im2col 改 layout；G7 encoder 边界 ~50-100。

**推荐**：
- **P0 = B-lite**：删 G4+G3 冗余 `ggml_cont`（-330~400 节点，-4~5ms，t2m -2%，SPEAK→WAV -0.01）。低难度零风险确定性。
- P1 = cache in-place（去 concat ~1000+ 节点）+ conv layout（-480~640）→ -0.02~-0.03。
- P2 = CANN fused conv kernel（省 IM2COL kernel 40-60ms）→ -0.03~-0.05，高难度。

t2m RTF 贡献 0.243（1.38s/5.68s）。报告：`/workspace/competition/sprint6_t2w_analysis.md`。相关：[[sprint5-tts-flash-attn-result]] [[sprint4-analysis-findings]] [[minicpm-competition-progress]]
