
Sprint 13（910B3，2026-08-16）：源码层冗余 CONT 消除实验。**禁止 post-build graph mutation** → 新增 `ggml_cont_if_needed(ctx, src)` helper，全部 222 个 `ggml_cont(ctx,X)` 替换，`src->op∈{ADD,MUL,CONCAT}` + `ggml_is_contiguous` 限定，env-gated（默认 off 零回归）。

**核心结论**：
- **ADD→CONT(175)+MUL→CONT(160)=335 个安全**：bit-identical ✓（4/4 SAME，TTS 确定性单独验证），t2m CONT 2721→2365，t2m.compute 229.6→220.6ms（-3.91%），RTF 0.374→~0.354（-2.5~5%）。**采纳**。
- **CONCAT→CONT(170) 不安全**：跳过改变数值（26% 样本 DIFF，max|diff| 0.28-0.42）——Sprint 12 判据（producer 连续+形状匹配）对 CONCAT 失效，推测 CANN concat 输出 buffer 别名/布局依赖。**必须排除**。
- patch OFF（env 默认）与 B-lite 逐项一致（nodes 13980/CONT 2721/vocoder 756）。
- per-CONT 成本：dispatch 13.2μs（GWS 4.8 + alloc 0.1 + launch 8.3），实测综合 ~26.8μs/CONT（含 kernel+拓扑）。

**环境值**：`OMNI_T2M_SKIP_REDUNDANT_CONT=ADD_MUL` 启用优化；ADD/MUL/CONCAT/ALL 为分型值。当前服务器 build-prof = patch 版（env off = B-lite）。报告 `/workspace/competition/sprint13_redundant_cont_report.md`。

**决策点**：保留 patch（默认 off 零回归，可选启用）。若要常驻优化：评测脚本 export `OMNI_T2M_SKIP_REDUNDANT_CONT=ADD_MUL`。相关：[[sprint12-graph-dispatch-analysis]] [[sprint7-b-lite-result]] [[minicpm-competition-progress]]
