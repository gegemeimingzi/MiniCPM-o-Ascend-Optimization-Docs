
Sprint 8 B-mid 可行性实验（910，2026-08-15）：

**Phase 1（源码确认）**：ggml_im2col 1D 需 **TCB 布局**（IW=ne0/IC=ne1）+ 连续（CPU assert nb10==4）；**CTB 直喂不可行**（维度错位）。CANN concat 接受非连续输入（aclnnConcat 实际 strides）。DiT causal conv（fmCausalConv1d chunk）的 permute→cont（原1102/1104）**可删**——后接 concat 物化连续 TCB；encoder conv（3927-3928）无 concat 前接 im2col → **不可删**。

**Phase 2/3（实测）**：删 1112/1114 → **4/4 样本 bit-identical**、图 13980→13340（-640）、CONT 2721→2081（-640）。但**性能为负**：t2m.compute 228.4→234.8ms（+2.8%）、RTF 0.570→0.582。根因：concat 读非连续 view（strided gather）内部物化，dispatch 省 ~6.4ms 被 concat 变慢抵消。

**结论**：
- A. B-mid 部分可行但**性能为负**，不采纳（已回退，build 恢复 B-lite 13980 节点）
- B. 最小修改=删 1112/1114 两个 ggml_cont（已验证+回退）
- C. 单 block：-640 节点、-640 CONT，但 t2m +6ms、RTF +0.012
- D. 全量 -640 节点/compute，但无端到端益
- E. **不值得 Sprint 9**（当前 concat 物化方式）；除非改 CANN strided concat 或上游产 TCB（超范围）
- F. **vocoder 有低风险 B-lite 型**：hg2 conv 的 reshape→cont 冗余（w_2d/y_tcb/b_1c1/im2col_2d，连续 tensor reshape 是连续 view），~4×N conv，预计 vocoder -1~-2%

报告：`/workspace/competition/sprint8_b_mid_feasibility.md` + `vocoder_layout_candidates.md`。相关：[[sprint7-b-lite-result]] [[sprint6-t2m-analysis]] [[minicpm-competition-progress]]
