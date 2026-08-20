
Sprint 9 vocoder B-lite 单变量实验（910，2026-08-16）：

**Phase 1 源码验证（4 候选）**：
- **im2col_2d cont**：中置信（reshape [K*Cin,T,B]→[K*Cin,T*B] 合并末两维，但 CANN im2col post-process permute 后布局需确认）
- **weight cont**：中低置信（[K,Cin,Cout]→[K*Cin,Cout] 合并首两维，view 连续性不确定）
- **bias cont**：**高置信冗余**（[Cout]→[1,Cout,1] 连续 view）
- **y_tcb cont**：**低置信/必需**（[Cout,T*B]→[T,Cout,B] 是转置，nb1=Cout*4≠T*4 → cont 物化转置，不可删）

**实验**：
- A（weight/bias 6 处）：**4/4 bit-identical**，voc.compute 108.9→108.1（-0.7% 噪声内）
- B（+im2col_2d 3 处）：**4/4 bit-identical**，voc.compute 110.6（无改善/略差）

**结论**：**vocoder B-lite 不值得**。vocoder 图仅 ~2321 节点，可删 6-9 cont（0.3-0.4%）→ dispatch 收益可忽略；im2col_2d/weight 视图可能非连续 → mul_mat 内部拷贝抵消。对比 t2m（14320 节点、删 340 = 2.4% → -7.7%），vocoder 太小。**收益与图规模成正比。**

**已回退**：git checkout 恢复原始 + 重新应用 B-lite → build 确认 13980 节点（B-lite 最优态）。Sprint 9 中 revert 脚本曾误加 cont（同行使句 + 多 t2m 站点），已用 git 恢复解决。

**教训**：vocoder 优化杠杆不在 B-lite 型 CONT；下一步选项 = t2m cache in-place（去 CONCAT ~1000+ 节点）或 IM2COL kernel（被禁）。

报告：`/workspace/competition/sprint9_vocoder_blite_report.md`。相关：[[sprint8-b-mid-feasibility]] [[sprint7-b-lite-result]] [[minicpm-competition-progress]]
