
Sprint 10 T2M cache in-place 深度分析（910，2026-08-16，纯分析零实现）：

**CONCAT 1681 拆解**：SAFE to in-place ~1120（K/V growth 624/625=320 + cache build 446=160 + conv new-cache 1120/1219=480 + step-pack tree 983/1018=160）；MAYBE 320（im2col 输入 1105，B-mid 已证 strided 变慢 → KEEP）；MUST KEEP ~240（CFG 933/通道 1532/encoder pad/rel-pos）。

**K/V cache 生命周期**（详见 t2m_kv_cache_lifecycle.md）：
- 全尺寸预分配（setup_cache 7727 dup + ggml_set_output）
- write offset = slot·H·nb2（fm_cfm_view_att_cache_packed）
- **attention concat（624/625）是 O(T²) 重拷贝**（每 step 重新 concat 完整增长上下文）——in-place 的真实价值点
- **new cache [D2,T,H,B] 布局 = persistent slot [D2,T,H,B]**（permute 0,2,1,3 产生）→ 直接 slot 写是连续写，无 strided
- conv cache [C,pad,B] slot ne0 偏移，同样连续

**tree-concat 为什么**：性能优化（O(N²)→O(N log N) 拷贝），源码注释确认（~978-984），非布局必需。

**候选**：A K/V slot（-640 节点，RTF -0.015~-0.025，高难高风险）；B conv slot（-480）；C/D=A 子集。

**收益模型（保守）**：best -10~-12% t2m / expected(A only) -5~-7% / worst net~0（B-mid 同款机制风险）。

**决策**：**DO LATER**——无低风险 candidate；不为优化 CONCAT 重构 cache。唯一值得下一步 = **Task 7 微实验**（只改 fm_attn_build_new_att_cache 单点，env 开关默认 off，bit-identical 门），证明 direct-write 机制后再考虑 A。build 保持 B-lite 最优态（RTF 0.564/0.702）。

报告：`/workspace/competition/sprint10_cache_inplace_analysis.md` + `t2m_kv_cache_lifecycle.md`。相关：[[sprint9-vocoder-blite-result]] [[sprint7-b-lite-result]] [[minicpm-competition-progress]]
