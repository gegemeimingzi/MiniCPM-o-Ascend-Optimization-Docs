
Sprint 11 T2M K/V cache slot-write 微实验（910B3，2026-08-16，Task 7 正式化）：

**设计**：只改 `fm_attn_build_new_att_cache`，`ggml_concat(k_perm,v_perm,0)` → 2×`ggml_set_inplace`（env `OMNI_T2M_SLOT_WRITE=1` 默认 off）。图可达性：`ggml_set_inplace` 返回 base view（data 保持 slot_buf 基址）→ 链式 s1(k)→s2(v) 两写均可达，返回 [D2,T,H,B] 从 slot[0] 起。关键：k_heads=k_total=[D,H,T,B]（ne1=H, ne2=T）。

**结果**：
- **Graph diff 精确**：-160 CONCAT、+320 SET、+160 节点（16×10，1 concat→2 SET）
- **Correctness 4/4 DIFFER**（硬门失败）；diagnostic mode-2（建 SET 返 concat）= base IDENTICAL → 差异 100% 来自执行的 SET 写
- **根因（独立探针证实）**：`aclnnInplaceCopy` 对 **strided dst 写入错误**（bad=7168/8192，t=0 对 t≥1 错位）；[D2,T,H,B] packed 布局 k/v 沿 dim0 交错写本质 strided，4D ggml 无法用 [2,D,T,H,B] 避
- **dispatch**：SET 13.7μs/op > CONCAT 10.7μs/op，净 +2.66ms/chunk（2 SET 换 1 concat 更贵）
- **性能**：t2m 222.99→230.34ms（+3.3%）、RTF 0.3566→0.3732（+4.7%）
- **成功条件 4 项全不满足** → FAIL

**已回退**：恢复 B-lite（0 处 OMNI_T2M_SLOT_WRITE，392 处 ggml_cont），重编译验证 t2m 图 13980 节点/CONCAT 1681/无 SET（基线精确复现）。**build 保持 B-lite 最优态**。

**机制结论**：direct-slot-write 经 CANN SET 不可行（InplaceCopy strided 缺陷）；若继续 in-place 必须改 CANN 后端（被禁）或改 cache 布局（= 完整 Candidate A，难度×2）。印证 Sprint 10 高难高风险判断。下一步 **T2W 非-cache 路线**。

**探针教训**：CANN executor 严格单次使用（复用崩）；aclrtMalloc 返回设备内存宿主不可直接写（段错误）；standalone aclInit/device 在部分进程挂起。报告 `/workspace/competition/sprint11_kv_slot_write_report.md`。相关：[[sprint10-cache-inplace-analysis]] [[minicpm-competition-progress]]
