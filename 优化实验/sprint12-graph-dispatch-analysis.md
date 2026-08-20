
Sprint 12 T2W 非-cache Graph/Dispatch 分析（910B3，2026-08-16，纯分析零优化）。数据：OMNI_T2W_DUMP_GRAPH 插桩 dump 22110 节点全图 + op-prof + launch-prof。

**Task 1 核心发现：B-lite 后仍有 537 冗余 CONT**（producer 已连续 s0c=1 + 形状完全匹配 0 错位）：
- ADD 175 + CONCAT 170 + MUL 160 + VIEW 15 + RESHAPE 12 + 其他 5
- 主组：[512,50,2,1]×320（DiT modulate/residual 输出，ADD/MUL→CONT→CONCAT）+ [1024,2,2,1]×160（CONCAT→CONT→RESHAPE）
- 源码：编码器 ue_uce_make_valid_mask(L4449)/build_right_pad_zeros_ctb(L4468)/prelook(L3989/4023) + flow matching block 路径

**Task 2 dispatch 排名**（[opg] wall，含部分 kernel）：IM2COL 36.2%（被禁）/ADD 13%/CONT 12.2%（9.38μs/op）/CONCAT 8.7%/MUL_MAT 8.5%/MUL 8.3%/NORM 7.1%。launch-prof：GWS 3μs+alloc 0.1μs+launch 7.2μs≈10.3μs/op。

**Task 3**：图在 setup_cache 构建一次并复用（L7787/7818，cache 作输入 T 固定）→ 无每 chunk 重建；但每 dispatch 重复 createTensor+GWS（executor 单次使用不可复用）。descriptor+GWS 缓存理论省 40-55ms/chunk（18-25%）但需改 CANN backend（**被禁**）。

**候选**：A=删 537 冗余 CONT（推荐，-537 节点/dispatch，t2m -2.4%~-12%，RTF -0.01~-0.05）；B=链折叠（=A 子集 170）；C=descriptor 复用（禁）；D=小算子合并（ARN 融合已证伪精度差）。

**微实验设计（Sprint 13）**：`OMNI_T2M_SKIP_REDUNDANT_CONT=1` —— t2m 图构建后把「producer 连续+同形状」CONT 原地转 VIEW（`n->op=GGML_OP_VIEW; n->data=src0->data`），单点 env-gated，bit-identical 门，覆盖全部 537。若失败回退→源码级 gating 编码器站点。

报告 `/workspace/competition/sprint12_graph_dispatch_analysis.md`。**build 保持 B-lite（13980 节点验证）**。相关：[[sprint7-b-lite-result]] [[sprint11-kv-slot-write-result]] [[minicpm-competition-progress]]
