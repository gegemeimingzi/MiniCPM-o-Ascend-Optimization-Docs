# tts_fa_graph_diff.md — TTS Decode Graph：Baseline vs FlashAttention

> Sprint 5 Step 2。`GGML_CANN_OP_PROF=1` 捕获，1 个固定样本（seed 42 / temp 0.3 / teacher-forcing 空操作），
> build-prof 同一二进制，仅 `OMNI_TTS_FLASH_ATTN` 开关切换。
> 样本：TTS-Seed zh `10002287-00000094.wav`（"简单地说，这相当于惠普把消费领域市场拱手相让了。"）

---

## 1. 图结构总览（TTS decode 步图）

| 配置 | 图 shape | 次数 | 节点数 | FLASH_ATTN_EXT | SOFT_MAX | CONT | ROPE | SET_ROWS | MUL_MAT |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline** (vanilla) | 203 节点 | 154 | 203 | **0** | **9** | 9 | 18 | 18 | 82 |
| **Baseline** (vanilla) | 242 节点 | 154 | 242 | **0** | **11** | 11 | 22 | 22 | 99 |
| **FA** (ENABLED) | 176 节点 | 145 | 176 | **9** | **0** | **0** | 18 | 18 | 64 |
| **FA** (ENABLED) | 209 节点 | 145 | 209 | **11** | **0** | **0** | 22 | 22 | 77 |

- 两种节点数（203/242 → 176/209）对应不同 n_past 桶的图变体（KV 长度变化）。
- **开启 FA 后：`FLASH_ATTN_EXT` 出现（9/11 次/图），`SOFT_MAX` 与 `CONT` 完全消失。**
- ROPE 与 SET_ROWS 保留（FA 仍需要 RoPE 预处理与 KV 行写入，与预期一致）。
- MUL_MAT 减少（82→64, 99→77）：QKᵀ/PV 注意力矩阵乘被融合进 FLASH_ATTN_EXT。
- 节点数减少：203→176（-27）、242→209（-33）。

## 2. 单图完整 op 构成（opg 行原文，op-prof 计时含测量开销）

**Baseline 242 节点**：`nodes=242 wall=10.39ms GLU=11(0.19ms) ADD=22(0.20ms) CONT=11(0.30ms) SOFT_MAX=11(0.40ms) ROPE=22(4.32ms) MUL_MAT=99(2.36ms) SET_ROWS=22(0.67ms) MUL=22(0.25ms) RMS_NORM=22(1.62ms)`

**Baseline 203 节点**：`nodes=203 wall=11.06ms GLU=9(0.11ms) ADD=18(0.14ms) CONT=9(0.20ms) SOFT_MAX=9(0.30ms) GET_ROWS=2(0.06ms) ROPE=18(0.51ms) MUL_MAT=82(7.62ms) SET_ROWS=18(0.54ms) MUL=19(0.16ms) RMS_NORM=19(1.36ms)`

**FA 209 节点**：`nodes=209 wall=11.23ms GLU=11(0.23ms) ADD=22(0.22ms) ROPE=22(4.66ms) FLASH_ATTN_EXT=11(1.05ms) MUL_MAT=77(1.72ms) SET_ROWS=22(0.69ms) MUL=22(0.31ms) RMS_NORM=22(2.26ms)`

**FA 176 节点**：`nodes=176 wall=8.74ms GLU=9(0.12ms) ADD=18(0.15ms) GET_ROWS=2(0.06ms) ROPE=18(0.45ms) FLASH_ATTN_EXT=9(0.55ms) MUL_MAT=64(5.33ms) SET_ROWS=18(0.51ms) MUL=19(0.15ms) RMS_NORM=19(1.36ms)`

> ⚠️ opg wall 被 op-prof 逐 op 计时膨胀，不可用于 latency 结论；真实 per-token decode 见 Sprint 5 Step 3 的 `OMNI_TTS_PROF` 数据。

## 3. 关键差异总结

| 项 | Baseline | FA | 差异 |
|---|---|---|---|
| FLASH_ATTN_EXT | 0 | 9/11 | 新增（FA kernel 生效） |
| SOFT_MAX | 9/11 | 0 | **消失**（被 FA 融合） |
| CONT | 9/11 | 0 | **消失** |
| MUL_MAT | 82/99 | 64/77 | 减少（QK/PV 融合） |
| ROPE | 18/22 | 18/22 | 不变（FA 仍需 RoPE） |
| SET_ROWS | 18/22 | 18/22 | 不变（FA 仍需 KV 写入） |
| 节点数 | 203/242 | 176/209 | -27/-33 |

## 4. 验证信号

- FA run 的日志**无** `llama_init_from_model: flash_attn is not compatible with CANN - forcing off` 警告（ENABLED 绕过 AUTO 启发式）。
- Baseline run 有该警告（AUTO → CANN 强制 DISABLED），确认 baseline 是 vanilla。
- t2m flow-matching DiT 的 6 个 FLASH_ATTN_EXT 图（396/437 节点，17/19 次）两个配置都存在 → CANN FA kernel 本就可跑，与 Sprint 4 结论一致。

**结论**：TTS decode 图结构确认 FA 生效——`FLASH_ATTN_EXT` 替代 `SOFT_MAX`/`CONT` 并融合 QK/PV matmul，图节点减少 ~13%。ROPE/SET_ROWS 不消失（预期），因此 FA 收益主要来自 SOFT_MAX+CONT 的消除（Step 3 实测 decode -10%）。
