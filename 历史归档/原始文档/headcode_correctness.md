# headcode_correctness.md — CPU vs NPU 正确性验证

> Sprint 3 Phase 4。固定 hidden 输入（TTS 实际解码产生的 768 维 F32 hidden state），
> CPU scalar matmul（Golden Reference）vs CANN NPU matmul（`OMNI_TTS_HEADCODE_NPU=1 OMNI_TTS_HEADCODE_VERIFY=1`）。

## 输入/权重规格

| 项 | 值 |
|---|---|
| 输入 hidden | float[768]，F32（llama_get_embeddings_ith 输出，CPU） |
| 权重 head_code.0.weight | [6562, 768] F32（GGUF 转置后 row-major 存储） |
| bias | 无 |
| 输出 logits | float[6562]，F32 |
| 验证方式 | 同一次调用内 CPU 与 NPU 双跑，逐元素对比 |

## 误差统计（~600 个固定 hidden 输入）

| 指标 | 值 |
|---|---|
| **max absolute error** | **~0.008**（0.0076 - 0.0090 区间） |
| **mean absolute error** | ~0.0015（0.0010 - 0.0021） |
| **RMSE** | ~0.002（0.0014 - 0.0026） |
| **top-1 agreement** | **100% MATCH**（全部 ~600 token） |
| **top-5 agreement** | **5/5**（586/587；1 个 4/5，边界 token） |

## 误差来源分析

- FP32 下 CPU（顺序累加）与 NPU（aclnnMatmul，可能分块/不同顺序累加）舍入顺序不同 → 微小差异。
- max_abs ~0.008 相对于 logits 量级（~±10-30）是 **~0.03-0.08% 相对误差**，可忽略。
- temperature=0.3 下误差放大 ~3.3×，但对 top-1 采样一致性无影响（top1 100% MATCH）。

## 采样一致性

- 采样使用 temperature/penalty/softmax/multinomial。
- top-1 logit 完全一致 → 高概率 token 采样结果稳定。
- 边界 token（top-5 第 5 位）有 1/587 次 4/5（第 5 名与第 6 名互换），但 top-1 不受影响。
- **音频质量影响**：见 Sprint 3 报告 Phase 8 的 WER/ASV 回归（20 样本 CPU vs NPU）。

## 结论

NPU head_code 与 CPU Golden Reference 数值一致（max_abs < 0.01），top-1 完全一致。
单变量实验正确性成立。
