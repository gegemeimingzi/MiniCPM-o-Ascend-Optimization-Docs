
Sprint 3 单变量实验（head_code CPU scalar → CANN NPU）已完成并验证（910C，2026-08-15）：

**结果**：
- headcode 每 token：CPU 3.36ms → NPU **0.15ms**（p50 3.34→0.14，~24×）
- TTS 每 token：10.5 → **7.35ms（-30%）**
- eval-harness SPEAK→WAV RTF：CPU avg 0.83 → NPU avg **0.71（-14%）**
- 正确性：max_abs 0.008（FP32 舍入）、**top1 100% MATCH、top5 5/5**
- 精度：TTS-Seed 20 样本 WER CPU 1.97% vs NPU **1.35%（无回归）**
- hidden D2H 未消失（~0.26ms，版本 1 走 llama_get_embeddings_ith）；logits D2H 新增 ~0.05ms 可忽略

**实现**：env `OMNI_TTS_HEADCODE_NPU=1`（默认关闭=CPU，一键回滚）。复用 projector_forward 的 ggml 图 + CANN backend 模式：weight 一次性转置上传（[6562,768]→ggml [768,6562]），每 token 建极小 mul_mat 图。`OMNI_TTS_HEADCODE_VERIFY=1` 双跑对比。文件：omni.h(headcode_npu_model) + omni.cpp(headcode_npu_init/forward + tts_headcode_compute dispatch)。失败自动回退 CPU。

**注意**：Sprint 2 的 25ms decode / 5.7ms headcode 数据受机器 load 91+ 污染；空闲态真实为 decode 6.5ms / headcode CPU 3.36ms。TTS decode 不是 launch 主导（空闲态 kernel 为主）。

报告：`/workspace/competition/sprint3_report_headcode_npu.md` + `headcode_correctness.md`。相关：[[sprint2-tts-t2w-findings]] [[minicpm-competition-progress]]
