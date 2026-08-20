
Sprint 2 (TTS/T2W 深度 profiling) 实测结论（910C，build-prof 插桩，2026-08-15）：

**TTS 每 token ≈ 25ms**（机器空闲）：decode/llama_decode 74%（launch wall 5ms + 小 kernel：MUL_MAT 37%、SET_ROWS 20%、ROPE 12%、vanilla-attn SOFT_MAX 8%）、headcode CPU 标量 matmul 15%（5.7ms）、d2h 9%、sampling 2%。
- TTS 用 **vanilla attention**（`flash_attn is not compatible with CANN - forcing off`）→ SET_ROWS/ROPE/SOFT_MAX 每层多 ~12ms/step。
- aclnn launch wall：TTS 场景 **launch 70% + GetWorkspaceSize 29%**（per-op 55μs）；LLM prefill 是 GetWorkspaceSize 主导（72%）。

**T2W 每调用 ~950ms**：**t2m.compute(flow-matching) 79%**（752ms）+ vocoder 19%（164ms）。NPU compute bound。
- T2W A/B：gpu:0 RTF 0.39 vs cpu ~12（**cpu 慢 30×**）→ 必须 NPU。

**Task 9 decode 复核**：LLM decode ~22ms/step（logits D2H+sync ~5ms 可变、sampling 0.25ms 可忽略）。

**管线**：t2w queue_wait=0.0（T2W 等 TTS，TTS 是慢侧）；队列无界，overlap 已最大化。**官方 RTF 对 overlap 免疫**（stages 相加，只有 encode=max(vpm,apm) 记并行）。

**推荐优化**：**headcode matmul 移到 NPU**（消除 headcode 5.7ms + d2h 3.2ms/step ≈ TTS 24%，RTF -0.05~-0.06，低风险）。次选 flow-matching 图级分析（t2m.compute 79%）。

插桩：env 开关（OMNI_TTS_PROF / GGML_CANN_OP_PROF / GGML_CANN_LAUNCH_PROF / OMNI_T2W_PROFILE / OMNI_DECODE_PROF），代码在 tools/omni/sp2_prof.h + omni.cpp + ggml-cann。测量脚本在 /tmp/sp2_*.sh（服务器），报告 sprint2_tts_t2w_analysis.md。相关：[[sprint2-queue-overlap-finding]] [[minicpm-competition-progress]]
