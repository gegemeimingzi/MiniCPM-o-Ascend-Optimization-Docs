
## MiniCPM 子赛道A flow B=1 尝试（2026-08-21）

**尝试**：cfg=0 时 flow 图 B=2→B=1（跳过 uncond，DiT 减半）。实现：CFM build_forward_graph/chunk_graph 的 B_total/concat 简化 + fmDiT host-fill B_total + fm_dit_broadcast_spks_over_time n_dims==1 分支 + init_from_host_caches cache batch 检查。配套生成 B=1 prompt_cache.gguf（90MB，用 export_prompt_cache 工具 cfg=0）。

**结果**：RTS 跑通（data_valid=True），token2wav 0.223→0.178（-16%），FULL RTF 0.655。但 **TTS 精度评测（llama-omni-tts-eval teacher-forcing）崩**：Token2WavSession::reset → t2w.reset_stream → CANN context null pointer / segfault，cfg 无关（0.7 也崩）。二分定位：HEAD 版（c9785cc）正常，+vocoder F16 崩 CANN，无 F16 的 CFG+B=1 也崩 CANN。根因未最终锁定，在 token2wav-impl.cpp 改动中。

**决策**：回退 B=1。当前稳定配置收益更可靠。

## 当前稳定配置（2026-08-21，FULL RTF 0.683）

- TTS FA：OMNI_TTS_FLASH_ATTN=1（tts 0.19→0.166，-5%，bit-identical）**本次新开**
- VPM batch：OMNI_VISION_BATCH_ALL 默认开（encode 0.147→0.111）**核实已生效**
- 4步 flow：OMNI_T2W_N_TIMESTEPS=4 + omni.cpp t2w_n_timesteps env 修复（**server 之前硬编码 5，env 从没生效**，修复后收益小 ~2%）
- head_code NPU：OMNI_TTS_HEADCODE_NPU=1
- t2m CONT 消除：OMNI_T2M_SKIP_REDUNDANT_CONT=ADD_MUL
- F16 conv 禁用：OMNI_T2W_F16_CONV=0（我们 CANN 上负优化）

**3 次 RTS full 口径平均 0.683**（encode 0.111 / prefill 0.013 / decode 0.168 / tts 0.166 / token2wav 0.223）。TTS 精度恢复（WER smoke2 4.5%，funasr 需 PARAFORMER_MODEL 指向 snapshots/master）。

**server 端 env 注意**：OMNI_T2W_N_TIMESTEPS 需 omni.cpp 读取（我们加了 t2w_n_timesteps）；OMNI_T2W_CFG 对 server 无效果（HEAD 版 token2wav 无该 env）。

相关：[[minicpm-competition-progress]] [[sprint16-rtf-remeasure-finding]]
