
MiniCPM 昇腾挑战赛子赛道 A 提交前 RTF 复测（2026-08-20），发现并修复两处问题：

**① stale binary**：`config.env` 的 `OMNI_SERVER_BIN` 一直指向 `build/bin/llama-omni-server`（libomni.so 08-15，缺 T2W B-lite/ADD-MUL），最终候选应为 `build-prof/bin`（libomni.so 08-16 20:09）。之前提交的官方 RTF 0.852/0.858 是过期二进制测得。已改三处 config.env。

**② `run_eval.py task_rts` 强制 `GGML_CANN_ACL_GRAPH=off`（+WEIGHT_NZ=off）会使双工轨迹退化为 8 SPEAK/4 轮**（正常 27 SPEAK/5 轮），`rtf.core` 只剩 2 帧不可测。修复：run_eval.py task_rts 的 server env 固定 ACL_GRAPH/WEIGHT_NZ="on"（精度任务仍走 config.env 的 off）。注：sed 无范围会误改全部 task env——必须限定在 task_rts 函数内。

**候选 env（RTS）**：`OMNI_TTS_FLASH_ATTN=0`（FA 与 ACL_GRAPH 交互会改流式轨迹；FA 精度中性 Sprint5）+ `OMNI_TTS_HEADCODE_NPU=1` + `OMNI_T2M_SKIP_REDUNDANT_CONT=ADD_MUL`。官方口径 = judge 双工 `rtf.core.rtf_aggregate`（掐头去尾），0.6823 是 TTS-eval 简化口径（非官方主指标）。

**结果**：core RTF 0.7633/0.7516/0.763/0.7696 → **0.762（-29.9% vs 官方 1.087）**，27 SPEAK/5 轮正常。提交包已更新：`/workspace/competition/submission_llamacpp_omni_20260820.tar.gz`，README 头条 0.762，COMPETITION_SUMMARY 增 §10 Sprint16。

**Why**：避免下次提交再踩 stale binary 和 ACL_GRAPH 双工退化的坑。
**How to apply**：重测 RTS RTF 时确认 OMNI_SERVER_BIN=build-prof、ACL_GRAPH=on、FA=0；官方数字以 judge core 为准。

相关：[[sprint13-cont-view-result]] [[sprint5-tts-flash-attn-result]]
