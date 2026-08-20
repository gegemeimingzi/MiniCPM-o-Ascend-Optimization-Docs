
Sprint 5 单变量实验（TTS FA on/off，910，2026-08-15）完成：

**改动**：omni.cpp:4456 TTS context 设 `flash_attn_type=ENABLED`，env `OMNI_TTS_FLASH_ATTN`（默认/1=ENABLED，0=AUTO=vanilla）。唯一改动点。

**结果**：
- **decode/token -7~-10%**（CPU headcode: 7.25→6.55 p50~6.84→6.16；NPU headcode 20 样本: 6.71→6.22）
- **TTS 阶段 RTF -7.6%**（0.192→0.177，排除 T2W）；decode RTF 0.160→0.148
- **SPEAK→WAV RTF 持平**：0.714→0.720。原因：**T2W（flow-matching 14K-op 图）占 wall 73-75%**，TTS 省 1.7s 被采样分叉的音频长度/T2W 波动淹没
- **正确性**：57 对齐 token logits max_abs=0.01、top1/top5 100%；first_hidden max_abs=0.0057；token 57 采样分叉（非数值错误）
- **WER**：1.35%→2.42%（配对 17/20 相等、2 差 1 好 → 采样噪声非系统性）
- **ASV 无回归**：wavlm_large 0.709→0.712（+0.2%）
- **graph**：FLASH_ATTN_EXT 9/11 替代 SOFT_MAX/CONT（0），节点 203/242→176/209；**ROPE/SET_ROWS 保留**（FA 不消除）

**修正 Sprint 4 预期**：FA 只消除 SOFT_MAX+CONT（~0.6ms/step，decode 的 ~10%），ROPE/SET_ROWS 仍在 → 非 -30~40%。

**结论**：FA 是真实 TTS 阶段加速，但**单独不能降 SPEAK→WAV RTF**（T2W 主导）。建议配合 T2W 图 op 融合（Sprint 4 #4）。FA 不建议默认开启（收益小、WER 点估计略升），用 `OMNI_TTS_FLASH_ATTN=0` 回滚。

**ASV 工具链修复**（此前 Sprint 3 误判缺失）：checkpoint 在 `/workspace/competition/appendix/wavlm_large_finetune.pth` + `wavlm_large.pt`；需 `/opt/clabagent/python-env/bin/python`（torch 2.13）+ `pip install s3prl`（0.4.18，装到 clabagent env）+ `S3PRL_REPO=/workspace/competition/s3prl`；`verification_pair_list_v2.py` 必须显式传 `--wav1_start_sr 0 --wav2_start_sr 0 --wav1_end_sr -1 --wav2_end_sr -1`（argparse 无默认 → 不传 TypeError）。

报告：`/workspace/competition/sprint5_tts_flash_attention_report.md` + `tts_fa_graph_diff.md`。相关：[[sprint4-analysis-findings]] [[sprint3-headcode-npu-result]] [[minicpm-competition-progress]]
