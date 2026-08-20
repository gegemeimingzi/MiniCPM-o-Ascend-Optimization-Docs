# MiniCPM-o 昇腾优化实验归档（子赛道 A：llama.cpp-omni 推理优化）

> 队伍：**doma** · 赛道：昇腾 AI 创新大赛 · 子赛道 A（推理性能 / RTF）
> 仓库：`llama.cpp-omni`（`bench/huawei` 分支）· 硬件：Ascend 910 ×2（CANN）
> 归档时间：2026-08-21

## 目录结构

```
.
├── README.md                ← 本说明
├── 项目总结.md               ← 项目总览（模型、管线、赛题、准入规则）
├── 优化成果.md               ← 活文档：RTF 演进、最终配置、保留/否决项（最重要）
├── 优化实验/                 ← 每个实验方向一个文件（✅有效 / ❌否决均保留）
│   ├── sprint2-*.md          ← TTS/T2W 流水线分析与 queue/overlap 发现
│   ├── sprint3-headcode-npu-result.md
│   ├── sprint4-analysis-findings.md
│   ├── sprint5-tts-flash-attn-result.md
│   ├── sprint6-t2m-analysis.md
│   ├── sprint7-b-lite-result.md
│   ├── sprint8-b-mid-feasibility.md
│   ├── sprint9-vocoder-blite-result.md
│   ├── sprint10-cache-inplace-analysis.md
│   ├── sprint11-kv-slot-write-result.md
│   ├── sprint12-graph-dispatch-analysis.md
│   ├── sprint13-cont-view-result.md
│   ├── sprint16-rtf-remeasure-finding.md
│   ├── flow-B1尝试.md         ← flow B=1（token2wav 减半，因精度评测崩溃否决）
│   └── ...
└── 历史归档/
    └── 总体进度与决策.md       ← 跨 Sprint 的完整进度、技术坑、决策记录
```

## 约定

- **优化成果.md**：RTF/配置有变化时更新，保持单一事实来源。
- **优化实验/**：一个方向一个文件，结论标注 ✅有效 / ❌否决（附受控数据），被否决的也保留，避免重复尝试。
- **历史归档/**：跨阶段的时间线、精度评测基建、技术坑等长记录。

## 核心结论速览

- **官方主指标**：`SPEAK→WAV` core RTF 基线 `1.087`（910C，F16）。
- **最终提交**：本轮（2026-08-21）FULL RTF ~0.69，core 预期 ~0.70（相对官方基线 **-35~-40%**）。
- **关键有效优化**：主 LLM FlashAttention（-56% decode）、TTS FlashAttention、head_code NPU、VPM 批量编码、4 步 flow、t2m 冗余 CONT 消除。
- **关键否决**：Q4/Q8 量化（launch 墙）、executor 复用（CANN 单次使用）、cache in-place / slot-write（CANN 后端不可行）、vocoder F16（负优化）、flow B=1（token2wav -16% 但 TTS 精度评测崩溃）。
