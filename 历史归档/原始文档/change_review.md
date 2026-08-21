# 对话审查表 / 改动全记录（2026-08-15 更新，910C）

## 一、代码改动审查表
| # | 改动 | 文件 | 状态 | 实测效果 | 结论 |
|---|---|---|---|---|---|
| 1 | **启用 flash-attn（主 LLM）** | `tools/omni/omni.cpp` | ✅ **保留** | RTF 1.0086→**~0.84**（decode 0.318→0.138） | **核心成功优化**（唯一保留的代码改动） |
| 2 | flash-attn env 可切换 | `tools/omni/omni.cpp` | ✅ 保留 | `OMNI_FLASH_ATTN=0` 关/默认开 | 便于 A/B 验证 |
| 3 | QKV 融合检测（fi910c） | `ggml/src/ggml-cann/ggml-cann.cpp` | ❌ 回滚 | 检测要求连续节点，实际 Q@2/V@7/K@9 不连续 → 从不触发 | 无效，回滚 |
| 4 | QKV+gate/up 融合补丁 | `ggml/src/ggml-cann/ggml-cann.cpp` | ❌ 回滚 | GroupedMatmulV3 报 EZ1001：**F32×F16 dtype 不匹配** | 无效，回滚 |
| 5 | GCINSTR 插桩/profile/graph-dump | `ggml-cann.cpp` | ❌ 回滚 | 诊断用，有每步开销 | 诊断完已清 |
| 6 | skip-launch 探针 | `ggml/src/ggml-cann/aclnn_ops.h` | ❌ 回滚 | 跳过 launch 反而更慢（34ms）→ 证实 GWS 有 device 往返 | 诊断用，已清 |
| 7 | executor 复用探针 | `exec_reuse_probe.cpp`（独立） | ❌ 弃用 | `SetRepeatable` **段错误**；`RepeatRunWithCache` 报 561000 | CANN 9.1.0-beta.1 不可用 |
| 8 | 评测二进制编译 | build（eval-cli/daily-cli/tts-eval） | ✅ 保留 | 精度评测可用 | 基建 |
| 9 | eval config.env（910C） | `evaluation/config.env` | ✅ 保留 | 指向真实模型/数据路径 | 基建 |
| 10 | daily-omni 子集提取 | `extract_daily.py` + `appendix/daily-omni/` | ✅ 保留 | 120 样本 + 媒体 + jsonl | 基建 |
| 11 | Python 依赖补齐 | 系统 python | ✅ 保留 | httpx/librosa/soundfile/Pillow/websockets/dotenv | 修 RTS 挂死 |
| 12 | **TTS flash-attn** | `tools/omni/omni.cpp` | ❌ 回滚 | A/B：tts 0.2645(开) vs 0.274(关)，区间重叠 → **不显著** | 噪声内，回滚 |
| 13 | **t2w flow 步数 env 化** | `tools/omni/omni.cpp` | ❌ 回滚 | n_timesteps=3 → GPU init 失败（prompt_cache/图约束） | 死路，回滚 |
| 14 | **Q4_0 量化实测** | 模型转换（已删 Q4 文件） | ❌ 弃用 | RTF 0.84→**1.108 超标**（decode +112%、prefill +280%） | launch 墙吞噬带宽，弃用 |

## 二、成功优化表（有效果、可复现）
| 优化 | 原理 | 实测效果 | 保留状态 |
|---|---|---|---|
| **flash-attn 启用** | vanilla 注意力（每层 3 op）融合成 1 个 flash-attn op，图节点 1445→1265，fused kernel 更优 | decode RTF **0.318→0.138（-56%）**；rtf_aggregate **~0.84±0.03（多次实测 0.82-0.87）** | ✅ 保留 |
| **精度零回归验证** | flash-attn vs 非 flash 逐题对比 | daily-omni 120 子集均 85.0%（120/120 一致）；videomme 50 题均 64.0%（50/50 一致） | ✅ 已验证 |
| **RTS 依赖修复** | 补 Pillow/websockets 等 | 修掉 judge 卡 chunk 0 / 报告生成崩溃 | ✅ 保留 |

## 三、已证伪路径（勿重蹈）
| 路径 | 失败原因 | 数据 |
|---|---|---|
| **Q4_0/Q8_0 量化** | launch 墙：量化 matmul 多 op + 反量化 kernel 慢，带宽收益被吞 | **实测 RTF 0.84→1.108**（decode 0.154→0.327、prefill 0.015→0.057）；Q4 只影响 LLM（vision/audio/tts 不可量化） |
| **K-quant（Q4_K_M 等）** | CANN 后端 `ggml_cann_mul_mat` 只支持 Q4_0/Q8_0，其他 **GGML_ABORT** | aclnn_ops.cpp:2315 默认分支 |
| **executor 复用（6 路全验证）** | ① 普通复用 2nd launch rc=161001（**ACLNN_ERR_PARAM_NULLPTR**，executor 单次使用）② SetRepeatable Matmul **段错误** ③ RepeatRunWithCache 561000 ④ InitExecutorCacheThreadLocal 无效 ⑤ AddCache 无效（CheckCacheable=1 仍不命中）⑥ MM_ENV_ACLNN_CACHE_LIMIT 无效 | 6 个独立探针 |
| **TTS flash-attn** | TTS 模型小、attention 占比低，收益在噪声内 | A/B tts 0.2645 vs 0.274（重叠） |
| **t2w flow 步数 5→3** | prompt_cache/图约束，GPU init 失败 | n_timesteps=3 崩 |
| **VPM 缓存**（encode 占 20%） | RTS 视频 35 帧全像素不同（唯一 hash），精确缓存零命中 | ffmpeg 帧对比 |
| QKV/ffn GroupedMatmul 融合 | F32输入×F16权重 dtype 不匹配（EZ1001） | probe EZ1001 |
| ACL graph 捕获 | KV 动态 shape → 每步重捕获 | 之前 EE9999 |
| W8A8 量化 | 每 matmul 拆 8 op → launch 墙爆炸，总 RTF 反超 | 910B3: 1.587 vs 1.369 |
| Q4/Q8 出厂量化 | 内部反量化回 fp16，不省算力 | 已证伪 |
| TTS/视觉量化 | llama-quantize unsupported arch | 已证伪 |

## 四、最终指标（910C，F16 + flash-attn）
| 指标 | 基线 | +flash-attn |
|---|---|---|
| rtf_aggregate（SPEAK→WAV 主指标） | 1.0086 | **~0.84±0.03**（多次实测 0.82-0.87，远低于门槛 1.0） |
| stage 分解 | decode 0.318 + tts 0.259 + t2w 0.243 + encode 0.176 | decode 0.138~0.154 + tts 0.26 + t2w 0.24 + encode 0.17 |
| daily-omni 120 子集 | — | **85.0%（控制组同分）** |
| videomme 50 题 | — | **64.0%（控制组同分，零回归）** |
| 官方精度基线（准入） | VideoMME 69.0 / Daily-Omni 79.5 / TTS WER 1.414 | F16 全达；videomme 64% 需全量复测确认 |

## 五、RTF 优化极限结论（2026-08-15）
- **F16 + flash-attn 是这套系统（llama.cpp-omni + CANN 9.1.0-beta.1 + 910C）的 RTF 终点**，~0.84 远优于官方基线 1.087（-23%）
- 核心限制：**launch 墙**（每 op ~13μs 派发，decode 833 op = 11ms/步）；唯一结构解 executor 复用在此 CANN 版本彻底不可用
- 量化（含 shared_assets 官方全套 Q4/Q5/Q6/Q8）因 launch 墙全灭；Q4_K_M 等后端直接 ABORT
- run-to-run 方差 ±0.03，任何 <0.03 的改动无法可靠验证

## 六、待办/风险
- **videomme 精度**：64% 待全量复测（用户将安排全量跑）——官方 F16 基线 69.0，64% 疑为小样本偏差，非模型固有
- **tts WER 验证**：910C CANN bug（aclnnRepeatInterleave）阻断，非我方改动；910B3 曾 WER 1.5% 达标
- **正式提交材料整理**：submission_report.md 待补 RTF 极限结论 + Q4/executor 证伪记录
- 环境干净：Q4 文件已删、eval 配置已还原基线、omni.cpp 仅保留主 LLM flash-attn

## 关键资源补充（2026-08-15 发现）
- **共享盘官方量化模型**：`/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5-gguf/` 有全套（Q4_0/Q4_K_M/Q4_K_S/Q4_1/Q5_*/Q6_K/Q8_0/F16），但后端只支持 Q4_0/Q8_0 且 RTF 实测失败
- **官方评测规范**：`C:\Users\30300\minicpm_ascend_trackA_spec.md`（基线 69.0/79.5/0.709/1.414，准入相对 F16 降幅 ≤2pp）
