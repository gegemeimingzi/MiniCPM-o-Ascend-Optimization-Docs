# 官方评测方法分析（2026-08-20）

> 来源：官方群公告后，对照官方最新分支 `tc-mb/llama.cpp-omni` `bench/huawei`（commit `b06198f #100` "refine rtf test"）的 `evaluation/README.md`、`SUBMISSION_GUIDE.md` 与 `evaluation/config.env` 整理。
> 我们的开发基线是 `c9785cc #98`，落后官方 1 个提交（该提交大改了评测套件）。

---

## 一、官方 RTF 评测口径（速度成绩）

**定义**：`RTF = Σ 稳定帧模型计算时间 / Σ 对应音频时长`（**pooled ratio**，不是先算每个输入 RTF 再平均）。

**单帧计算时间**：
```
compute = max(VPM, APM) + LLM_prefill + LLM_decode + TTS + token2wav
```
分子来自 server 上报的各阶段耗时（SSE metrics 的 `vpm_ms / apm_ms / llm_prefill_ms / cost_llm_ms` + `stage_timing.jsonl` 的 `tts_ms / token2wav_ms`）。**计时字段含义、起止点、上报时机属于评测口径，不能改**；改了要按"架构级改动"人工复核。

**core 帧筛选**：每轮语音去**首帧**（冷启动）+**尾帧**（含 flush），中间才算 core，只对 core 帧汇总。单帧抖动大，样本太少不算数。

**正式评测门槛**：`RTS_MIN_CORE_FRAMES` 官方取远高于 3 的值（`run_eval.py` 默认 30），靠**多输入池化**满足。**`batch_validity` 不通过就不出主 RTF**。

## 二、正式评测流程

| 项 | 官方做法 |
|---|---|
| 输入 | **不公开的预切分 1s WAV/JPG 片段**（`RTS_TEST_CASE_DIR`），**不在评测时解码 MP4**（避免解码耗时/取帧抖动混进成绩）|
| 多卡 | `RTS_ASSIGNMENT_MODE=rotating_groups`，每个输入在每张卡各跑一次，轮数=卡数，抵消卡间差异 |
| 状态隔离 | 相邻输入间整体重启 server（`RTS_MODEL_LOAD_SLEEP_S`），保证干净状态 |
| 计分 | 整批 pooled：`Σ 合法 core compute / Σ 音频`。任一输入不合法则整批不自动出成绩，转人工复核 |

**正式评测与本文件（自测 config.env）的差别只有三处**：测试集合（未公开）、卡数、轮换次数。其余全部相同。

## 三、有效性检查（新增 `run_validity.py`）

自动流程对每个输入检查归帧与因果是否自洽，任一不通过即整批无效：
- 输入 chunk 有没有被丢弃
- 模态编码有没有静默失败
- SPEAK 帧是否都产出了 WAV
- WAV 是否归到了正确的源帧
- 是否出现负的因果延迟
- 非尾帧 TTS 的 token 数是否正常

（这些是异步流水线出竞态时的特征，不属于性能指标，但必须成立 RTF 才有意义。）

## 四、提交规范（全新 `SUBMISSION_GUIDE.md`）

**`submission.zip` 里必须且只能有 4 个文件**：
```
submission.zip
├── README.md              优化说明+构建运行+复现+集成+结果说明（UTF-8）
├── demo.mp4               完整功能演示视频
├── llama.cpp-omni.zip     git archive HEAD（基于 bench/huawei 的已提交源码）
└── integration-support.zip  MiniCPM-o Demo 等仓库外支持代码
```
- **禁止**放模型权重 / 优化后的 GGUF / 评测数据 / 生成结果 / 日志 / `.git` / build 产物 / 密钥 / 系统元数据。
- 代码必须**全部 commit 到 bench/huawei，`git status --short` 干净**；`git log -1 --oneline` 写入提交说明。
- **不得修改、删除或绕过评测入口、计时逻辑、正确性检查及结果校验逻辑**；架构级改动确需调整时按 §1.2 提供设计说明、自定义测量方法、与官方口径逐项对应、可复现命令与对比结果。
- 需要改后端 env（如 `GGML_CANN_ACL_GRAPH`）→ 在**外层 README** 里逐项声明（§9.1 表格：变量/官方默认值/提交值/适用模块/作用及必要性），不能靠改脚本或个人 shell。
- `llama.cpp-omni.zip` 必须用 `git archive --format=zip --prefix=llama.cpp-omni/ --output=../llama.cpp-omni.zip HEAD` 生成，只含已跟踪已提交文件。
- 密钥/密码/私钥不得写入任何提交文件。

## 五、与我们的现状对照（关键差异）

| 维度 | 官方要求 | 我们现状 | 是否一致 |
|---|---|---|---|
| 代码基线 | b06198f（#100）| c9785cc（#98）| ❌ 落后 1 个提交（评测套件大改）|
| RTF 自测口径 | 新归帧逻辑，样例只有 ~3 core 帧，基线 F16 1.1~1.2 | 旧 judge 跑到 21 core 帧、RTF 0.762 | ❌ 需用新 judge 重测 |
| 有效性检查 | 新增 `run_validity.py` 批量校验 | 无 | ❌ 缺 |
| 提交包格式 | 4 文件 `submission.zip` | 旧 5 目录结构（含评测结果/日志）| ❌ 需重建 |
| 评测入口 | **不得修改** `run_eval.py` | 我们改了 task_rts（ACL_GRAPH=on）| ⚠️ 需回退 + 改 README 声明 |
| demo.mp4 | 必需演示视频 | 无 | ❌ 缺 |
| env 变量 | 外层 README 声明 | 写在 config.env/脚本里 | ⚠️ 需移入 README |

**核心结论**：官方这次更新把评测套件和提交规范整体升级了一版，我们**自测 RTF（0.762）和提交包结构目前都与官方不一致**。

**下一步待办**（尚未执行，等确认）：
1. 拉官方 b06198f 分支作为新基线，把我们的优化补丁 rebase 上去；
2. 用新版 judge（含有效性检查）重测 RTF，拿到与官方同口径的数字；
3. 按新 4 文件结构重建提交包，`run_eval.py` 恢复原样、env 改到外层 README 声明，并补 demo.mp4。
