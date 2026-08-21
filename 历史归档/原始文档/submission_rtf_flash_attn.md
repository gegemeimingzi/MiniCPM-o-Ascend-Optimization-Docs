# RTS RTF 优化记录（910C，2026-08-14）

## 结果
- **rtf_aggregate = 0.8175 / 0.8405 / 0.8478（三次独立运行，平均 ~0.835）**，官方硬门槛 <1.0，安全垫 0.16
- 910C 干净基线（F16，无优化）：rtf_aggregate = **1.0086**（stage: decode 0.318 + tts 0.259 + t2w 0.243 + encode 0.176）

## 唯一代码改动（1 行）
`tools/omni/omni.cpp` 的 `omni_init` 中，创建主 LLM context 时启用 flash attention：
```cpp
llama_context_params ctx_params = common_context_params_to_llama(*params);
ctx_params.n_ctx                = params->n_ctx;
ctx_params.flash_attn_type     = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // [OPT] flash attention
```

## 原理
- 原始 decode 图用 vanilla 注意力：每层 K@Q→softmax→V@S（3 个 aclnn op，1445 节点/步）
- flash-attn 把注意力融合成 1 个 op（1265 节点/步，-180 op），且用 fused kernel 处理 KV cache
- decode 每步 25ms → 22ms（探针），真实 RTS 中 decode RTF 0.318 → 0.138（-56%，因为真实上下文更长时 fused kernel 收益更大 + 少 180 次 launch）
- 输出质量无退化：SPEAK 文本连贯，aligned_pairs 27/27，无空文本

## 调试发现（关键）
1. **decode 每步调用 graph_compute**（1445 节点），launch 墙 = 每步 11ms 派发（833 个 aclnn op，GetWorkspaceSize 有 device 往返）+ 9ms 图构建
2. executor 复用（SetRepeatable/RepeatRunWithCache）在 CANN 9.1.0-beta.1 **段错误**，不可用
3. GroupedMatmulV3 融合（QKV/gate-up）因 **F32输入×F16权重 dtype 不匹配**（EZ1001）失败
4. ACL graph 捕获被 KV 动态 shape 卡住（vLLM bucket 方案是潜在路径，但已不需要）

## 复现方法
```bash
# 依赖（judge 需要，缺了会卡/崩）
pip install Pillow websockets httpx librosa soundfile numpy

# 跑 RTS（必须 --gpu 0，judge 的 _find_free_npu 会选错 Phy-ID）
cd /workspace/competition/llama.cpp-omni/evaluation/judge-final
MODEL=/workspace/competition/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-F16.gguf \
  ./run_judge_direct.sh --gpu 0

# 离线生成报告
python3 scripts/eval_duplex_e2e_latency.py --offline sessions/<最新session>

# 读结果
python3 -c "import json; r=json.load(open('sessions/<session>/eval_e2e_report.json')); print(r['rtf']['rtf_aggregate'], r['rtf']['stage_rtf'])"
```

## 待办
- [ ] 精度回归（videomme ≥67 / daily-omni ≥77.5 / tts WER ≤1.56）确认 flash-attn 无回归
- [ ] 正式提交材料整理


## 精度回归验证（2026-08-14）
- **daily-omni 120 子集**：flash-attn = 85.0%（102/120），非 flash-attn = 85.0%（102/120）
- **逐题对比：120/120 答案完全一致，零回归**
- flash-attn 改为 env 可切换：`OMNI_FLASH_ATTN=0` 关闭，默认开启（tools/omni/omni.cpp）
- 数据来源：共享盘 Daily-Omni parquet（内嵌媒体），提取到 /workspace/competition/appendix/daily-omni/
- videomme 缺视频（900 个需下载，未跑）；tts 缺打分模型（paraformer/wavlm）
