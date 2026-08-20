
Sprint 2 Task 7 (TTS→T2W pipeline overlap) finding from real duplex eval data
(`evaluation/judge-final/sessions/20260814_*/stage_timing.jsonl`, 20 files, 2026-08-14~15):

- `t2w_queue_wait_ms` = **0.0 in ALL sessions** (mean/p50/p90/max all 0.0).
- Interpretation: T2W thread is blocked on `cv.wait` when TTS hasn't produced a full chunk yet;
  it dequeues freshly-pushed items (`dequeue_time ≈ oldest_enqueue_time`). T2W is the FASTER side
  and waits for TTS — i.e. **TTS is the pipeline bottleneck**, not T2W.
- Both tts_ms sum and token2wav_ms sum contribute ~equally to RTF (e.g. tts_sum=8200-8746ms,
  t2w_sum=7863-8256ms per ~34-wav session).
- The T2W queue is **unbounded** (push never checks MAX_QUEUE_SIZE) → producers (LLM/TTS) never
  block on queue space → overlap is already maximal for a producer-consumer pair.
- **Official RTF is immune to overlap**: `eval_duplex_e2e_latency.py` computes
  `compute_ms = sum(encode, llm_prefill, llm_decode, tts, token2wav)` (stages summed per frame),
  only `encode = max(vpm_ms, apm_ms)` credits parallelism. So increasing queue depth / overlap
  does NOT lower official SPEAK→WAV RTF — only reducing tts_ms or token2wav_ms directly does.

Implication for optimization: T2W compute is NOT hidden from the official metric even though it
overlaps in wall-clock. Both tts and t2w must be reduced. See [[minicpm-competition-progress]].
