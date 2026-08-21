# vocoder_layout_candidates.md — Vocoder B-lite 型冗余 CONT 扫描

> Sprint 8 Phase 5，只读扫描（不改 vocoder）。范围：token2wav-impl.cpp vocoder 区（~4920-6800，hg2/hift/stft/sine/resblock）。

---

## 1. 高置信候选：hg2 conv 的 `reshape→cont`（连续 tensor 的 reshape 是连续 view）

**模式**（hg2_f0_predictor_conv1d / hg2_resblock_conv1d / hg_hift conv，约 3-5 处 conv）：
```cpp
ggml_tensor * im2col = ggml_im2col(...);
ggml_tensor * im2col_2d = ggml_reshape_2d(ctx, im2col, ...);
im2col_2d               = ggml_cont(ctx, im2col_2d);   // ← 冗余候选 1
ggml_tensor * w_2d = ggml_reshape_2d(ctx, w_kic_oc, ...);
w_2d               = ggml_cont(ctx, w_2d);             // ← 冗余候选 2（常量权重）
ggml_tensor * mm    = ggml_mul_mat(ctx, im2col_2d, w_2d);
ggml_tensor * y_tcb = ggml_reshape_3d(ctx, mm, ...);
y_tcb               = ggml_cont(ctx, y_tcb);           // ← 冗余候选 3
ggml_tensor * b_1c1 = ggml_reshape_3d(ctx, bias, ...);
b_1c1               = ggml_cont(ctx, b_1c1);           // ← 冗余候选 4（常量 bias）
```

**判断**：`ggml_reshape` 对连续 tensor 是连续 view（数据已按新形状连续排列）。im2col 输出（CANN post-process 写入连续 dst）、mul_mat 输出、权重、bias 均连续 → 4 个 cont 都是**纯冗余拷贝**（同 B-lite G4 模式）。

**必须验证**：
- `im2col_2d`：确认 CANN im2col 的 dst 是连续（post-process 后）。reshape [K*Cin,T,B]→[K*Cin,T*B] 是合并末两维 → 连续 view。
- `w_2d` / `b_1c1`：常量权重/bias 在 load 时预 reshape+cont 一次即可，图内每次重算是浪费。
- 风险：`ggml_mul_mat` 对 a/b 的布局要求（当前传 view，理论上连续 → 应可）。

**预计**：每处 conv -4 CONT；vocoder 图（2321 节点）若 ~3-5 conv → **-12~-20 CONT**。vocoder.compute ~109ms → 预估 -1~-2%。

## 2. 中置信候选：STFT/sine 区 cont（~513-551）

`stft_real_tfb`/`stft_imag_tfb`/`x_tcb`（多次）/`si0`/`y` 等 cont。需逐处确认 producer 是否连续（STFT framing、phase 运算、reshape/view 后）。部分可能是布局必要。

## 3. 低置信 / 需保留

- `x_tcb = cont(permute(...))`：permute 后 layout 转换（G6 类），非 B-lite。
- `h_ctb`/`h_2d`/`f0_rep`/`f0_perm`/`s_over`：需逐处确认。

## 4. 结论

Vocoder 有低风险 B-lite 型优化（hg2 conv 的 reshape→cont，~4×N conv），预计 **vocoder.compute -1~-2%**、SPEAK→WAV RTF **-0.005~-0.01**。若做，建议在 B-mid 之后单独验证（vocoder 与 t2m 是独立图，改动隔离）。
