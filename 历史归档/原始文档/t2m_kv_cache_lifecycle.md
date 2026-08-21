# t2m_kv_cache_lifecycle.md — K/V Cache 生命周期

> Sprint 10 Task 2。纯源码追踪（token2wav-impl.cpp，B-lite 状态）。

---

## 1. K/V cache 生命周期完整追踪

### 1.1 数据流（current）

```
DiT block (每 ODE step，16 blocks × 10 steps)
  qkv = fused_linear(x)                      [3D, H, dt, B]
  q/k/v_heads = view_4d(qkv, D, H, dt, B)    [D, H, dt, B]
    │
    │  att_cache (persistent packed) [D2=2D, T, H·blocks·steps, B]
    │    └─ fm_attn_slice_att_cache → k_cache_slice [D, H, t_cache, B]
    │       └─ fm_attn_permute_cache_to_heads → [D, t_cache, H, B]  (permute 0,2,1,3)
    ▼
  k_total = concat(k_heads, k_cache_heads, 2)   ← ① K/V growth concat (624/625)
  v_total = concat(v_heads, v_cache_heads, 2)      [D, H, T_total, B]   (320 nodes)
    │   T_total = dt + t_cache；cap 600 (fm_attn_tail_view_time_dim)
    ▼
  flatten_heads → attention → output y
    │
    │  new cache:
    ▼
  fm_attn_build_new_att_cache:                  ← ② cache-build concat (446) (160 nodes)
    k_perm = permute(k_total, 0,2,1,3)          [D, T_total, H, B]
    v_perm = permute(v_total, 0,2,1,3)
    new_att = concat(k_perm, v_perm, 0)         [D2, T_total, H, B]  ← 布局=packed slot
    │
    ▼  (16 blocks collected)
  tree_concat(new_att_vec, 2)                   ← ③ step-pack tree-concat (983) (~150 nodes)
    [D2, T, H·16, B]  (16 blocks → 1)
    │  × 10 steps collected
    ▼
  tree_concat_outer(all_step_packs, 2)          ← ④ outer tree-concat (1018) (~9 nodes)
    [D2, T, H·16·10, B]
    ▼
  ggml_cpy(packed, est_att_cache)               ← ⑤ persistent write-back (cpy)
    sess_->est_att_cache  [D2, T, H·16·10, B]  (full-size preallocated)
```

### 1.2 关键问题回答

| # | 问题 | 答案（源码证据） |
|---|---|---|
| 1 | cache 第一次在哪分配？ | `setup_cache`（7653）：`est_att_cache = ggml_dup_tensor(ctx, est_cache_setup_out.att_cache)`（7727），`ggml_set_output`（7737）。 |
| 2 | 是否 full-size preallocated？ | **是**。dup 自 setup 输出 shape（含 H·blocks·steps），2GiB no_alloc ctx + union gallocr（7827-7849）一次分配全图。 |
| 3 | 实际 write offset？ | `fm_cfm_view_att_cache_packed`（658）：`slot = step·depth + block_idx`；`offset = slot·H·nb2`。每 (block,step) 占 H 个 head plane。 |
| 4 | 为什么需要 concat（①）？ | attention 需要完整上下文 [D,H,T_total,B]（new+cache）。concat 物化完整历史供 flatten/attention。**每 step 重新拷贝整个增长上下文 → O(T²) 累计拷贝。** |
| 5 | 为什么需要 pack（③④）？ | 持久 cache 需存 16 blocks × 10 steps 的全部 cache 供下 chunk 恢复。pack 把 block/step 沿 ne2（head 维）排列。**布局 = [D2, T, H·blocks·steps, B]。** |
| 6 | persistent cache layout？ | `est_att_cache [D2=2D, T, H·16·10, B]`，head 沿 ne2，slot (block,step) = 连续 H heads。 |
| 7 | 下游 attention 需要什么 layout？ | `fm_cfm_view_att_cache_packed` 提取 [D2, t_cache, H, B]（view），`fm_attn_permute_cache_to_heads` → [D, t_cache, H, B]，再 concat new heads。 |
| 8 | 中间 tensor 仅为 layout？ | ①k_total/v_total（new+old 拼接）、②new_att（K/V 打包）、③④step packs。②的输出布局**已与 persistent slot 一致** → 直接 slot 写可免去③④。 |

## 2. Before / After

### Before（current）
```
new_kv [D,H,dt,B]
   └─① concat(new, cache_heads) → [D,H,T_total,B]     ← O(T²) 重拷贝
        └─② concat(k_perm, v_perm) → [D2,T_total,H,B]  ← 布局=slot
             └─③ tree_concat(16 blocks) → [D2,T,H·16,B]
                  └─④ tree_concat(10 steps) → [D2,T,H·160,B]
                       └─⑤ cpy → persistent est_att_cache
```

### After（theoretical candidate）
```
new_kv [D,H,dt,B]  (per block, per step)
   └─ 写 persistent est_att_cache slot (block,step) 直写
        ①' view [D2, t_cache+dt, H, B] (persistent 读，无需 concat)
        ②' attention 直接消费 view（免 k_total 物化）
```
- 消除 ①②③④（~640 concat 节点），剩 16 blocks × 10 steps = 160 次 slot 写。
- 消除 O(T²) 重拷贝（每 step 不再复制整个增长上下文）。

## 3. Layout 兼容性结论

- **new cache [D2, T, H, B]（②输出）与 packed slot [D2, T, H, B]（⑤目标）布局一致**（permute 0,2,1,3 已产生）→ **直接 slot 写是连续写，无 strided**。
- 前提：persistent cache 的 head stride = D2·T·4（全连续 packed 布局）→ 满足（setup 连续分配）。
- **注意**：new cache 的 T = T_total（每 step 增长），slot 写需按当前 T 位置写入（cache T 维动态增长，cap 600）。

## 4. 风险点

1. **cache T 动态增长**：persistent cache T 维固定（setup shape），需在写入时管理 T 位置（view 偏移）。
2. **graph 内写 persistent cache**：需 cache 作为 graph 输入+输出（当前已用 ggml_cpy 实现，机制存在）。
3. **跨 chunk 语义**：gf_nonlast/gf_last 的 cache 传递需保持一致。
4. **in-place 写与读的依赖顺序**：graph 调度需保证 slot 写先于后续 step 读。
