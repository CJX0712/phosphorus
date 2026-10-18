# 基准矩阵报告（T5-C5）

- 生成时间：2026-10-18 23:12:07
- 环境：0.1.0 | python 3.13.14 | numpy 2.5.4
- 平台：Windows AMD64 AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD | cpu_count=16
- native C99 内核：✅ 可用
- 口径：**seq = 序列总长**（prompt 128-n_new / 512-n_new + 生成 32）；greedy（temperature=0）确定性解码；speedup = 相对同 (batch, seq) fp32 基线
- 投机配置：K=8，draft_layers = n_layer-1（O1 自草稿，extra_bytes=0）
- rss_mb_peak = 进程峰值 RSS（Windows: psapi `PeakWorkingSetSize`；POSIX: `getrusage` 高水位；含解释器 + numpy + 权重常驻）
- ppl = teacher-forcing 困惑度；无 tokenizer 的合成模型记 null；spec 行复用同 backend ppl（同一份权重，不变量 ② 分布等价）

## synthetic_12l.phos

### seq=128（总长）

| 配置 | batch | 状态 | tokens/s（端到端） | tok/s（解码相） | ms/token | ttft_ms | rss_mb_peak | ppl | α | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | 1 | ok | 7.5 | 8.0 | 134.28 | 280.9 | 220 | — | — | 1.00× |
| int8 | 1 | ok | 2.3 | 2.7 | 435.07 | 1922.8 | 325 | — | — | 0.31× |
| tl | 1 | ok | 8.3 | 12.4 | 121.10 | 1371.6 | 414 | — | — | 1.11× |
| tl+spec | 1 | ok | 5.0 | 11.0 | 198.08 | 3849.7 | 475 | — | 1.0 | 0.68× |

<pre>
端到端 tokens/s（seq=128，柱长按峰值归一）
       tl b=1  ████████████████████████████      8.3
     fp32 b=1  █████████████████████████      7.5
  tl+spec b=1  █████████████████      5.0
     int8 b=1  ████████      2.3

解码相 tokens/s（剔除 ttft，seq=128）
       tl b=1  ██████████████████████████████████████████     12.4
  tl+spec b=1  █████████████████████████████████████     11.0
     fp32 b=1  ███████████████████████████      8.0
     int8 b=1  █████████      2.7
</pre>

## tiny.phos

### seq=128（总长）

| 配置 | batch | 状态 | tokens/s（端到端） | tok/s（解码相） | ms/token | ttft_ms | rss_mb_peak | ppl | α | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | 1 | ok | 353.2 | 409.4 | 2.83 | 13.0 | 78 | 57.56 | — | 1.00× |
| int8 | 1 | ok | 113.1 | 153.1 | 8.84 | 74.0 | 95 | 57.57 | — | 0.32× |
| tl | 1 | ok | 185.1 | 306.0 | 5.40 | 67.7 | 109 | 57.57 | — | 0.52× |
| tl+spec | 1 | ok | 132.1 | 335.4 | 7.57 | 173.0 | 115 | 57.57 | 1.0 | 0.37× |

<pre>
端到端 tokens/s（seq=128，柱长按峰值归一）
     fp32 b=1  ████████████████████████████    353.2
       tl b=1  ███████████████    185.1
  tl+spec b=1  ██████████    132.1
     int8 b=1  █████████    113.1

解码相 tokens/s（剔除 ttft，seq=128）
     fp32 b=1  ████████████████████████████████    409.4
  tl+spec b=1  ███████████████████████████    335.4
       tl b=1  ████████████████████████    306.0
     int8 b=1  ████████████    153.1
</pre>

## 方法论与诚实性说明

1. **seq 口径**：`seq` 为序列总长（prompt + 生成）。若按「prompt 长度」语义，
   prompt+生成 = 512+32 超出全部可用模型 max_seq_len
   （tiny.phos=256 / synthetic_12l=512），`forward` 的滑窗守卫会静默截断，
   指标即失真——故矩阵不采用该口径，容量不足组如实 skipped。
2. **batch+spec 未组合**：`generate_tokens` 为单序列实现；batch>1 的 spec 组
   记 skipped（多草稿树 + 连续批处理的组合在 T5 范围外，属后续工作）。
3. **ppl 口径**：teacher-forcing 分块困惑度（块内 reset cache），非生成式 ppl；
   合成模型（无 tokenizer）记 null——随机 token 上的 ppl 不具解释意义。
4. **rss_mb_peak 为进程级高水位**：同进程多次运行取 `ru_maxrss`（Windows 为
   psapi `PeakWorkingSetSize`）峰值，含 Python 解释器 + numpy + 模型权重常驻；
   跨 backend 差异主要来自反量化缓存（fp32/int8 解包 int8，tl 保留 5-bit 打包字节）。
5. **解码相 tok/s**：`n_new / (总时长 − ttft)`，由实测时刻差推导（非估算）。
   n_new=32 时端到端吞吐对微模型属**预填主导区**（spec 还需双 cache
   预填），解码相口径才能反映投机机制的真实收益区间——长生成场景见
   BENCHMARKS.md §5.6（n_new=256，解码主导）。
6. **中位数聚合**：每组重复 N 次取中位数（本机实测单次采样抖动可达 ±35%，
   共享/虚拟化 Windows 上时序噪声不可忽视）；raw_timings 全量在 results.json。
7. **无任何估算/外推数字**：未运行的组一律 skipped + 原因。
