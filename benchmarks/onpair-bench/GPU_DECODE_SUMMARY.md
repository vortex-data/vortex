# OnPair GPU decode — dataset summary (compression + GPU decompression)

Single **NVIDIA GH200** (Hopper, sm_90). Decode = OnPair string-decompression
CUDA kernel time only (no host transfer). Compression = one OnPair dictionary
per chunk, sampled ~1 GB of the column.

- **bits** = OnPair dictionary code width (12 → ≤4096 dict entries; 16 → ≤65536).
- **ratio** = decoded bytes ÷ on-disk `.vortex` bytes (≈ in-memory ratio).
- **kernel** = auto-selected by `pick_auto_kernel`: `split8read` when the dict is
  small (bits12) and ≥90% of tokens are ≤8 B, else `4tpt`.
- GPU clocks are not locked, so absolute GiB/s drifts ±5–10% between runs;
  intra-table comparisons are reliable.

## Summary — `chunk1000mb` (one chunk, ~1 GB sampled)

| dataset | bits | decoded | compressed | ratio | decode kernel | GiB/s | vs 4tpt |
|---------|-----:|--------:|-----------:|------:|---------------|------:|--------:|
| fineweb       | 12 | 1000 MB | 441.8 MB | 2.26 | split8read | **567.3** | +11% |
| fineweb       | 16 | 1000 MB | 347.5 MB | 2.88 | 4tpt       | 470.1     | —    |
| book-reviews  | 12 |  522 MB | 200.6 MB | 2.60 | split8read | **606.6** | +4%  |
| book-reviews  | 16 |  522 MB | 158.5 MB | 3.29 | 4tpt       | 561.8     | —    |
| wikipedia     | 12 |  703 MB | 324.6 MB | 2.17 | split8read | **537.9** | +9%  |
| wikipedia     | 16 |  703 MB | 249.8 MB | 2.82 | 4tpt       | 537.8     | —    |
| ps_comment    | 12 |  988 MB | 157.9 MB | 6.26 | 4tpt       | 1117.3    | —    |
| ps_comment    | 16 |  988 MB | 169.8 MB | 5.82 | 4tpt       | 866.2     | —    |

`split8read` is selected (and wins +4–11%) only for the short-token text columns
at bits12; long-token (ps_comment) and all bits16 columns use `4tpt`.

## Chunk-size sweep (compression + decode)

Decode columns show **4tpt → split8read** ms (lower is better); ratio is mem ratio.

### fineweb / bits12
| chunk  | n_chunks | dict (total) | ratio | 4tpt ms | split8read ms |
|--------|---------:|-------------:|------:|--------:|--------------:|
| 10 MB  | 96       | 1.6 MB       | 2.238 | 2.614   | 2.427         |
| 100 MB | 10       | 171 KB       | 2.257 | 1.902   | 1.723         |
| 1000 MB| 1        | 17 KB        | 2.264 | 1.821   | 1.640         |

### wikipedia / bits12
| chunk  | n_chunks | dict (total) | ratio | 4tpt ms | split8read ms |
|--------|---------:|-------------:|------:|--------:|--------------:|
| 10 MB  | 68       | 1.1 MB       | 2.168 | —       | —             |
| 100 MB | 7        | 118 KB       | 2.158 | 1.393   | 1.269         |
| 1000 MB| 1        | 17 KB        | 2.166 | 1.331   | 1.217         |

### wikipedia / bits16
| chunk  | n_chunks | dict (total) | ratio | 4tpt ms | split8read ms |
|--------|---------:|-------------:|------:|--------:|--------------:|
| 10 MB  | 68       | 26.1 MB      | 2.345 | —       | —             |
| 100 MB | 7        | 3.17 MB      | 2.745 | 1.344   | 1.431         |
| 1000 MB| 1        | 456 KB       | 2.815 | 1.217   | 1.283         |

**Takeaways**
- **bits12 dicts saturate** at 4096 entries (~17 KB) within ~10 MB of text, so the
  per-chunk dict size — and the compression ratio — are flat across chunk sizes
  (fineweb 2.238→2.264, wikipedia 2.158→2.166). Chunk size is free to pick.
- **bits16 / high-cardinality dicts do not saturate**: smaller chunks replicate a
  large dict (wikipedia/bits16 dict 456 KB→26 MB going 1000 MB→10 MB) and the
  ratio drops 2.815→2.345. **Prefer large chunks for bits16 compression.**
- Larger chunks also decode faster (fewer, larger kernel launches).

## Notes

- Best decode kernel found: `onpair_shmem_4tpt` (baseline) and
  `onpair_shmem_4tpt_split8read` (bits12 short-token columns, +4–11%).
- `split8read` reads 8 B (`uint2`) from the 32 KB `dict_s8` for the common case,
  relieving the L1/TEX-request bottleneck; the 16 B padded dict is only touched
  for `len > 8` tokens.
- Frequency-ordered dictionary codes (encoder-side, not applied) would add a
  validated +8–13% on bits16 columns by improving dict L1 residency.
- Full optimization log and rejected ideas:
  `vortex-cuda/kernels/src/ONPAIR_GPU_DECISION_TREE.md`.
