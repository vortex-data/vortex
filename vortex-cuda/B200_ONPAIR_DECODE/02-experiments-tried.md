# Experiments tried — the full ledger

Every approach attempted on B200, with the measured outcome. The **negatives are the valuable
part**: they map the boundaries and each has a measured reason. All "tried" kernels are
byte-exact unless noted (ablation proxies are timing-only).

## Shipped (in the selector)

| change | result | why it works |
|---|---|---|
| `b128o12` (128-thread, 75% occ) general default | +6–12% vs old `4tpt` | 128-thread granularity is the B200 lever |
| `split8read_b128o12` for high-frac dicts ≤16384 entries | +18–46% on text | 8 B reads from L1-resident `dict_s8` |
| `frac_le8` gate 0.90→0.70 (sm_100) | captured URL bits12 (+5.5%) | URL frac 0.81 wins; l_comment/ps_comment (<0.70) excluded |
| dict-size gate 4096→16384 (sm_100) | captured bits14 (+9%) | `dict_s8` ≤128 KB still fits L1 |
| arch split (`pick_general_blackwell`/`_hopper`) | GH200 unchanged | granularity regressed on single-die GH200 |

## Tried and rejected — with the measured reason

| approach | kernel(s) | result | reason it lost |
|---|---|---|---|
| **split4read** (4 B reads) | `split4read_b128o12` | −8 to −23% vs split8 | 4 B is below the 32 B sector → cuts no transactions, adds >4 B fallback. 8 B is optimal. |
| **length-bucket dict** (stride {4,8,12,16}) | `lenbucket_b128` | ties `b128o12` on bits16 (+0.6–0.9%), regresses long-token | bucket-branch divergence offsets the smaller working set; bits16 still > L1 |
| **L2 persistence** (pin dict in L2) | `ONPAIR_L2_PERSIST` env | **no effect** (637→637) | the 1 MB dict is *already* L2-resident; limiter is L1 gather, not L2 eviction |
| **cluster-DSMEM** (dict sharded across a thread-block cluster's distributed shared mem) | `cluster_dsmem` | byte-exact but **−75 to −80%** | ~1 block/SM occupancy collapse + remote DSMEM gather saturates the GPC fabric |
| **variable-width exact** (offset:24\|len:8 dir + packed bytes) | `vwidth`, `vwidth_b128` | **−69 to −79%** | arbitrary byte offsets force *unaligned* 16 B loads (memcpy) — catastrophic |
| **variable-width quantized** ({4,8,12,16}, 4-aligned) | `vwidth4`, `vwidth4_b128` | +6% on bits12, **−8 to −19% on bits16** | aligned loads recover most of vwidth, but 768 KB working set still > L1 + 2nd directory gather |
| **8 tokens/thread** (more amortization/ILP) | `8tpt`, `8tpt_b128` | −2 to −22% | doubled register pressure cuts occupancy/spills; 4tpt is the amortization peak |
| **dict-in-shared** (stage `dict_s8` in shared, persistent grid) | `shdict8` (and older `pdict`) | byte-exact but **−47 to −65%** | cooperative-load + `__syncthreads` + occupancy hit; the gather was never the bottleneck (see ablation) |
| **bits10 / bits11** (smaller code budget) | data only | worse ratio (1.2–1.4×), no speed gain | dict already L1-resident at bits12 |
| **bits15 / "half-filled 16-bit dict"** | data only | ratio 2.6× but decode 637 = full bits16 | 256 KB `dict_s8` exactly fills L1 → no headroom; need ≤128 KB |
| **hot-subset cache** (cache top-N entries) | — (not built) | unviable | bits16 access is near-uniform (top-4096 = 47%); no hot subset exists |
| **freq-ordered codes / hot-dict** (`Track D/E`) | removed | −6 to −8% / forbidden | freq-ordering interferes with other pipeline parts (project constraint) |

## Ablation NCU-proxies (timing-only, not byte-exact, not selectable)

Built to find the limiter since NCU is blocked. `_ablate` = full byte-exact baseline; `_no*`
remove one stage; `_cfree` = conflict-free emit addressing.

| proxy | what it isolates | finding |
|---|---|---|
| `ablate_nogather` | dict gather cost | ~40% of runtime |
| `ablate_noemit` | emit (byte-staging) cost | **~70% of runtime — the bottleneck** |
| `ablate_nodrain` | output drain cost | ~8% |
| `ablate_noscan` | warp prefix-scan cost | ~20–25% |
| `ablate_cfree` | bank-conflict vs store-count | recovers only 3% → emit is **store-count bound** |

## The one justified-but-unbuilt next step

**Shuffle/`byte_perm` emit.** The emit is 70% and store-count-bound (~900 shared byte-stores/
warp). Assemble aligned 16 B output chunks in registers via `__shfl` + `byte_perm` funnel shifts
(processing 4 bytes/instruction on the ALU/shuffle path, *off* the LSU), then write
~`warp_total/16` *coalesced* `uint4` stores → **~25× fewer store instructions**. Potential 612 →
toward 2000 GiB/s. It is a genuine rewrite (variable-length cross-lane byte assembly is the hard
primitive the current code dodges via shared staging), so it needs careful implementation, but
it is the first change backed by a measured diagnosis rather than a guess.

**Other open items:** confirm the limiter with **NCU on B200** (needs container relaunch with
`--cap-add SYS_ADMIN`); pinned-memory H2D (~5× whole-decompress on this PCIe5 box, still
transfer-bound); batching tiny columns per launch (the ~15 µs launch floor — dbtext columns are
launch-bound, not decode-bound).
