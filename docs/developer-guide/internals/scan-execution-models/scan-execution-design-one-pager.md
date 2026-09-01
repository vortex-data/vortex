# Scan Execution Design: One-Pager

**Model in one sentence:** stateful operator pipelines do the work; an advisory out-of-band
demand plane makes work not exist; one gather node changes cardinality; morsels typed by row
domain are the unit of parallelism, memory, and priority.
Full version: [design](scan-execution-design.md).

```text
              CONTROL PLANE (out-of-band, advisory, supersets only, never blocks)
   [pruning verdicts] [conjunct bounds] [join SIP: blooms/IN-lists] [limit counter]
        ^ write              ^ write                                  | read at admission only
        |                    |                                        v
 scan open                   |                +--------------------------------------------+
+-----------------+     +---------+  claim    |  MORSEL  (row domain R, rows [a,b))        |
| lower + bind    | --> | morsel  | --------> |  1 PLAN once: demand snapshot; skip empty; |
| routing table   |     | queues  | work-     |    defer gated ("plan X when fact seals")  |
| (composed maps) |     | per     | stealing  |  2 IO: late demand look, batch-coalesced   |
| kernel table    |     | domain  |           |  3 FILTER: conjunct pipelines (CPU);       |
+-----------------+     +---------+           |    masks stream IN-BAND, meet at countdown |
                                              |  4 GATHER: sealed mask = survivor map      |
                                              |    (the ONE cardinality change)            |
                                              |  5 PROJECT: reuse stash decodes; pack      |
                                              |  6 EMIT (pull-driven, ordered) -> RETIRE   |
                                              |                                            |
                                              |  STASH (edge,range)->arrays/masks; dropped |
                                              |  wholesale at retire = the memory unit     |
                                              +---------------------+----------------------+
                                                                    | gate seals (offsets/codes)
                                                                    v  PRIORITY over new outer claims
                                              +--------------------------------------------+
                                              | CHILD-DOMAIN MORSELS (list elems, dict     |
                                              | values): own demand (sealed at birth),     |
                                              | pipelines, stash; results -> parent stash  |
                                              | (list) or scan-wide cells (dict values)    |
                                              +--------------------------------------------+
```

## Laws

- **Values are positional** over their domain; dead rows are undefined; gather-by-map is the
  only cardinality change, always an explicit planned node.
- **Commutation**: `f(sel(x)) = sel(f(x))` for row-local, non-trapping kernels, through the
  edge map — so work on stale superset demand is always correct; selection later fixes it.
- **Demand is advisory**: read at admission, never waited on; the only sync point is the
  gather's in-band mask being final. Late/lost demand costs performance, never correctness.

## Parts

- **Exec plans**: DuckDB/Velox-style stateful pipelines, one per conjunct, few per projection;
  state owned by the morsel; struct is a stage, not a pipeline break.
- **Demand plane**: bind-time composed routing (producer -> consumer maps); content is any
  superset summary (bounds, zone verdicts, blooms, limit). SIP is this plane, not a feature.
- **Scheduler**: morsels + work stealing; optimistic IO/CPU below the watermark; cascade vs
  parallel = snapshot staleness at admission, one code path; conjunct order = admission
  pricing, not plan structure.

## Rules

- Pruning is warm-up: first wave optimistic and unpruned; stats fact seals; bulk verdicts;
  every later morsel pruned for free.
- Planning is one-off per morsel; shrink captured by one late demand look per read; gated
  subtrees are deferred planning, not re-planning.
- Depth-first: child-domain morsels before new outer claims (WIP ≈ workers × depth).
- No new morsels when: all rows claimed (every domain) | memory limit (deferred) | limit
  sealed the tail.
- Stash is the buffering home; cross-morsel sharing = short list of scan-wide cells
  (dictionaries, stats, prune fact); straddling chunks decode twice (bounded duplicate).
- Demand cells only shrink; needs that grow across morsels (dict value pages) are keyed-cell
  dedup, not demand; ordered limit = per-morsel first-k cells shrinking as earlier survivor
  counts seal (superset by construction, exactness enforced at emit).

## Layout author writes

Declarations (edges, maps, coverage, kernels) + a combine (zip/wrap/intersect/take, priced if
per-row) + optional planning override (Zoned, Dict, List).
Never: scheduling, demand, coordinates, buffering, ordering, pipelines (compiled from the
declarations).

## Gates

Eager oracle, row-hash on need set; OOB disabled/delayed must match; `can_trap` audit before
elective gathers (v1's `CAST(a,u8)` comment is the counterexample); deterministic scheduler
simulator (memory × ordering × limit deadlocks); perf: Q01/Q06, FineWeb `select *`, selective
strings, dict page skip, cold-scan IO parity, entries-per-admission ≈ 1.
