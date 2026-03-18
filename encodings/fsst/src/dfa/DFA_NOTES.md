# DFA Refactoring Notes

## Summary of changes (from 1229 → 1110 lines)

Unified 5 DFA structs down to 3:

| Before | After | What happened |
|--------|-------|---------------|
| `ShiftDfa` | (deleted) | Dead code — `FsstContainsDfa` only routed needles >14 to it, but `ShiftDfa::MAX_NEEDLE_LEN` was 14, so the arm was unreachable |
| `FsstContainsDfa` | (deleted) | Dispatch enum wrapping dead `ShiftDfa` arm; only the `FusedDfa` path was reachable |
| `FlatBranchlessDfa` | `FlatContainsDfa` | Merged with `FusedDfa` into single struct with `EscapeStrategy` enum |
| `FusedDfa` | `FlatContainsDfa` | Merged (see above) |
| `BranchlessShiftDfa` | `BranchlessShiftDfa` | Unchanged |
| `FlatPrefixDfa` | `FlatPrefixDfa` | Simplified escape transition building |

Other changes:
- Extracted `build_escape_folded_table()` (shared by `BranchlessShiftDfa` and `FlatContainsDfa`)
- Extracted `compose_packed()` (shared by `build_pair_compose` and `build_compose_4b`)
- Extended `FlatContainsDfa` folded range from 14 → 127 (2*127+1=255 fits in u8)
- Simplified `FlatPrefixDfa` escape transitions (reuse byte_table directly)
- Deleted `pack_escape_shift_table` (only caller was `ShiftDfa`)

## Removed code (recoverable from git)

All removed code is in commit `e08fb69ad` (the starting point). Key pieces:

### `ShiftDfa` (~70 lines)
Shift-packed `[u64; 256]` DFA using escape sentinel. Was identical in scan loop to
`BranchlessShiftDfa` but without the hierarchical 4-byte compose optimization.
Recovery: `git show e08fb69ad:encodings/fsst/src/dfa.rs` lines ~956-1027.

### `pack_escape_shift_table` (~15 lines)
Built a separate shift-packed escape transition table. Only used by `ShiftDfa`.
Recovery: same commit, lines ~418-433.

### `FsstContainsDfa` enum (~25 lines)
Dispatch enum: `ShiftDfa` for len ≤ 14, `FusedDfa` for len > 14.
Since caller guaranteed len > 14, the `ShiftDfa` arm was dead.
Recovery: same commit, lines ~592-615.

## Benchmark results: escape strategy comparison

Sentinel-only is 28-45% slower than folded for needles 8-14.
Both strategies must be kept in `FlatContainsDfa`.

| Benchmark | Needle len | Folded (ms) | Sentinel (ms) | Regression |
|-----------|-----------|-------------|---------------|------------|
| contains/log | 9 | 5.449 | 7.480 | +37% |
| contains/json | 10 | 2.390 | 3.466 | +45% |
| contains/path | 14 | 0.937 | 1.199 | +28% |

## Current benchmark baseline (post-refactor)

```
fsst_like         fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ fsst_contains                │               │               │               │         │
│  ├─ cb          1.593 ms      │ 2.122 ms      │ 1.725 ms      │ 1.745 ms      │ 100     │ 100
│  ├─ email       492.9 µs      │ 697.7 µs      │ 526.3 µs      │ 544.2 µs      │ 100     │ 100
│  ├─ json        2.282 ms      │ 2.731 ms      │ 2.401 ms      │ 2.406 ms      │ 100     │ 100
│  ├─ log         5.191 ms      │ 5.919 ms      │ 5.426 ms      │ 5.439 ms      │ 100     │ 100
│  ├─ path        894.3 µs      │ 1.076 ms      │ 941.1 µs      │ 952.8 µs      │ 100     │ 100
│  ├─ rare        1.674 ms      │ 4.55 ms       │ 1.814 ms      │ 1.992 ms      │ 100     │ 100
│  ╰─ urls        736.8 µs      │ 959.6 µs      │ 837.1 µs      │ 844.6 µs      │ 100     │ 100
╰─ fsst_prefix                  │               │               │               │         │
   ├─ cb          541.7 µs      │ 761 µs        │ 585.2 µs      │ 598.1 µs      │ 100     │ 100
   ├─ email       197.9 µs      │ 305.8 µs      │ 208.2 µs      │ 214.6 µs      │ 100     │ 100
   ├─ json        141.9 µs      │ 352.6 µs      │ 145.5 µs      │ 151.8 µs      │ 100     │ 100
   ├─ log         259.6 µs      │ 378.1 µs      │ 278.5 µs      │ 285.3 µs      │ 100     │ 100
   ├─ path        214.2 µs      │ 281.1 µs      │ 227.1 µs      │ 230.9 µs      │ 100     │ 100
   ├─ rare        153.7 µs      │ 191.9 µs      │ 157.1 µs      │ 160.8 µs      │ 100     │ 100
   ╰─ urls        260.7 µs      │ 445.4 µs      │ 294.2 µs      │ 297.7 µs      │ 100     │ 100
```

DFA routing per benchmark:
- cb, email, rare, urls (needle ≤ 7) → `BranchlessShiftDfa`
- log (9), json (10), path (14) → `FlatContainsDfa` (folded)
- No benchmark exercises sentinel path (would need needle > 127)

## Post integer-type cleanup benchmarks

After eliminating `u16`, tightening `usize` → `u8` in `compose_packed`, `pack_shift_table`,
`kmp_failure_table`, and `kmp_byte_transitions`. All within noise of baseline.

| Benchmark | Baseline (ms) | Current (ms) | Delta |
|-----------|--------------|-------------|-------|
| contains/cb | 1.725 | 1.695 | -1.7% |
| contains/email | 0.526 | 0.542 | +2.9% |
| contains/json | 2.401 | 2.452 | +2.1% |
| contains/log | 5.426 | 5.447 | +0.4% |
| contains/path | 0.941 | 0.949 | +0.8% |
| contains/rare | 1.814 | 1.762 | -2.9% |
| contains/urls | 0.837 | 0.812 | -3.0% |
| prefix/cb | 0.585 | 0.568 | -3.0% |
| prefix/email | 0.208 | 0.215 | +3.0% |
| prefix/json | 0.146 | 0.145 | -0.2% |
| prefix/log | 0.279 | 0.270 | -3.1% |
| prefix/path | 0.227 | 0.224 | -1.2% |
| prefix/rare | 0.157 | 0.159 | +1.1% |
| prefix/urls | 0.294 | 0.288 | -2.1% |

## Optimization ideas for later

### 1. 8-byte-per-iter BranchlessShiftDfa
Extend `BranchlessShiftDfa` to process 8 bytes/iteration via two 4-byte composes.
Would reduce loop overhead for long compressed strings. Tables stay the same size,
just add a `compose_8b` level on top of `compose_4b`.

### 2. Branchless prefix DFA
`FlatPrefixDfa` currently uses escape sentinel + branch. Could use escape-folding
(like the contains DFAs) to make the prefix scan branchless. Needs 2*prefix_len+1
states, so max prefix would drop. Worth it if prefix matching is a bottleneck.

### 3. Further struct merging
`BranchlessShiftDfa` and `FlatContainsDfa` (folded) share the same escape-folded
state layout. They differ only in table representation (shift-packed u64 vs flat u8).
Could theoretically be merged, but the hierarchical 4-byte compose in
`BranchlessShiftDfa` is fundamentally different from the flat scan, so sharing code
wouldn't simplify much.

### 4. Suffix pushdown (`%suffix`)
Two approaches noted in the module doc:
- Forward DFA with non-sticky accept (check state == accept after all codes)
- Backward scan of compressed stream
