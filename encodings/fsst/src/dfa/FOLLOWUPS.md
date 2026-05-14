# FSST LIKE follow-ups

Self-contained briefs for the three remaining DFA prefilter items. Each
section is sized to be handed to a subagent as a complete prompt:
context, constraints, files-to-touch, exit criteria, validation. The
shared "Required context" block below applies to all three tasks.

---

## Required context (read before any task)

Vortex's FSST LIKE pushdown lives in `encodings/fsst/src/dfa/`:

| File | Role |
|---|---|
| `dfa/mod.rs` | `FsstMatcher` enum + `LikeKind::parse`. KMP byte-table and symbol-transition builders. Wildcard/CI helpers. |
| `dfa/folded_contains.rs` | `FoldedContainsDfa` (`%needle%`, needle ≤127 bytes). Owns `scan_to_bitbuf` routing: escape-only memmem → dense short-circuit → Teddy + fused SSA → 1-byte → row_loop. |
| `dfa/flat_contains.rs` | `FlatContainsDfa` (`%needle%`, 128–254 bytes). Same routing minus Teddy. |
| `dfa/prefix.rs` | `FlatPrefixDfa` (`prefix%`). Single-direction DFA, fail-state on mismatch. |
| `dfa/suffix.rs` | `SuffixMatcher` (`%suffix`). Backward scan. |
| `dfa/multi_contains.rs` | `MultiContainsDfa` (`%seg1%seg2%`). Chained KMP. |
| `dfa/anchor_scan.rs` | Streaming Teddy-2 / Teddy-3 + bucket collection. AVX2 / AVX-512 / NEON / scalar variants. SSA fusion is inline in each. |
| `dfa/skip.rs` | Per-string skip strategies (memchr1/2/3, bitmap). |
| `compute/like.rs` | `LikeKernel` entry — parses options, builds `FsstMatcher`, dispatches to `scan_to_bitbuf`. |
| `benches/fsst_like.rs` | Divan benches. `fsst_contains` is parametric on a mined-needle corpus; `fsst_contains_{htt,ear,https}_*` cover SSA-density regimes; `fsst_not_contains_*` covers NOT LIKE. |

The DFA construction pipeline is:

1. **Parse** the pattern into a `LikeKind` (Prefix/Suffix/Contains/MultiContains).
2. **Build the byte table** (`kmp_byte_transitions` for contains/multi, `build_prefix_byte_table` for prefix, `build_suffix_byte_table` for suffix). Wildcards (`b'_'`) fill the row with the advancing state; case-insensitive matching sets both letter cases via `set_advance`.
3. **Build symbol transitions** (`build_symbol_transitions`) — for each `(state, FSST symbol code)`, simulate feeding the symbol's expansion bytes through the byte table.
4. **Fuse** symbol transitions + the ESCAPE row into the final `(state × 256) → state` table.
5. **Bucket-extract** progressing-c1 codes and (c2,c3) advancement sets for Teddy.
6. **Capture SSA** (`single_step_accept_codes`) — codes whose state-0 transition lands directly at `accept`. These are passed through to the streaming Teddy as `ssa_codes`; Teddy AVX2/AVX-512/NEON fuse them via an extra per-block PSHUFB.

`scan_to_bitbuf` routes between paths in this order:

```
escape-only memmem      [when no symbol contains any needle byte and pattern is wildcard-free + case-sensitive]
ssa_saturated → 1-byte  [SSA-density estimated > ~32k candidates total]
triple Teddy + fused SSA + pair fallback
pair Teddy + fused SSA
escape_pair (1 bucket, c1=ESCAPE, ≤3 c2's, no SSA)
1-byte progressing bitset
row_loop scan_to_bitbuf_with
```

Negation (`NOT LIKE`): all streaming paths handle the `negated: bool` correctly by initializing the bitbuf inversely and unsetting on match.

Test conventions: `rstest` cases preferred; tests live in `dfa/tests.rs`; `assert_arrays_eq!` for array comparisons; helpers `sym(bytes)` and `escaped(bytes)` build symbols and all-escape-encoded code streams.

Bench conventions: `cargo bench -p vortex-fsst --bench fsst_like --features _test-harness`. Compare against the bench snapshots in `git log` for recent commits.

Validation gates for any change:
- `cargo test -p vortex-fsst --lib` (must keep passing all existing tests)
- `cargo +nightly fmt --all` (clean)
- `cargo clippy -p vortex-fsst --all-targets --all-features` (no new lints in changed files; preexisting lints on `mod.rs:498` / `anchor_scan.rs:3100+` / `dfa_compressed/` are out of scope)
- `./scripts/public-api.sh` if any public API moves
- `cargo bench -p vortex-fsst --bench fsst_like --features _test-harness` showing no regression on the existing parametric set

---

## Task A — Shift-Or / Bitap for short needles (≤8 bytes)

**Goal.** Add a `ShiftOr` matcher variant in the FSST DFA stack, used when `Contains` patterns have needles of length ≤ 8 bytes AND no SSA codes are present in the trained dict for the needle. Bit-parallel single-`u64`-state matcher; one ALU op per code byte instead of one table lookup + the Teddy verifier dispatch.

**Why.** For very short needles, the Teddy + verify dispatch is the bottleneck — there are many candidates per byte but the actual DFA work is trivial. Shift-Or replaces both the prefilter and the verifier with a single `state = (state << 1) | B[byte]` per byte, accept on a fixed high bit.

**Where it fits in the routing ladder.** Before fused-Teddy, after escape-only memmem and dense short-circuit. So in `folded_contains.rs::scan_to_bitbuf`, the order becomes:
```
escape-only memmem → dense short-circuit → shift-or → triple Teddy → pair Teddy → 1-byte → row_loop
```

**Algorithm sketch.**
- Needle of length L (1 ≤ L ≤ 8): build a 256-entry `B: [u64; 256]` table.
- For each pattern position `i` in 0..L, for each byte value `b` that "matches" `needle[i]` (literal, or any byte if wildcard, or case-folded byte when `ci`): set `B[b]` bit `i` to `0` (rest are `1`).
- State starts at `!0u64`. For each input byte `b`: `state = (state << 1) | B[b]`. Match when `state & (1 << (L - 1)) == 0`.
- For FSST: build B over the COMPRESSED stream by composing through the symbol table. Each FSST symbol code `c` (length up to 8) maps to a transition function that updates the state by feeding all of the symbol's bytes through `B`. Precompute `B_sym[c]: fn(u64) -> u64`. Or, equivalently, store `B_sym[c]` as `(shift: u32, or_mask: u64)` — the shift is `8 * symbol_length`, the or_mask comes from `OR` of shifted `B[symbol_byte_i]`.
- ESCAPE_CODE handling: ESCAPE consumes the next byte as a literal; encode that as a two-step state transition. Either build a dedicated ESCAPE table or fall back to the standard byte path for escape pairs.

**Files to touch.**
- New file `dfa/shift_or.rs` containing `ShiftOrDfa` with `new(symbols, lengths, needle, ci)` and `matches(codes) -> bool`. Inner-loop must be `#[inline(always)]`.
- `dfa/mod.rs`: add a `ShiftOr(ShiftOrDfa)` variant to `MatcherInner`. Add a build-time gate in `FsstMatcher::try_new_with`: prefer `ShiftOr` over `FoldedContains` when needle ≤ 8 bytes AND `!case_insensitive_with_non_ascii`.
- `dfa/mod.rs::scan_to_bitbuf` dispatches new variant via `scan_to_bitbuf_with`.

**Constraints.**
- Wildcard support: `_` at position `i` makes `B[b]` bit `i` zero for ALL `b`. Same as ANY-byte match.
- Case-insensitivity: when `ci`, also clear `B[b]` for the case-flipped letter byte.
- The needle-length cap is 8 because we need `L - 1 ≤ 63` for the high-bit accept; we also want `state >> (L * 8)` to be defined for the symbol composition.
- Don't enable when SSA codes exist — Shift-Or matches deterministic byte sequences; SSA introduces multi-byte symbol shortcuts that would over-approximate without a separate verifier.
- The FSST symbol composition is the subtle part — write it carefully and add a property test that compares against the FoldedContains result on the same needle+data.

**Exit criteria.**
- New `dfa/shift_or.rs` (~200 lines including doc-comment + tests).
- `ShiftOrDfa` selected for ≤ 8-byte needles in `FsstMatcher::try_new_with` (unless SSA is present, falling through to FoldedContains).
- At least 5 tests: 1-byte / 2-byte / 8-byte needles; wildcard; case-insensitive; with FSST symbols; vs FoldedContains on random data (property test).
- All 169+ existing tests pass.
- New bench `fsst_contains_short_<dataset>` (≤4-byte needles like `%abc%`, `%xy%`) added to `benches/fsst_like.rs`. ShiftOr should be ≥1.5× faster than FoldedContains on selective short needles.

**Risk / known pitfalls.**
- The FSST symbol composition is easy to get wrong on multi-byte symbols. Cross-check against the existing FoldedContains for the same needle.
- Beware that the state bit-shift direction is convention-dependent — I described "right-to-accept" where state bit `i` means "matched i+1 bytes ending here". Be consistent.
- Don't ship if Shift-Or regresses any existing fsst_contains parametric bench.

---

## Task B — Fat Teddy / multi-pattern OR

**Goal.** Support `LIKE x OR LIKE y OR …` (and analogous `IN (x, y, …)` LIKE-of-string patterns) with a single Hyperscan-style Fat Teddy pass: 16 buckets, 4-byte fingerprint per bucket, AVX2 or AVX-512. Today every LIKE in the OR runs as a separate scan + boolean OR of results; Fat Teddy collapses them into one streaming pass.

**Why.** ClickBench Q23 (`UserAgent LIKE … OR LIKE … OR LIKE …`) is the canonical workload. Real-world `LIKE IN (…)` lists are typically 3–50 needles. A naive N-pass implementation is `N×` slower than necessary; Fat Teddy can reach ~1.5× the single-pattern cost.

**Where it fits.** This is a NEW entry point in the FSST `compute/like.rs` / `kernel.rs` layer (one that takes `&[&[u8]]` patterns and returns one BoolArray per pattern OR a single `OR`-merged result). The single-pattern code path stays unchanged.

**Algorithm sketch.** (Reference: Hyperscan's `fdr/fdr_loadval.h` + `teddy_avx2.h`. There is a Rust port in `aho_corasick::packed::teddy` — *do not copy-paste*, but study the structure.)
- Build 16 buckets across all needles. Pack needles into buckets greedily by first-byte (and second-byte for collision avoidance).
- For each bucket, build a 4-byte fingerprint table: 16-entry nibble lookup × 4 successive byte positions (c1 through c4). The fingerprint match is bit-AND of the 4 PSHUFB results.
- Per 32-byte block (AVX2): 4 loads (`v1` at `i`, `v2` at `i+1`, `v3` at `i+2`, `v4` at `i+3`). 8 PSHUFB lookups. 3 ANDs. movemask → 32-bit candidate mask. For each set bit, identify the bucket (the bits in the mask are per-bucket; bit `b` in lane `j` set ⇒ candidate for bucket `b` at offset `j`). Run the per-bucket verifier (DFA matches) for that needle.
- Buckets that share more than one needle pack via the cross-bucket scheme (also from Hyperscan).
- ESCAPE / wildcard / FSST-symbol semantics: each per-bucket verifier is the existing single-pattern DFA, so the symbol semantics are already correct.

**Files to touch.**
- New file `dfa/fat_teddy.rs` containing the bucket-packing algorithm, the AVX2 + AVX-512 + scalar passes, and bucket-identification helpers.
- `dfa/mod.rs`: add a `MultiNeedleMatcher` (separate from `FsstMatcher`) with `try_new_multi(symbols, lengths, &[&[u8]])` returning per-pattern `FoldedContainsDfa`s + a packed `FatTeddyIndex`.
- New entry point in `compute/like.rs` or a separate `compute/like_multi.rs` for the multi-pattern case. Or, simpler: add `fn like_or(...) -> Option<ArrayRef>` to `LikeKernel`-style trait. (Confer with the LikeKernel API; if there's no existing precedent for batched OR, defer the API design question and write a free function for now.)

**Constraints.**
- All needles must be `Contains` shape (`%x%`). Mixing prefix/suffix needles in a single Fat Teddy pass is out of scope.
- Wildcard and case-insensitive flags must be consistent across all needles in the batch.
- Number of needles per Fat Teddy pass capped at 16 (1 per bucket). For more, chunk into groups of 16 and OR-merge across passes.
- Fall back to N-pass when Fat Teddy doesn't apply (mixed shapes, > 16 of certain dense buckets, etc.).

**Exit criteria.**
- New `dfa/fat_teddy.rs` (~600–1000 lines including SIMD variants).
- A new test set covering: 3, 8, 16-needle OR; ensure result equals OR-of-N-single-pattern results on the same data.
- New bench `fsst_contains_or_<n>_<dataset>` with `n ∈ {3, 8, 16}`. Fat Teddy should be ≥1.5× faster than N single-pattern passes for n ≥ 4.
- All existing 169+ tests pass.

**Risk / known pitfalls.**
- The bucket-packing greedy can collide pathologically — write the property test against random needle sets to expose this.
- ESCAPE_CODE c1 collides across patterns; needs cross-bucket handling per Hyperscan FDR.
- Don't ship if the single-pattern parametric benches regress.

---

## Task C — Engine planner / cost-model routing

**Goal.** Replace the hardcoded routing cascade in `FoldedContainsDfa::scan_to_bitbuf` (and the parallel `FlatContainsDfa` / `MultiContainsDfa` cascades) with a small cost-model planner that picks the best scan path before execution.

**Why.** The cascade is hardcoded — every new variant (Shift-Or, Fat Teddy, dense short-circuit) is a manual `if` branch. The SSA-density gate I shipped is a one-off hack of this kind. A planner cleanly extends to N variants and is the right place to wire in column statistics (min/max, bloom filters, histogram) when those land.

**Inputs available at scan time** (free):
- `n` (row count), `all_bytes.len()` (total compressed bytes).
- `needle.len()`, `accept_state`, `|progressing_codes|`, `|SSA_codes|`, `|buckets|`, `|triple_buckets|`.
- Whether `escape_only_pattern` is feasible.
- Architecture features (AVX2/AVX-512/NEON detected).

**Inputs available with sampling** (~µs):
- SSA density (already computed by `ssa_saturated`).
- Estimated candidate density per path.

**Decision table to encode.**

```
1. escape_only_pattern is Some?               → escape_only memmem
2. SSA codes present AND density > THRESHOLD? → 1-byte progressing bitset
3. needle.len() ≤ 8 AND no SSA?               → ShiftOr (after Task A)
4. all_bytes.len() < 4 KiB?                   → row_loop
5. triple buckets exist?                      → triple Teddy + SSA fusion + pair fallback
6. pair buckets exist (no SSA, ≤3 c2)?        → escape_pair (specialized)
7. pair buckets exist?                        → pair Teddy + SSA fusion
8. progressing codes exist?                   → 1-byte progressing bitset
9. fallback                                   → row_loop
```

**Architecture.** Introduce a `ScanPlanner` struct that owns the inputs and exposes `plan(&self) -> ScanPlan`. `ScanPlan` is an enum of `EscapeOnly | OneByteSaturated | ShiftOr | TripleTeddy | PairTeddy | EscapePair | OneByteBitset | RowLoop`. `scan_to_bitbuf` dispatches on the plan instead of branching inline.

**Cost model** (start simple — calibrated constants, not a learned model):

```rust
fn cost(plan: ScanPlan, ctx: &ScanContext) -> u64 {
    // Setup + per-byte scan + per-candidate verify, in approximate ns.
    match plan {
        EscapeOnly         => SETUP_MEMMEM + ctx.all_bytes / MEMMEM_THROUGHPUT_BYTES_PER_NS,
        ShiftOr            => SETUP_SHIFTOR + ctx.all_bytes * SHIFTOR_NS_PER_BYTE,
        TripleTeddy        => SETUP_TEDDY + ctx.all_bytes * TEDDY_NS_PER_BYTE + ctx.estimated_candidates * VERIFY_NS,
        ...
    }
}
```

Constants come from the existing benches — calibrate per architecture (a tiny build-time decision: AVX-512 vs AVX2 vs NEON vs scalar).

**Files to touch.**
- New `dfa/planner.rs` with `ScanPlanner`, `ScanContext`, `ScanPlan`, `cost`, `plan`.
- `dfa/folded_contains.rs::scan_to_bitbuf`: extract each path into a method (`run_escape_only`, `run_shiftor`, `run_triple_teddy`, …), then have the top-level dispatch be `match self.planner().plan() { ... }`. Same refactor (smaller) for `flat_contains.rs` and `multi_contains.rs`.
- Tracing: under `VORTEX_FSST_PLAN_TRACE`, print the inputs + chosen plan + estimated cost. Required to validate the planner picks the right thing on each bench.

**Constraints.**
- The planner must not perform any allocation in its decision (only constant-time arithmetic + the existing SSA-density sample).
- The planner must produce the same plan that the existing cascade produces today on the bench corpus (lock this in with a per-bench `assert_eq!(plan, …)` test) — i.e. the planner refactor must not regress any bench, then improvements come from cleaner additions.
- Architecture detection happens once at construction time, not per-call.

**Exit criteria.**
- New `dfa/planner.rs` (~300 lines).
- `scan_to_bitbuf` in folded / flat / multi all dispatch via the planner.
- New tests `test_planner_picks_*` covering each decision-table row.
- All 169+ existing tests pass, all existing benches at parity ± noise (1.05× threshold).
- A regression-trace test that confirms the planner picks the same path as the legacy cascade for every existing fsst_like bench.
- Doc comment on `ScanPlanner` explaining the cost-model rationale and the calibration source (which bench produced which constant).

**Risk / known pitfalls.**
- The per-bench parity assertion is the load-bearing test. If a single planner decision diverges from the legacy cascade for any bench, you've either improved or regressed something — figure out which before declaring done.
- Don't try to add column-statistics integration in this task. Just the routing refactor.

---

## Running these as subagents

Each task is sized for a single subagent worktree run. The expected protocol:

1. Launch with `subagent_type: "general-purpose"` and `isolation: "worktree"`.
2. Pass the **Required context** block + the chosen task section verbatim as the prompt.
3. Add at the end: *"Implement the task, run `cargo test -p vortex-fsst --lib` plus the validation gates listed, then commit on the worktree branch and print the diff stat + branch name."*
4. After the subagent returns, review the diff, run benches locally on the parent worktree, integrate.

Tasks are independent and can run in parallel — none of them touch overlapping files in a way that would conflict (Shift-Or adds a new file + a `MatcherInner` variant; Fat Teddy adds new files + a separate matcher type; the planner refactors `scan_to_bitbuf` only).

Recommended order if running sequentially: **Task A (Shift-Or) → Task C (planner) → Task B (Fat Teddy)**. The planner becomes much more valuable after Shift-Or is in (it adds a routing decision); Fat Teddy is the largest and benefits from a stable planner.
