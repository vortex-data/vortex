<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Integer `Narrow` PR briefs: NR1 to NR6

These briefs cover the integer `Narrow` track. Each PR is written from scratch from its brief
(D38). A local rewrite of the closed PRs #10223 and #10332 exists in
`/Users/connor/spiral/vortex-data/vortex3`. It is a reference, not a base: read its code and tests
for edge cases, reuse its test cases, and do not cherry-pick it. The rewrite predates D25, D26, D28
and D36, so its structure differs from these briefs. FX3 fixed the dictionary fill bug of the
rewrite with its own check, and it merged as `dcb4f50ae2` (#10745).

No build, test or benchmark ran for these briefs. Every claim about code comes from reading it,
and it is marked "Not verified" where it matters.

## Shared context for the track

### The reference rewrite

| Commit | Subject | Use it for | Do not copy |
| --- | --- | --- | --- |
| `f7408481ce` | feat(array): add in-memory Narrow integer encoding | NR1: the behavior of `encode`, `validate_dtypes`, the compare constant shortcut, the selection rules, and the cases of `narrow/tests/{encoding,compute}.rs` | The candidate list and the failed casts in `encode` (NR1 has one width helper). The integer-only parts of the compare kernel and the cast rule (see below). The `TakeReduce` rule, which puts `Narrow` above a lazy `Dict` (D25). The serialization stub |
| `8d548646ee` | feat(array): forward Narrow integer aggregates | NR3: the oracle helper `assert_aggregate_batches`, and the cases of `narrow/tests/aggregates.rs` and `encodings/fastlanes/tests/narrow.rs` | All of the source. It accumulates at the child dtype, casts the partial, and registers one fallback kernel for every aggregate. NR3 builds the partial of the parent dtype (D36) |
| `74aa17644e` | feat(array): push arithmetic and scalar operations through Narrow | NR4: the interval arithmetic, the cast, fill-null, zip and between rules, and the cases of `narrow/tests/{numeric,pushdown}.rs` | The replacement of the NR1 cast and fill-null rules. NR1 writes those rules so that NR4 extends them. The `integer_types` list, and the declines that build errors |
| `af4b663059` | feat(array): retain Narrow storage through merges and indices | NR5: `TakeIndicesReduce` and `test_narrowed_take_indices` | The chunked and interleave merge rules in `narrow/merge.rs`, which put `Narrow` above `Chunked` and `Interleave` (D25) |
| `7e2e5e4d7a` | fix(array): avoid truncating dictionary fill codes | FX3 fixed the same bug with its own check: it declines when the fill index is above the maximum of the codes type, and it keeps the fast path for non-nullable codes. It merged as `dcb4f50ae2` (#10745). NR5 reuses `test_fill_null_after_narrowing_take_indices` | Nothing |
| `adc1e83b9d` | feat(compress): retain narrow integer children through compression | NR1: `encode_values`. NR6: the list of call sites. WR1 to WR3: background | The `Narrow` wrappers for internal children, and the preview edition with only `vortex.narrow`. The `integer_types` list |

All these commits are reachable from every local worktree, because the worktrees share one bare
repository. The five rewrite commits are not on `origin`. Only the local branches
`ct/narrow-pushdown` (checked out in `vortex3`) and `ct/narrow-encoding` hold them, and both names
also exist on `origin` with the older commits of the closed PRs. Do not push, reset or delete
these two local branches. A remote agent cannot read the reference until it is pushed to a new
backup branch. `adc1e83b9d`, `0d94e44a91` and `07fafc2a5b` are on `origin/ct/narrow-compression`.

### Changes on `develop` since the rewrite

The rewrite is based on `61ff28d970`. NR1 to NR4 are based on `origin/develop` at `afa4bc87cd` or
later. These changes on `develop` matter when an agent reads the rewrite:

| Change on `develop` | Commit | What new code does |
| --- | --- | --- |
| `vortex_array::array_session()` removed | `d834a11faa` (#10681) | Every test module uses a module `static SESSION`, with the pattern below. The tests of the rewrite call `array_session()`, so port their cases, not their setup |
| Interleave uses named slots | `96ac9f2cc5` (#10688) | `InterleaveArray::try_new(values, array_indices, row_indices)` keeps its signature. The slot order is the same: selectors at 0 and 1, values from 2. `InterleaveSlots` is `pub(super)` (`vortex-array/src/arrays/interleave/array.rs:24`). The NR5 pull-up rule builds a new array with `InterleaveArray::try_new` |
| Storage pushdown in the aggregate session | `d7562aa0e6` (#10566), `3bf14ba647` (#10616) | `AggregateFnSession::default` has `register_storage_ordered` calls (`vortex-array/src/aggregate_fn/session.rs:127-129`). Register the NR3 kernels after the `Dict` kernels (`:133-135`). The lookup tries the exact aggregate id first and then the `None` key (`:328-337`) |
| `fill_null` on an empty input | `633c52deab` (#10627) | `FillNull` returns an empty array for a non-constant fill on an empty input (`vortex-array/src/scalar_fn/fns/fill_null/mod.rs:145`). The `Narrow` rule receives a `Scalar`, so it is not affected |
| Kernel for constant dictionary codes | `a169338930` (#10643) | An execute-parent kernel handles `Constant` codes (`vortex-array/src/arrays/dict/take.rs:181-205`). The NR5 rule for `Narrow` codes is a reduce rule, so the two do not overlap |
| Deprecated APIs and `legacy_session` removed | `85ae34d5f0` (#10595) | None of the removed APIs is needed |
| Execution contexts in validation and patches | `c48457b572`, `7216cb5cb8`, `673fd2e064`, `b549eb5a97` | `ListViewArray::try_new`, `Patches::new`, `MaskedArray::try_new`, `RunEndArray::try_new` and `SparseArray::try_new` take a context. Reduce rules have none, so the NR5 brief names constructors without a context |
| Error types without categories | `f3cbb329dc` (#10080) | `VortexError` has no categories, and `vortex_err!` takes the message only. New errors state the requirement first and end with `, got {actual}` |
| `as_constant` returns a borrowed scalar | `46429f5c53` (#10790) | The compare constant shortcut of NR1 reads the borrowed scalar, and clones it only where it must own it |
| Cast-rule contract on `CastReduce` | `bbb562a395` (#10798) | The `Narrow` cast rule pushes the cast into the child, which holds the validity, so it needs no `trivially_cast_nullability` guard and no `CastKernel`. `test_cast_conformance` checks the contract with a partially valid array (`vortex-array/src/scalar_fn/fns/cast/kernel.rs:26-44`) |
| `StatsSession` moved out of `vortex-array` | `d02df89980` (#10734) | `StatsSession` is in `vortex-expr` (`vortex-expr/src/stats/session.rs:24`). The NR2 `SESSION` registers `KernelSession`, `AggregateFnSession` and `MemorySession` only |
| FX3 merged | `dcb4f50ae2` (#10745) | The dictionary `fill_null` kernel declines a fill code that the codes cannot hold. NR5 no longer waits for FX3 |
| Benchmark guide | `eecdc7b20a` (#10808) | `docs/developer-guide/benchmarking.md` already asks for at least a few µs of real work and under 1 ms per iteration. #10808 makes `mimalloc` mandatory for every bench binary and adds the "What Not to Do" list |
| Decimal benchmarks (CD1) | `76d8371ddc` (#10752) | `encodings/decimal-byte-parts/benches/dbp_compute.rs` sets the conventions that NR2 follows |
| Wider dictionary codes in `fill_null` | `0d4ddfb616` (#10806) | The dictionary `fill_null` kernel widens the codes when the fill code does not fit, instead of declining. NR5 reads it before it ports `test_fill_null_after_narrowing_take_indices` |
| `between` over a constant null | `33675d9e39` (#10744) | `between` over a constant null array is null, not false (`vortex-array/src/scalar_fn/fns/between/mod.rs`). The NR4 between rule keeps this result |
| Non-generic array constructors | `ffaec9ffd1` (#10737) | The generic constructors of `PrimitiveArray`, `ConstantArray`, `DecimalArray`, `StructArray` and `ChunkedArray` call a private, non-generic builder. Their signatures are unchanged |

NR1 starts from `origin/develop` at `f476db42e3`, which holds the last three rows. The line numbers
in these briefs are from `afa4bc87cd`. Read each cited site at the base before changing it.

`VTable`, `OperationsVTable`, `ValidityChild`, the kernel registry, `CompareKernel`,
`execute_compare`, `execute_numeric`, the reduce adaptors, `ChunkedSlots::CHUNKS_OFFSET`,
`DictSlots` and `BitBuffer::full_in` keep the signatures that the rewrite uses at `afa4bc87cd`.
The execute-parent registry keeps a list for each key
(`vortex-array/src/optimizer/kernels.rs:263-276`). So the compare kernel of NR1 and the numeric
kernel of NR4 can both register for `Binary` on `Narrow`.

#### Test session pattern

PR #10681 gave each test module one session. `VortexSession::empty()` creates the vortex-array
session variables on first use, and the default `KernelSession` runs `arrays::initialize`
(`vortex-array/src/optimizer/kernels.rs:336-349`), so the `Narrow` kernels are present.

```rust
// Module level, as in vortex-array/src/arrays/dict/compute/fill_null.rs:110 on develop.
use std::sync::LazyLock;

use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);
```

A test outside vortex-array that needs another crate registers it in the closure, as
`encodings/fastlanes/src/bitpacking/array/tests.rs:43-47` does:

```rust
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_fastlanes::initialize(&session);
    session
});
```

### Stack and branches

The closed PRs used `ct/narrow-encoding` (#10223), `ct/narrow-pushdown` (#10332) and
`ct/narrow-compression` (#10333). All three still exist on `origin`. The new branches avoid them.

| PR | Branch | Base |
| --- | --- | --- |
| NR1 | `ct/narrow-in-memory` | `origin/develop` (`afa4bc87cd` or later) |
| NR2 | `ct/narrow-benchmarks` | NR1 |
| NR3 | `ct/narrow-aggregate-kernels` | NR2 |
| NR4 | `ct/narrow-arithmetic-kernels` | NR3 |
| NR5 | `ct/narrow-structural-rules` | NR4. FX3 is on `develop` (`dcb4f50ae2`) |
| NR6 | `ct/narrow-encode-values` | `develop` after NR1 merges. Merge it after NR5, so that the stack lands first (D35) |

NR1 to NR5 form one linear stack (D35). NR3 needs no code from NR2, but it sits on NR2 so that the
benchmarks exist before the kernels change. NR6 needs only NR1. The stack ends at NR4 for now. NR5
joins it later with `gh stack add ct/narrow-structural-rules` on the top branch, which keeps D35.

The titles and labels of NR1 to NR4 are proposals. Connor makes them final at each gate:

| PR | Title | Label |
| --- | --- | --- |
| NR1 | Add an in-memory `Narrow` encoding for integers | `changelog/feature` |
| NR2 | Add compute benchmarks for integer `Narrow` | `changelog/chore` |
| NR3 | Compute aggregates on the child of a `Narrow` | `changelog/performance` |
| NR4 | Run arithmetic and scalar operations on the child of a `Narrow` | `changelog/performance` |

#### Worktree and commands

One worktree holds the whole stack: `/Users/connor/spiral/vortex-data/vortex2`. No other worktree
checks out the four branches. `vortex1` stays on the research branch, and agents read the briefs
there by absolute path. Do not run `gh stack` in `vortex1` or `vortex3`. Do not check out `develop`
in `vortex2`, because `develop` is the trunk of the stack.

Base the stack on `origin/develop`. The local `develop` ref can be far behind `origin/develop` and
miss the commits that these briefs depend on. Never branch from it or rebase onto it without a
fetch.

The installed `gh stack` is v0.0.8. It has no `merge` command, and it keeps one stack catalog for
each worktree. Load the `connor-skills:gh-stack` skill, not `gh-stack`, which describes features of
a later version. Run every command in `vortex2`:

1. NR1: `git fetch origin`, make sure that the local `develop` equals `origin/develop`, then
   `git switch -c ct/narrow-in-memory origin/develop`, then
   `gh stack init --base develop ct/narrow-in-memory`. The bare repository checks out `develop`,
   so `git branch -f develop` can fail. Done on 2026-10-26 at `f476db42e3`.
2. NR2 to NR4: `gh stack add <branch>` on top of the previous branch.
3. After each step, check the stack with `gh stack view --json`.

Every commit has only the line `Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>` and no
co-author line. Every commit compiles and passes its tests, so the stack can be bisected.

#### Gates and pull requests

Connor reviews each PR, its diff summary and its PR text, before it is pushed and opened, and
before the next PR starts. After the approval:

1. Push with `gh stack push`.
2. Open the draft PR with `gh stack submit --auto`, which links the stack on GitHub. It takes the
   title from the commit subject or from the branch name, so set the title, body and label
   afterwards with `gh pr edit`. If you open the PR with
   `gh pr create --draft --base <previous branch>` instead, link the stack afterwards with
   `gh stack link --base develop` and the branches from the bottom up, for example
   `gh stack link --base develop ct/narrow-in-memory ct/narrow-benchmarks`. `link` uses `main` as
   the default base, and it pushes the branches that it names. The merge steps below hold only for
   a linked stack.
3. The body uses the repository template with `Tracking Issue: #10733`. It says that the stack
   replaces #10223 and #10332, and it names the PR below it.
4. Keep the PR as a draft until Connor marks it ready. Update the line of the PR in #10733 with
   `gh issue edit`.

To change a lower PR: `gh stack checkout <branch>`, commit, then `gh stack rebase --upstack`.
Compare each branch with `git range-diff origin/<branch>...<branch>`, then push with
`gh stack push`, which uses force-with-lease. After a rebase with conflicts, check
`git diff --cached`, because rerere stages the resolutions that it remembers.

Merge in the GitHub UI, bottom PR first, then run `gh stack sync` and `gh stack view --json`. Never
merge a PR whose base is still a `ct/` branch. For a stack that `gh stack submit` or
`gh stack link` links, GitHub retargets the next PR to `develop` after a squash merge and rebases
the PRs above it. A manual `git rebase --onto` with a force push races with that rebase and can
overwrite it, so do not use one.

#### CodSpeed on stacked PRs

CodSpeed baselines come from pushes to `develop` (`.github/workflows/codspeed.yml:13-17`). CodSpeed
compares a stacked PR with the head of its parent branch when that head has a finished run.
Otherwise it compares with `develop` and adds an `[^unexpected-base]` footnote.

- Read a report only when its "Comparing <branch> with <parent branch>" line names the parent at
  its current head, with no `[^unexpected-base]` footnote. Otherwise, ask Connor, and run the
  "Codspeed Benchmarking" workflow of the PR again after the run of its parent succeeds.
- NR3 compared with NR2 is the comparison that D19 wants. There is no need to wait for NR1 and NR2
  to merge.
- The run of NR1 measures NR1 merged into `develop` at that time. Keep NR1 rebased on the current
  `develop` when you read the reports.
- Each rebase of the bottom branch runs the full CodSpeed suite again on every PR of the stack.
- `action/bench-*` labels work only once a PR targets `develop`. Apply them to NR1, or to a later
  PR after GitHub retargets it. Check the CI of a stacked PR by hand before marking it ready.

#### Hooks

- The pre-push hook runs a workspace-wide clippy on the checked-out tree, once for each
  `git push`. Each push also asks for Connor's 1Password approval. Plan each push, submit and sync
  for a time when Connor can approve it.
- A push of several branches lints only the checked-out branch, and one failure blocks every
  branch. So run the clippy check of the brief on each branch before `gh stack submit` or
  `gh stack push`. If the hook fails on code outside the PR, stop and ask Connor.
- The hook checks sign-offs from the old remote tip. After a rebase onto a newer `develop`, that
  range holds `develop` commits without a sign-off, and the check fails. Confirm that our own
  commits are signed off, and ask Connor before any `--no-verify`.
- The pre-commit hook checks formatting with a floating nightly. Format with `nightly-2026-09-10`.
  If the hook rejects formatting that the pinned toolchain accepts, ask Connor.

### Write the kernels so that decimal support is additive

CD4 opens `Narrow` for decimals. The rewrite has integer-only assumptions that break then (table
below). The new NR code avoids them from the start where that costs little:

- No `as_ptype()` and no `PType::try_from` on a path that a decimal child can reach once
  `validate_dtypes` accepts decimals. An integer path matches `DType::Primitive` and declines any
  other dtype with `Ok(None)`. For example, the compare of two `Narrow` operands matches
  `DType::Primitive` on both children, compares at the wider child, and declines otherwise. CD4
  adds the decimal branch.
- The out-of-range compare takes the direction from
  `constant.partial_cmp(&Scalar::zero_value(constant.dtype()))`, not from `i64::try_from` and
  `is_signed_int()`.
- An aggregate runs in the accumulator of the parent dtype, and every `Narrow` kernel builds the
  exact partial of the parent (D36). The extrema, `IsConstant` and `IsSorted` kernels are exact for
  decimals as they are. The sum kernel declines decimal children, so a decimal sum takes the
  default path. CD4 can add a decimal branch for `Sum` with `skip_nans` only.
- `execute` can still yield a cast for integers, so that the child fuses decoding and widening.
  CD4 adds the decimal path, because the decimal cast of D15 returns a `Narrow`.
- Kernels that do not support decimals decline them with `Ok(None)`, so canonicalization handles
  them.

These sites of the rewrite assume integers. Most of them decline a decimal with `Ok(None)`, which
is safe. The others give a panic, a wrong result or a likely execution loop once `validate_dtypes`
accepts decimals:

| Site | Assumption | Effect with a decimal `Narrow` | Safe for decimals |
| --- | --- | --- | --- |
| `af4b663059:vortex-array/src/arrays/narrow/mod.rs:73-88` `validate_dtypes` | `PType::try_from`, integer, signedness, byte width | The gate. It rejects decimals | Gate |
| `f7408481ce:vortex-array/src/arrays/narrow/compare.rs:52-61` | `as_ptype()` and `DType::Primitive` for two `Narrow` operands | Panic: `DType::as_ptype` panics on a non-primitive dtype (`vortex-array/src/dtype/dtype_impl.rs:209-215`) | No |
| `f7408481ce:vortex-array/src/arrays/narrow/compare.rs:89` | `below = lhs.dtype().is_signed_int() && i64::try_from(&constant)? < 0` | Wrong result: `is_signed_int` is false for a decimal (`dtype_impl.rs:226-231`), so a negative constant below the child range answers `Lt`, `Lte`, `Gt` and `Gte` wrongly | No |
| `f7408481ce:vortex-array/src/arrays/narrow/compare.rs:76-86` | Constant cast to the child dtype | Correct when the child label is true (D10): the decimal scalar cast checks the precision (`vortex-array/src/scalar/typed_view/decimal/dvalue.rs:158-182`) | Yes |
| `af4b663059:vortex-array/src/arrays/narrow/rules.rs:316-346` `BetweenReduce` | Constant cast to the child, else two compares | Inherits the wrong `below` result through `two_compares` | No |
| `8d548646ee:vortex-array/src/arrays/narrow/aggregates.rs:83-89` | Accumulate at the child dtype, then cast the partial | Wrong result for `Sum` and `SumV2`: the decimal sum state is `Decimal(p + 10, s)` (`vortex-array/src/aggregate_fn/fns/sum/mod.rs:81-86`), so a child state overflows to null before the parent state does | No |
| `f7408481ce:vortex-array/src/arrays/narrow/vtable.rs:111-116` `execute` | Yields `values.cast(dtype)` | Today the decimal cast executes a copy (`vortex-array/src/arrays/decimal/compute/cast.rs`). D15 says a decimal cast to a wider canonical width returns a `Narrow` over its input. If CD4 implements that as a reduce rule, `execute` returns the same `Narrow` again and execution does not converge. Not verified | Depends on CD4 |
| `f7408481ce:vortex-array/src/arrays/narrow/rules.rs:84-97` `CastReduce` | `as_ptype()` on the child dtype | Panic for an unsigned integer target: `is_signed_int()` is false for a decimal parent and for the target, so the signedness check passes and `as_ptype()` panics on the decimal child. NR1 matches `DType::Primitive` on the target and the child instead | No |
| `af4b663059:vortex-array/src/arrays/narrow/rules.rs:111-143` `CastReduce` | `PType::try_from` for target and child | Declines decimals. CD4 adds the decimal cast rules of D15 | Yes |
| `af4b663059:vortex-array/src/arrays/narrow/rules.rs:146-194` fill null, `integer_storage_type` | `PType::try_from`, `i64::try_from`, `min_*_ptype_for_value` | Declines decimals | Yes |
| `af4b663059:vortex-array/src/arrays/narrow/rules.rs:240-278` `zip_values` | `PType::try_from`, `DType::Primitive` | Declines decimals | Yes |
| `af4b663059:vortex-array/src/arrays/narrow/rules.rs:281-314` `TakeIndicesReduce` | Dictionary codes are integers | Not reachable for decimals | Yes |
| `af4b663059:vortex-array/src/arrays/narrow/numeric.rs:70` | `PType::try_from(array.dtype())` | Declines decimals | Yes |
| `af4b663059:vortex-array/src/arrays/narrow/encoding.rs:29` `encode` | Takes a `PrimitiveArray`, rejects non-integers | No decimal producer. CD4 adds one | Yes |

## Briefs

### NR1: In-memory integer `Narrow` encoding

- Goal: Add `Narrow`, an in-memory encoding that stores an integer column as a strictly narrower
  integer child with the same signedness and nullability. Selection, compare, mask, cast and fill
  null pass to the child. `Narrow` has no wire form yet, so the serializer writes its canonical
  array instead (D26).
- Depends on: None. Soft: None. Blocks: NR2, NR3, NR4, NR5, NR6, CD4, CD7.
- Decisions: D8 (one encoding with ID `vortex.narrow`), D15, D18, D22, D23, D25, D26, D28, D38.
- Branch: `ct/narrow-in-memory`, based on `origin/develop` (`afa4bc87cd` or later).
- Scope:
  - Write the encoding, `validate_dtypes`, the compare kernel, the take kernel, and the selection,
    mask, cast and fill-null rules. `f7408481ce` is the reference (D38).
  - Add `NarrowArray::encode_values`, marked `#[deprecated]` (D28). `adc1e83b9d` has a reference
    version.
  - Add the serialization workaround of D26. It is crate-private, in
    `ArraySerializationTree::try_new`: before the serializer lookup, the tree builder lowers a
    `Narrow` to its canonical array. No public trait method is added. `ArraySession` does not
    register `Narrow`, and WR1 adds the registration.
  - Add a self-reduce that flattens a `Narrow` child, so D15 holds after physical rewrites.
- Out of scope: Aggregate kernels (NR3). Arithmetic, between, zip, float and cross-signedness casts
  (NR4). The rules that move structural encodings above `Narrow`, and the dictionary-codes rule
  (NR5). The switch of the existing narrowing call sites to `encode_values` (NR6). The
  `vortex.narrow` wire form and editions (WR1, WR2). Compressor changes (WR3). Decimals (CD4).
  Lowering in the direct callers of `array_serialize`.

Files (checked at `afa4bc87cd`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/narrow/mod.rs` | New. The module doc, with the D15 flatten and the D26 note. The `mod` declarations, the re-exports, `pub struct Narrow`, `initialize`, and `#[cfg(test)] mod tests;` last |
| `vortex-array/src/arrays/narrow/array.rs` | New. Reference: `f7408481ce`. `NarrowArray`, `NarrowSlots` with `#[slot(0)]`, `try_new`, `validate_dtypes` |
| `vortex-array/src/arrays/narrow/vtable.rs` | New. Reference: `f7408481ce`. `VTable` with the flatten in `VTable::reduce`, `OperationsVTable` with `probe_scalar`, and `ValidityChild` |
| `vortex-array/src/arrays/narrow/encode.rs` | New. `NarrowArray::encode`, the deprecated `encode_values` and the `pub(super)` width helper. References: `f7408481ce` and `adc1e83b9d:vortex-array/src/arrays/narrow/encoding.rs:45-90` |
| `vortex-array/src/arrays/narrow/compare.rs` | New. Reference: `f7408481ce`. Compare kernel and `constant_fits_child` |
| `vortex-array/src/arrays/narrow/rules.rs` | New. Reference: `f7408481ce`. `PARENT_RULES` with the slice, filter, mask, cast and fill-null rules, and `wrap_values` |
| `vortex-array/src/arrays/narrow/take.rs` | New. `TakeExecute for Narrow` |
| `vortex-array/src/arrays/narrow/tests/mod.rs` | New. The one module `SESSION` and the shared named fixtures |
| `vortex-array/src/arrays/narrow/tests/encoding.rs` | New. Port the cases of the reference tests, and add the flatten, `encode_values` and serialization tests |
| `vortex-array/src/arrays/narrow/tests/compute.rs` | New. Port the cases of the reference tests, and add the conformance tests |
| `vortex-array/src/serde.rs:172-191` | `ArraySerializationTree::try_new` lowers a `Narrow` before the serializer lookup |
| `vortex-array/src/arrays/mod.rs:85-91` | Add `pub mod narrow;` with `pub use narrow::Narrow;` and `pub use narrow::NarrowArray;` between `masked` and `null` |
| `vortex-array/src/arrays/mod.rs:144-145` | Add `narrow::initialize(session);` between `map` and `null` |
| `vortex-array/src/session/mod.rs` | No change. WR1 registers `Narrow` together with its wire form |

Design notes:

```rust
pub struct Narrow;
pub type NarrowArray = Array<Narrow>;
#[array_slots(Narrow)]
pub struct NarrowSlots {
    #[slot(0)]
    pub values: ArrayRef,
}

impl NarrowArray {
    pub fn try_new(values: ArrayRef, dtype: DType) -> VortexResult<Self>;

    /// Stores `array` as a `Narrow` over a child of the smallest integer type with the same
    /// signedness, or returns it unchanged when no narrower type holds its values.
    pub fn encode(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef>;

    /// Returns `array` at the smallest integer type with the same signedness, without a `Narrow`.
    /// Floats return unchanged.
    #[deprecated(note = "Changes the dtype. Each caller must decide whether it needs a Narrow (#10733)")]
    pub fn encode_values(array: PrimitiveArray, ctx: &mut ExecutionCtx)
        -> VortexResult<PrimitiveArray>;
}
pub(super) fn validate_dtypes(storage: &DType, logical: &DType) -> VortexResult<()>;

// rules.rs, from `f7408481ce:vortex-array/src/arrays/narrow/rules.rs:46-52`.
/// Returns `values` when it has `dtype`, and a `Narrow` of `dtype` over `values` otherwise.
pub(super) fn wrap_values(values: ArrayRef, dtype: DType) -> VortexResult<ArrayRef>;

// compare.rs. The caller handles a null constant first.
/// Returns whether the range of `child` holds `constant`, or `None` when `child` is not a
/// primitive dtype.
pub(super) fn constant_fits_child(constant: &Scalar, child: &DType) -> Option<bool>;
```

NR4 calls `wrap_values`, `constant_fits_child` and the width helper, so their names and signatures
are part of the NR1 contract.

- `validate_dtypes` implements D22: both dtypes are integer primitives, the signedness is equal,
  the child byte width is strictly smaller, and the nullability is equal. `VTable::validate` and
  `try_new` both call it.
- `encode.rs` owns one width helper. It is `pub(super)`, so only the `narrow` module sees it. It
  maps an inclusive range and a signedness to the smallest `PType`, and it is built on
  `PType::min_signed_ptype_for_value` and `PType::min_unsigned_ptype_for_value`
  (`vortex-array/src/dtype/ptype.rs:819-845`), which raise no error on the normal path. `encode`,
  the deprecated `encode_values` and the NR4 numeric kernel all use it. There is no
  `integer_types` list.
- `encode` and `encode_values` share one private, non-deprecated function, so `encode` itself needs
  no `#[expect(deprecated)]`. That function takes `min_max` of the valid values and passes the
  bounds to the width helper. An empty or all-null input gets the 1-byte type of the same
  signedness. A 1-byte input returns unchanged. `encode` fails on a float input, and
  `encode_values` returns a float unchanged. An `I64` column with values 0 to 200 gets an `I16`
  child (D22).
- `encode_values` has no non-test caller in this PR. Its tests carry
  `#[expect(deprecated, reason = "...")]`. CD7 (the DecimalByteParts MSP) and NR6 (the existing
  call sites) call it, each with `#[expect(deprecated, reason = "...")]` (D28).
- `try_new` flattens a `Narrow` child. Physical rewrites keep only the slot dtype
  (`vortex-array/src/array/erased.rs:595-625`), so a rewrite can replace a child with a `Narrow` of
  the same dtype. So the flatten is also a self-reduce in `VTable::reduce`. It returns
  `NarrowArray::try_new(child.values().clone(), dtype)` when the child is a `Narrow`, and `None`
  otherwise. The parent rule set is named `PARENT_RULES`, as in most encodings on `develop`.
- `execute` yields `values.cast(dtype)`, so the child can fuse decoding and widening. Validity
  comes from the child. CD4 changes `execute` for decimals, because the decimal cast of D15 would
  loop.
- `probe_scalar`, as `Extension`, `ZigZag` and `FoR` implement it. `ProbeState::slot` returns
  `VortexResult<Option<ArrayProbe>>` (`vortex-array/src/array/probe/array.rs:238`), so the chain is
  `state.slot(NarrowSlots::VALUES)?.ok_or_else(|| vortex_err!(..))?.execute_scalar(index, ctx)?`,
  then a cast of the scalar to the logical dtype. A null child scalar casts to a null of the
  logical dtype. `scalar_at` delegates through `ProbeState::once`, and `type ProbeState = ()`.
  This cast is the `Narrow` decode. NR1 writes it inline. NR3 moves it into one function that
  `probe_scalar` and the aggregate kernels call.
- The serialization workaround (D26):
  - Today `ArraySerializationTree::try_new` fails with "Array … does not support serialization"
    when `array_serialize` returns `None` (`vortex-array/src/serde.rs:173-178`).
  - A plugin cannot return the wire form of another encoding. `array_serialize` checks that the
    returned ID is one of the plugin's `serialized_ids` (`vortex-array/src/session/mod.rs:144-151`).
  - So the fallback lives in the serializer, and it is crate-private. Before the serializer
    lookup, `try_new` executes a `Narrow` to `RecursiveCanonical`
    (`vortex-array/src/canonical.rs:674`) with `session.create_execution_ctx()`, so that no lazy
    child reaches the serializer. It keeps the statistics of the `Narrow` with
    `StatsSetRef::inherit_from` (`vortex-array/src/stats/array.rs:148`), and it builds the node from
    the lowered array. It lowers once. Any other array without a wire form fails as today.
  - There is no new `VTable` method, no new `ArrayPlugin` method and no new `ArraySessionExt`
    accessor.
  - The code carries `TODO(connor)[Narrow]: remove once vortex.narrow is in core (LT1, #10733)`.
    LT1 deletes the branch. WR1 adds its "ID not permitted" condition at the same site.
  - The direct callers of `array_serialize` do not lower: the Python metadata
    (`vortex-python/src/arrays/mod.rs:150`), the zstd buffers
    (`encodings/zstd/src/zstd_buffers.rs:82`) and the web crate
    (`vortex-web/crate/src/wasm.rs:650`). They get the "not registered for serialization" error,
    and the PR body says so.
- `deserialize` fails with "Narrow deserialization is not supported", because no wire form exists
  yet. `ArraySession::default` does not register `Narrow`. The blanket `ArrayPlugin` impl of a
  `VTable` would also add a reader entry under `vortex.narrow` that can only fail, which hides the
  `allow_unknown` foreign fallback and lists the ID in footers written without editions.
  `serialize` panics with a `TODO(connor)[Narrow]`, because only a session that registers `Narrow`
  itself can reach it. WR1 adds the registration. No compressor produces a `Narrow` in this PR.
- Compare kernel: register it with `CompareExecuteAdaptor`
  (`vortex-array/src/scalar_fn/fns/binary/compare/mod.rs:71`), as
  `vortex-array/src/arrays/dict/vtable/kernel.rs:21` does.
  - Two `Narrow` operands match `DType::Primitive` on both children and compare at the wider
    child. Any other pair declines.
  - A constant inside the child range compares against the child. `constant_fits_child` checks
    the range: it compares the constant with `PValue::min_value` and `PValue::max_value` of the
    child ptype (`vortex-array/src/scalar/typed_view/primitive/pvalue.rs:209-223`), not with a
    failed cast. The kernel declines when it returns `None`.
  - A constant outside the child range gives a constant answer with the child validity. `Eq` is
    false and `NotEq` is true. `Lt`, `Lte`, `Gt` and `Gte` follow from whether the constant is below
    the range, which `constant.partial_cmp(&Scalar::zero_value(constant.dtype()))` decides.
  - A null constant gives all nulls. Follow "Write the kernels so that decimal support is
    additive" above.
- Cast rule in this PR: match `DType::Primitive` on the target and on the child, and decline any
  other dtype. There is no `as_ptype()` and no `PType::try_from` as control flow. Take an integer
  target of the same signedness. If it is no wider than the child, the cast goes to the child with
  the same target. If it is wider, the rule returns a `Narrow` over the child cast to its own ptype
  with the target nullability (D15 as amended), so a cast to non-nullable still fails on nulls.
  Other targets decline. The rule pushes the cast into the child, which holds the validity, so it
  needs no `trivially_cast_nullability` guard and no `CastKernel` (#10798). Write the cast and
  fill-null rules so that NR4 can add target kinds without replacing them.
- Fill null declines when the fill value does not fit the child dtype, so canonical execution
  handles it.
- Take: `TakeExecute for Narrow` (`vortex-array/src/arrays/dict/take.rs:41`), registered with
  `TakeExecuteAdaptor(Narrow)` for the `Dict` parent in `narrow::initialize`, as `Extension` does.
  It gathers the child, executes the gather, and wraps the result in a `Narrow`. A one-line comment
  says why the gather is executed: a lazy `Dict` under a `Narrow` breaks D25 and loops against the
  NR5 `Dict` pull-up rule. NR1 adds no `TakeReduce`.

Steps:

1. Set up the branch in `vortex2`, as "Stack and branches" describes. Do not reuse the `vortex3`
   worktree or its branch.
2. Read `f7408481ce` and its tests with `git show f7408481ce`.
3. Write the commits in this order. Each commit compiles and passes its tests:
   1. The encoding, validation, `encode`, the vtable with `probe_scalar`, and the registration in
      `arrays/mod.rs`.
   2. The D26 lowering in `serde.rs` and its tests.
   3. The flatten.
   4. The selection, mask, cast and fill-null rules, and the take kernel.
   5. The compare kernel.
   6. The deprecated `encode_values`.
   7. The conformance tests.
4. Use the module `SESSION` in every test, and port the cases of the reference tests. Each commit
   has the sign-off line and no co-author line.

Tests:

- Port the cases of these tests in `f7408481ce:vortex-array/src/arrays/narrow/tests/encoding.rs`:
  `test_signed_encoding` and
  `test_unsigned_encoding` (`rstest` width boundaries), `test_null_payloads_and_empty`,
  `test_reject_invalid_dtypes`, `test_validate_slots`, `test_flatten_and_scalar_dtype`,
  `test_encoded_child_execution`, `test_smallest_width_unchanged`, `test_reject_float_encoding`,
  `test_deserialization_unsupported`.
- `test_serialization_unsupported` becomes `test_serialization_writes_canonical`. Serialize a
  nullable and a non-nullable `Narrow`, and decode the bytes. The result is a canonical primitive
  array with the logical dtype and equal values, and its wire ID is the primitive ID. An exact min
  or max that is set on the `Narrow` reads back. Two more cases test the `RecursiveCanonical`
  lowering: a nullable `Narrow` over an encoded child whose validity is not canonical, such as a
  `Dict`, and a struct with a `Narrow` field. Each reads back as a canonical array with equal
  values.
- New, in `narrow/tests/encoding.rs`:
  - `test_encode_values_keeps_signedness`, an `rstest`. `[0i64, 127]` gives `I8`, `[0i64, 200]`
    gives `I16`, `[-1i32, 200]` gives `I16`, an empty `i32` gives `I8`, all-invalid `u32` gives
    `U8`, and `[0u64, u64::MAX]` stays `U64`. Validity is kept in every case.
  - `test_encode_values_preserves_floats`, from `adc1e83b9d`.
  - Both `encode_values` tests carry `#[expect(deprecated, reason = "...")]`.
  - `test_reduce_flattens_nested_narrow`. Build a `Narrow` with `try_from_parts` over a `Narrow`
    child, and call `optimize()`. Check that the result is one `Narrow` over the inner child, with
    the same values.
- The existing `PiecewiseSequence` test
  (`vortex-array/src/arrays/piecewise_sequence/tests.rs:159-182`) stays the regression for an
  array with no wire form. NR1 adds no `serde.rs` test for it.
- Port the cases of these tests in `f7408481ce:vortex-array/src/arrays/narrow/tests/compute.rs`:
  `test_selection_and_fill_null`,
  `test_cast_preserves_checked_nullability`, `test_fill_null_preserves_result_dtype`,
  `test_arithmetic_uses_logical_width`, and `test_comparison`. The last one is an `rstest` over six
  operators. It covers constants on both sides of the child range, both operand orders, and a
  second `Narrow`.
- The ported take case asserts that the optimized root of `slice.take.filter` is a `Dict` over a
  `Narrow`, with equal values. The reference asserts a `Narrow` root, which the take kernel
  changes.
- A compare with a constant outside the child range gives the constant answer with the child
  validity, as the reference does with a `BoolArray`
  (`f7408481ce:vortex-array/src/arrays/narrow/compare.rs:96-107`). The test checks the values and
  the validity for a non-nullable and a partially valid child.
- Conformance: one `rstest` in `narrow/tests/compute.rs` over the named fixtures of
  `narrow/tests/mod.rs`: signed, unsigned, non-nullable, partially valid, all invalid, and one row.
  It runs `test_cast_conformance`, `test_filter_conformance`, `test_take_conformance`,
  `test_mask_conformance` and `test_array_consistency`. The `CastReduce` contract (#10798) needs
  the partially valid case. NR3 and NR4 reuse these fixtures.

Benchmarks: None in this PR. NR2 adds them. No change on existing benchmarks is expected. Record
the `take_patches` results for the NR6 investigation.

Acceptance checks:

- `cargo clippy -p vortex-array --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-array`
- `cargo nextest run -p vortex-array narrow` and `cargo nextest run -p vortex-array serde`
- Serializing a `Narrow` writes its canonical array. The read-back array is a canonical primitive
  array with equal values and the statistics of the `Narrow`.
- No array node with the ID `vortex.narrow` is written, and no footer lists the ID, because
  `ArraySession` does not register `Narrow`.
- `encode_values` carries `#[deprecated]` and has no non-test caller.

Review focus: The constant shortcut of the compare kernel for both operand orders and null
constants. Checked nullability in the cast rule. Nullability of `take` results, which comes from
the gathered child and the indices, and of `mask` results, which follow the child. The take kernel
leaves no lazy `Dict` under a `Narrow` (D25). The fill-null fallback. The flatten in
`VTable::reduce`. The D26 lowering: it runs before the serializer lookup and only for a
`Narrow`, it lowers once, it keeps the statistics of the `Narrow`, and it adds no public API.
Registration of the compare and take kernels in `narrow::initialize`.

Size: Large. The reference commit has about 550 lines of source and 435 lines of tests.
`encode_values`, the D26 lowering, the take kernel, the conformance tests and their tests come on
top.

PR body: say that the direct callers of `array_serialize` do not lower, and that files written
with `disable_editions()` list `vortex.narrow` in the footer.

Open questions: None.

### NR2: Benchmarks for compute on integer `Narrow`

- Goal: Add one benchmark file that runs each `Narrow` operation on three inputs. They are the
  `Narrow`, the equivalent wide canonical array, and the `Narrow` decoded inside the timed region.
  CodSpeed on NR3 to NR5 then shows each kernel against the fallback.
- Depends on: NR1. Soft: None. Blocks: NR3, NR4 and NR5 (their CodSpeed evidence).
- Decisions: D19 (benchmarks land before the changes that they cover), D15.
- Branch: `ct/narrow-benchmarks`, based on NR1.
- Scope: A new `vortex-array/benches/narrow.rs` and its `[[bench]]` entry.
- Out of scope: Kernel changes (NR3 to NR5). Workflow changes. Decimal benchmarks (CD1).

Files (checked at `afa4bc87cd`):

| Path | Change |
| --- | --- |
| `vortex-array/benches/narrow.rs` | New |
| `vortex-array/Cargo.toml:332-334` | Add `[[bench]]` with `name = "narrow"` and `harness = false` after the `probe` entry |

Design notes:

- Follow the conventions of CD1 (`encodings/decimal-byte-parts/benches/dbp_compute.rs`, #10752)
  and of the benchmark guide (`docs/developer-guide/benchmarking.md`):
  - An SPDX header and a module doc.
  - The `MiMalloc` global allocator.
  - A `main` that forces `SESSION` with `LazyLock::force` and then calls `divan::main()`.
  - `SESSION` starts from `VortexSession::empty()` and adds `KernelSession`,
    `AggregateFnSession` and `MemorySession` with `.with::<...>()`, as
    `encodings/alp/benches/alp_compress.rs:66-68` does. `StatsSession` moved out of `vortex-array`
    (#10734), so the old recipe of this brief does not compile. `main` creates the three
    variables, so no timed case pays for their default constructors. Then the `Wide` and `Decoded`
    cases stay flat when NR3 adds registrations.
  - Return the result from `bench_values`. Use `vortex_expect`, not `unwrap`. Give `Input` a
    `Display` impl (`wide`, `narrow`, `decoded`). Use one `bench_case(bencher, input, operation)`
    helper.
- Do not tag any case with `#[vortex_bench_support::cpu_features]`. These kernels are dispatch
  work, so the cases stay in simulation.
- The name `narrow` matches no prefix in `.github/workflows/codspeed.yml:92-104`, so the cases run
  in the "builders" shard. No workflow change is needed.
- Data: `LEN = 8_192` rows, so the wide input holds 64 KiB, as `filter_fixed_width.rs:47-54`
  recommends. Values are `i16` from `StdRng::seed_from_u64`, uniform in `-10_000..=10_000`. The
  `Narrow` input has logical dtype non-nullable `I64` over that `i16` child. The wide input holds
  the same values as `i64`.
- Fresh inputs. `sum` caches its result on the array
  (`vortex-array/src/aggregate_fn/fns/sum/mod.rs:63-65`), and the NR4 arithmetic kernel calls
  `min_max` on both operands, so reused inputs hide work:
  - Keep the `Buffer<i16>` (and a second one from another seed), the `Buffer<i64>`, the selection
    `BitBuffer`, the validity `BitBuffer` of `fill_null_wide` and the index `Buffer<u32>` outside
    the bencher.
  - Inside `with_inputs`, build the arrays with `PrimitiveArray::new` and `NarrowArray::try_new`
    (not `encode`), a new `Mask` and new index arrays.
  - Never clone a built `ArrayRef`, and never use a `static` array. A doc comment on `Input` says
    why, as CD1 does (`dbp_compute.rs:193-197`).

```rust
#[derive(Clone, Copy, Debug)]
enum Input {
    /// The canonical `i64` array with the same values.
    Wide,
    /// A `Narrow` with logical dtype `i64` over an `i16` child.
    Narrow,
    /// The `Narrow`, executed to a `PrimitiveArray` inside the timed region, then the operation.
    Decoded,
}

const INPUTS: &[Input] = &[Input::Wide, Input::Narrow, Input::Decoded];
const NARROW_INPUTS: &[Input] = &[Input::Narrow, Input::Decoded];

#[divan::bench(args = INPUTS)]
fn compare_in_range(bencher: Bencher, input: Input) { /* ... */ }
```

Workloads. Each one ends with `execute::<Canonical>`, or returns the aggregate scalar:

| Bench | Operation | PR whose kernel changes the `Narrow` case |
| --- | --- | --- |
| `compare_in_range` | `binary(Constant(0i64), Lt)` | NR1 (child compare) |
| `compare_out_of_range` | `binary(Constant(100_000i64), Lt)`, above `i16::MAX`. Keep it only if it does real work. NR1 asserts the constant result in a unit test | NR1 (constant answer) |
| `between_in_range` | `between(Constant(-1_000i64), Constant(1_000i64))`, both bounds non-strict | NR4 |
| `filter` | `filter(mask)`, random mask with density 0.5 | NR1 |
| `take` | `take(indices)`, `LEN / 4` random `u32` indices | NR1 (take kernel) |
| `slice` | `slice(LEN / 4..3 * LEN / 4)`, only `Narrow` and `Decoded` inputs | NR1 |
| `sum` | `sum(&array, ctx)` with the default options (`skip_nans`), the only configuration that NR3 serves | NR3 |
| `min_max` | `min_max(&array, ctx, NumericalAggregateOpts::default())`. It covers the extrema kernel | NR3 |
| `add_constant` | `binary(Constant(1i64), Add)` | NR4 |
| `add_narrow` | `Narrow + Narrow`, second operand from another seed | NR4 |
| `add_wide` | `Narrow + wide i64 array`, only `Narrow` and `Decoded` inputs. The wide operand is the `Wide` input from the second seed | NR4 (the kernel declines it) |
| `cast_f64` | `cast` to `F64`, a float target that NR4 sends to the child | NR4 |
| `fill_null_wide` | A nullable input from the same values with about 10% nulls, filled with a value that fits `i32` but not `i16` | NR4. NR1 declines it, and NR4 fills at `I32` |
| `zip` | `zip` of two inputs from the two seeds, with a random mask | NR4 |
| `chunked` | `execute::<Canonical>` of a `ChunkedArray` of four inputs of `LEN / 4` rows | None. It guards the decode of each `Narrow` chunk |
| `interleave` | `execute::<Canonical>` of an `InterleaveArray` of two inputs of `LEN / 2` rows, `array_indices = i % 2` as `u8`, `row_indices = i / 2` as `u32` | None. It guards the interleave kernels |

- `Decoded` for `chunked` and `interleave` decodes each input, then builds the parent.
- `Decoded` for `add_narrow` and `zip` decodes both operands. `Wide` for `add_narrow` adds two
  `i64` arrays. `add_wide` omits `Wide`, because it is the same case.

Steps:

1. Run `gh stack add ct/narrow-benchmarks` on top of NR1, as "Stack and branches" describes.
2. Add `vortex-array/benches/narrow.rs` and the `[[bench]]` entry.
3. Commit as `bench(array): add Narrow compute benchmarks`, with the sign-off line. The PR title
   comes from the table in "Stack and branches".

Tests: None. The benchmark compiles under `cargo clippy --all-targets`.

Benchmarks: This PR only adds cases, so CodSpeed reports them as new. Read the reports of later PRs
with the CodSpeed rule in "Stack and branches". Qualitative expectations for later PRs:

- NR3: `sum[Narrow]` and `min_max[Narrow]` move from about the `Decoded` level toward or below
  `Wide`.
- NR4: `add_constant`, `add_narrow`, `between_in_range`, `cast_f64`, `fill_null_wide` and `zip`
  change for `Narrow`. `add_wide[Narrow]` stays near `add_wide[Decoded]`, because the NR4 kernel
  declines a wide operand and canonical execution widens it once.
- `chunked` and `interleave` stay flat. They call no aggregate, and NR5 changes no tree that they
  build.
- `Wide` and `Decoded` cases must stay flat in NR3 to NR5. A move there points at noise or at a
  side effect of session registration.

Acceptance checks:

- `cargo clippy -p vortex-array --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-array`
- `cargo bench -p vortex-array --bench narrow` runs every case.
- Each case does at least a few µs of real work and stays under 1 ms in CodSpeed simulation, as
  the benchmark guide requires (`docs/developer-guide/benchmarking.md`).

Review focus: The three inputs hold the same logical values. The arrays are built inside
`with_inputs` from shared buffers, fresh for each iteration, and no built `ArrayRef` is cloned.
`Decoded` decodes inside the timed closure. `main` forces the session.

Size: Medium. One file with 16 workloads.

Open questions: None.

### NR3: Aggregate kernels read the child

- Goal: Add aggregate kernels for `Narrow` that build the exact partial of the parent dtype from
  the child, so that `min_max`, `is_constant`, `is_sorted` and `sum` stop widening the child
  buffer. An aggregate on a `Narrow` runs in the accumulator of the parent dtype (D36). Every other
  aggregate keeps the default path.
- Depends on: NR1, NR2 (stack, D35: the benchmarks exist before the kernels change). Soft: None.
  Blocks: NR4 (stack), CD4.
- Decisions: D15, D36, D38.
- Branch: `ct/narrow-aggregate-kernels`, based on NR2.
- Scope: Four kernels in `narrow/aggregates.rs` and their registrations in
  `AggregateFnSession::default`. A shared `SumState::to_scalar`, a crate-private kernel lookup by
  exact id, and two doc comments in the aggregate code. `8d548646ee` is the reference for the test
  oracle and the test cases only.
- Out of scope: Decimal aggregates (CD4). Pushdown of sum, min and max into the MSP (LT3). Kernels
  for `SumV2` and for `Sum` with `include_nans`. Grouped aggregate kernels. A fast path for
  `UncompressedSizeInBytes`. The engine issues in "Issues to draft at the gate" below.

The rule (D36). An aggregate over a `Narrow` runs in the accumulator of the parent dtype. Every
partial that reaches it is exactly the partial that the parent accumulator builds from the decoded
values. No accumulator and no aggregate vtable receives a batch of another dtype, and no partial
state is cast. A `Narrow` kernel can read the child in two ways only:

- It asks the child for values or a verdict that the `Narrow` decode does not change (min and max,
  is constant, is sorted), and it decodes any value with the `Narrow` decode.
- It sums the stored values into the sum state of the parent, and it widens each value exactly.

The precedent is `vortex-array/src/aggregate_fn/fns/min_max/extension.rs:24-32`, which runs
`min_max` on the storage and wraps the two values again. A kernel that cannot prove the parent
partial declines.

The old algorithm of this brief, which fed the child batch to an accumulator of the parent dtype,
cannot work: the accumulator rejects a batch of another dtype with "Input DType mismatch"
(`vortex-array/src/aggregate_fn/accumulator.rs:237`). The reference `8d548646ee` accumulates at the
child dtype and casts the partial, which gives a decimal sum a lower precision than the parent.

Kernels:

| Kernel | Aggregates | How |
| --- | --- | --- |
| `NarrowMinMaxKernel` | `MinMax`, `Min`, `Max` | `min_max(child)`, decode both values, and build the parent partial with `make_minmax_dtype(parent)`, in the shape of `DictMinMaxKernel`. `Min` and `Max` take one decoded value as a nullable scalar of the parent dtype. A child with no valid value gives the null partial |
| `NarrowIsConstantKernel` | `IsConstant` | `batch.is_empty() \|\| is_constant(child)`, then `IsConstant::make_partial` on the `Narrow`. The empty guard is required: the free `is_constant` reports an empty array as not constant (`aggregate_fn/fns/is_constant/mod.rs:99-102`) |
| `NarrowIsSortedKernel` | `IsSorted` | The verdict follows the vtable semantics, not the length shortcut of the free function (`aggregate_fn/fns/is_sorted/mod.rs:87-89`). An all-null batch gives `!strict`, as `is_sorted/primitive.rs:35` does. Otherwise `is_sorted` or `is_strict_sorted` on the child, then `IsSorted::make_partial` on the `Narrow` |
| `NarrowSumKernel` | `Sum` with `skip_nans` only | Declines, cheapest check first (below). Otherwise executes the child to a `PrimitiveArray` and calls `accumulate_primitive` into `make_zero_state` of the parent return dtype |

Design notes:

- One decode function in `narrow/` casts a child scalar to the parent dtype. NR3 moves the cast
  of NR1's `probe_scalar` into it, and `probe_scalar` and the kernels call it.
- `NarrowSumKernel` is bit-exact with the parent, because the stats bridge also sums each batch
  from zero and then merges (`accumulator.rs:265-283`). Only `Sum` with `skip_nans` has that bridge
  (`vortex-array/src/expr/stats/mod.rs:223`). The kernel declines, cheapest check first:
  1. Options without `skip_nans`.
  2. A child with a kernel for the exact `Sum` id, such as `RunEnd` or `Sparse`. On the default
     path the cast pushes into such a child, and its own kernel runs at the parent dtype. The
     fallback of `Chunked` for the `None` key (`session.rs:132`) does not count, because it sums
     widened chunks. `find_aggregate_kernel` always falls back to the `None` key
     (`session.rs:328-337`), so the check uses the exact-id lookup below.
  3. `values.is::<Constant>()`, which executes nothing. The default path sums a constant cheaply.
  4. A non-integer child (a decimal, after CD4).
  5. More than 2^32 values. An `NB:` comment beside the check explains the bound: a child is at
     most 32 bits wide, so no prefix sum of at most 2^32 values leaves the 64-bit range, and
     neither path overflows.
- The kernel returns the partial of its batch alone, as every aggregate kernel does. The
  accumulator merges it with the checked add of `Sum`.
- Registration: each kernel is registered for its own aggregate id in `AggregateFnSession::default`
  (`vortex-array/src/aggregate_fn/session.rs`), after the `Dict` kernels (`:133-135`). That is
  `(Narrow, MinMax)`, `(Narrow, Min)` and `(Narrow, Max)` for `NarrowMinMaxKernel`, then
  `(Narrow, IsConstant)`, `(Narrow, IsSorted)` and `(Narrow, Sum)`.
  - Never register them from `narrow::initialize`. `KernelSession::default` runs
    `arrays::initialize` on a throwaway session (`vortex-array/src/optimizer/kernels.rs:345-346`),
    so an aggregate registration there is lost without an error.
  - There is no `(Narrow, None)` fallback. The lookup tries the exact id and then the `None` key
    (`session.rs:328-337`), so such a fallback runs for every aggregate on a `Narrow`. It shadows
    the cheap `try_accumulate` paths of `First`, `Last`, `Count`, `NullCount`, `AllNull` and
    `AllNonNull`, and it covers future aggregates without a review.
- Default path, no kernel: `SumV2`, `Sum` with `include_nans`, `First`, `Last`, `Count`,
  `NullCount`, `AllNull`, `AllNonNull`, `BoundedMin`, `BoundedMax` and `UncompressedSizeInBytes`.
  The default path executes the `Narrow` into a cast to the parent dtype. `NanCount`, `AllNan` and
  `AllNonNan` do not apply to integers. Grouped aggregates take the default grouped path.
  - `SumV2` and `Sum` with `include_nans` have no stats bridge. They add each value into the
    running state, and they check after every value for 64-bit input. A kernel that returns the
    partial of one batch does not match them near the `i64` bound.
- Changes outside `narrow/`:
  - `SumState::to_scalar` in `aggregate_fn/fns/sum/mod.rs` replaces the private `sum_state_scalar`
    of `SumV2` (`aggregate_fn/fns/sum_v2/mod.rs:523-540`). `Sum`, `SumV2` and the kernel use it.
    No behavior changes.
  - A `pub(crate)` method on `AggregateKernels` (`aggregate_fn/session.rs:298-326`) looks up the
    key `(array_id, Some(agg_fn_id))` only, with no `None` fallback. The kernel calls it through
    `ctx.aggregate_kernels()` (`vortex-array/src/executor.rs:408`), and the test oracle uses it.
  - A doc comment on `accumulate_primitive` (`aggregate_fn/fns/sum/primitive.rs:24`) states the
    contract that the kernel relies on. It adds the valid values to the state and widens each value
    exactly. The input is in the same class as the state (signed, unsigned or float). It returns
    true on overflow.
  - One sentence on `AggregateFnVTable::accumulate` (`aggregate_fn/vtable.rs:253`) says that
    `batch` has the dtype `args.dtype`.
- Child value queries are allowed with the `Narrow` decode. When the child has no cached
  statistic and no kernel, `min_max`, `is_constant` and `is_sorted` build an accumulator of the
  child dtype inside. They return only two stored values or a bool, and they cache it on the
  child. No child partial reaches the parent.

Files (checked at `afa4bc87cd`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/narrow/aggregates.rs` | New. The module doc owns the rule. Four `pub(crate)` kernels, each with a short exactness argument in its doc |
| `vortex-array/src/arrays/narrow/mod.rs` | `pub(crate) mod aggregates;`. The module doc names the aggregate kernels |
| `vortex-array/src/arrays/narrow/vtable.rs` | Move the cast of `probe_scalar` into the `pub(super)` decode function, which the kernels also call |
| `vortex-array/src/aggregate_fn/session.rs:45-51` | Import `Narrow` and the four kernels |
| `vortex-array/src/aggregate_fn/session.rs:133-135` | The six registrations after the `Dict` kernels |
| `vortex-array/src/aggregate_fn/session.rs:298-326` | The `pub(crate)` exact-id lookup on `AggregateKernels` |
| `vortex-array/src/aggregate_fn/fns/sum/mod.rs` | `SumState::to_scalar`. `Sum::to_scalar` uses it |
| `vortex-array/src/aggregate_fn/fns/sum_v2/mod.rs:523-540` | Use `SumState::to_scalar`, and delete `sum_state_scalar` |
| `vortex-array/src/aggregate_fn/fns/sum/primitive.rs:24` | The doc comment on `accumulate_primitive` |
| `vortex-array/src/aggregate_fn/vtable.rs:253` | One sentence on the `accumulate` doc |
| `vortex-array/src/arrays/narrow/tests/mod.rs` | `mod aggregates;` |
| `vortex-array/src/arrays/narrow/tests/aggregates.rs` | New |
| `encodings/fastlanes/tests/narrow.rs` | New integration test with a `BitPacked` child, with the closure session pattern. It keeps `#![cfg(test)]`, or clippy fails on `tests_outside_test_module` |

Steps:

1. Run `gh stack add ct/narrow-aggregate-kernels` on top of NR2.
2. Add `SumState::to_scalar`, the exact-id lookup and the two doc comments. No behavior changes.
3. Move the cast of `probe_scalar` into the decode function. Write `narrow/aggregates.rs` and the
   registrations in `aggregate_fn/session.rs`.
4. Write the tests with the module `SESSION`, and the fastlanes integration test with the closure
   session.
5. Commit with the sign-off line.

Tests:

- The oracle helper `assert_aggregate_batches`, ported from
  `8d548646ee:vortex-array/src/arrays/narrow/tests/aggregates.rs:44-77`. For each batch, it checks
  the kernel partial, the running partial and the final value against a fresh parent accumulator on
  the canonical batch. It looks kernels up with the exact-id lookup through
  `ctx.aggregate_kernels()`, so a registration in the wrong place fails a test.
  - The reference asserts that every batch encodes to a `Narrow` and that the kernel returns a
    partial (`:60-65`). The port accepts a mix of `Narrow` and canonical batches, and it checks the
    kernel partial only on the `Narrow` batches.
  - For a declined configuration, the lookup misses or the kernel returns `None`. The oracle then
    compares only the running partials and the final value.
- Shapes: `i64` over `i8`, `i16` and `i32`, and `u64` over `u8`, `u16` and `u32`, with the child
  bounds. A descending and an equal batch boundary. Empty, all-null and partially valid batches.
- Strict `IsSorted` on `[null]` alone, and followed by `[1, 2]`.
- `IsConstant` on `[[], [7, 7]]`. This catches a missing empty guard.
- A `ChunkedArray` of a `Narrow` chunk and a `PrimitiveArray` chunk gives the canonical result.
- Running overflow: `[i64::MAX - 5]` then `[10]`, and `[i64::MAX - 5]` then `[10, -10]`. No
  narrower child holds `i64::MAX - 5`, so the first batch is a canonical `i64` batch, and the later
  batches are `Narrow`. `Sum` with `skip_nans` and each declined configuration (`SumV2`, `Sum` with
  `include_nans`) match canonical.
- Statistics land on the right array: the parent dtype on the `Narrow`, the child dtype on the
  child.
- A default-path test: the kernel lookup misses, and the result equals canonical, for `First`,
  `Last`, `Count`, `NullCount`, `AllNull`, `AllNonNull`, `BoundedMin`, `BoundedMax` and
  `UncompressedSizeInBytes`. `UncompressedSizeInBytes` uses `UncompressedSizeOpts` and reports the
  logical width.
- `encodings/fastlanes/tests/narrow.rs`: a `BitPacked` child. Port the reference test (filter,
  take, compare, sum and execution), and add `min_max`, `is_constant`, `is_sorted` and `sum`
  against the canonical array.

Benchmarks: NR2 `sum` and `min_max` with the `Narrow` input are expected to improve. All other
cases stay flat.

Acceptance checks:

- `cargo clippy -p vortex-array -p vortex-fastlanes --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-array -p vortex-fastlanes`
- `cargo nextest run -p vortex-array narrow`
- `cargo nextest run -p vortex-fastlanes --test narrow`
- Every kernel partial, running partial and final value equals the result of the parent
  accumulator on the canonical batch.
- No accumulator receives a batch of another dtype, and no partial state is cast.

Review focus: The exactness of each kernel against the parent partial: nulls, empty batches, batch
boundaries, and strict `IsSorted` on an all-null batch. The order and cost of the `Sum` declines,
the exact-id lookup of the child kernel, and the `NB:` comment on the 2^32 bound. Exact-id
registrations in `AggregateFnSession::default`, with no `None` fallback. Every value goes through
the one `Narrow` decode function. Statistics land on the right array.

Size: Medium. About 150 lines of source in `narrow/aggregates.rs`, small changes in the sum
modules, and the tests.

Decimal readiness for CD4: The extrema, `IsConstant` and `IsSorted` kernels are exact for decimals
as they are. The sum kernel declines decimal children. CD4 can add a decimal branch for `Sum` with
`skip_nans` only.

Issues to draft at the gate. Read the code first to confirm each one:

- `Sum` and `SumV2` handle overflow at a batch boundary in different ways, depending on whether
  the stats bridge runs. The `RunEnd` and `Sparse` sum kernels have the same gap.
- The length shortcut of `is_strict_sorted` disagrees with the `IsSorted` vtable for one null.
  `DictIsSortedKernel` has the same gap.
- `FoRIsConstantKernel` and `DictIsConstantKernel` call `is_constant` without an empty guard. Not
  confirmed yet.

Open questions: None.

### NR4: Arithmetic and scalar operations forward to the child

- Goal: Execute checked add, subtract, multiply and divide at the smallest width that holds both
  operands and the result interval, when each operand is a `Narrow` or a constant. Push `between`,
  `zip`, `fill_null` and more casts through `Narrow`.
- Depends on: NR3 (stack). Soft: NR2. Blocks: NR5.
- Decisions: D15, D22, D38.
- Branch: `ct/narrow-arithmetic-kernels`, based on NR3.
- Scope: Write the numeric kernel and the rules. `74aa17644e` is the reference for the interval
  arithmetic and for test cases. Extend the cast and fill-null rules of NR1 instead of replacing
  them. Reuse the NR1 width helper.
- Out of scope: Structural rules and the dictionary-codes rule (NR5). Decimal arithmetic (D20).
  Decimal casts (CD4).

Files (checked at `afa4bc87cd`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/narrow/numeric.rs` | New. Reference: `74aa17644e`. A kernel for `Binary` that implements `ExecuteParentKernel` directly |
| `vortex-array/src/arrays/narrow/rules.rs` | Extend the cast and fill-null rules. Add `ZipReduce`, `ZipFalseReduce` and `BetweenReduce` |
| `vortex-array/src/arrays/narrow/encode.rs` | Created by NR1. The numeric kernel, `integer_storage_type` and `encode` use its `pub(super)` width helper. There is no `integer_types` list |
| `vortex-array/src/arrays/narrow/mod.rs` | `mod numeric;`, `initialize` registers the numeric kernel |
| `vortex-array/src/scalar_fn/fns/between/mod.rs:129` | Make `as_two_compares` `pub(crate)` |
| `vortex-array/src/arrays/narrow/tests/mod.rs` | `mod numeric;` and `mod pushdown;` |
| `vortex-array/src/arrays/narrow/tests/numeric.rs` | New. Port the cases of the reference tests |
| `vortex-array/src/arrays/narrow/tests/pushdown.rs` | New. Port the cases of the reference tests |

Design notes:

- Numeric kernel: it implements `ExecuteParentKernel` directly and handles child index 0 and 1,
  for `Add`, `Sub`, `Mul` and `Div`.
  - It declines with `Ok(None)` unless each operand is a `Narrow` or a constant. Canonical
    execution then widens a wide or lazy operand once.
  - It executes each stored operand once with `execute::<Columnar>`, which keeps a constant
    constant. It takes `min_max` of the executed array as `i128` bounds, and it passes the executed
    arrays to the cast and to the checked `execute_numeric`.
  - It computes the result interval, and it picks the smallest type of the same signedness, no
    wider than the logical type, that holds both operands and the interval, with the NR1 width
    helper. It wraps the result with the NR1 `wrap_values`.
  - When no narrower type holds the interval, it runs at the logical width, so the checked kernel
    reports a real overflow as canonical execution does.
  - An empty parent gives an empty canonical array. An all-null operand gives a null constant.
- Declines are pattern matches, not errors built for control flow. For example:
  `let DType::Primitive(..) = dtype else { return Ok(None) };`.
- Cast rule (D15): a float target goes to the child with the same target. An integer target
  first picks a storage type. It is the child type with the signedness of the target. For an
  unsigned child and a signed target, it is the next wider signed type. A target no wider than the
  storage type goes to the child with the same target. A wider target returns a `Narrow` over the
  child cast to the storage type. Non-primitive targets decline.
- Fill null: `integer_storage_type` picks the child type, or the smallest type that holds the fill
  value, without reading values. The child is cast to it before the fill. A fill value that needs
  the logical width gives a plain array.
- Zip: both branches. The other branch is a `Narrow` or a constant. Otherwise decline.
- Between: when both bounds are constants inside the child range, run `between` on the child.
  `constant_fits_child` from NR1 checks each constant against the child range. When both bounds
  are constants or `Narrow` arrays, call `as_two_compares`, so each compare uses the NR1 kernel.
  Otherwise decline, so that `Between::execute` widens the `Narrow` once.
- Decimal declines: the numeric kernel (on the logical dtype, before it reads bounds), the cast
  rule (on the target and the child), `integer_storage_type` (fill null and zip with a constant)
  and the zip of two `Narrow` operands decline a non-primitive dtype. `between` needs no decline,
  because the NR1 compare kernel is decimal-safe. CD4 adds the decimal tests.

Steps:

1. Run `gh stack add ct/narrow-arithmetic-kernels` on top of NR3.
2. Write the commits in this order. Each commit compiles and passes its tests: the numeric kernel,
   the cast extension, fill null, zip, and between.
3. Port the test cases of `74aa17644e` into the commit of each change, with the module `SESSION`.
4. Each commit has the sign-off line.

Tests:

- `narrow/tests/numeric.rs`:
  - `test_signed_arithmetic`: `rstest` over width growth, skipped widths, full width, division at
    `i8::MIN` and across zero, and an overestimated interval.
  - `test_signed_domain`: every `i8` value against small divisors, for all four operators.
  - A brute-force reference test of the interval function: for all pairs of small `i8` intervals,
    the computed interval holds every result.
  - Regression cases: the multiplication of two negative intervals, a zero divisor behind a null,
    an all-zero divisor, and `Constant(i64::MIN) / Narrow(-1)`.
  - `test_logical_errors`: overflow and division by zero give the same error messages as
    canonical execution.
  - `test_unsigned_arithmetic`, `test_unsigned_underflow`, `test_nulls_and_empty`,
    `test_encoded_child_and_operand_order`, `test_overflow_at_each_logical_width`,
    `test_logical_overflow_behind_null`.
  - `test_binary_numeric_array` runs over the NR1 fixtures.
- `narrow/tests/pushdown.rs`:
  - `test_between_bounds`: `rstest` over bounds inside and outside the child range, and both
    strictness values.
  - `test_cast_child`: unsigned to signed, signed to unsigned, and float. The float target needs
    this test, because `test_cast_conformance` runs no float target for an integer input.
  - `test_cast_conformance` runs over the NR1 fixtures.
  - Cast failures assert `is_err`, because the error text names the child ptype.
  - `test_fill_width`, `test_zip_width`, `test_between_nullable_bounds`,
    `test_zip_false_branch_and_null_mask`, `test_signedness_cast_checks_values_and_nulls`.

Benchmarks: NR2 `add_constant`, `add_narrow`, `between_in_range`, `cast_f64`, `fill_null_wide` and
`zip` change for the `Narrow` input. `add_wide[Narrow]` stays near `add_wide[Decoded]`, because the
kernel declines a wide operand.

Acceptance checks:

- `cargo clippy -p vortex-array --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-array`
- `cargo nextest run -p vortex-array narrow`
- Every arithmetic result equals the canonical result, including errors and nulls.
- The brute-force interval test passes, and the arithmetic error messages equal the canonical
  messages.

Review focus: Interval arithmetic for division across zero and for `i64::MIN / -1`. Overflow
checks stay with the checked kernel at the logical width. The cast rule for unsigned children and
signed targets. Each stored operand executes once, and the kernel declines a wide or lazy operand.
Declines build no errors. Null payloads outside the chosen width.

Size: Large. The reference has about 450 lines of source and 425 lines of tests.

Open questions: None.

### NR5: Structural encodings move above `Narrow`, and dictionary codes use the child

- Goal: Keep generic structural encodings above `Narrow` (D25). Reduce rules turn a `Narrow` over
  `Dict`, `RunEnd`, `Sparse`, `Constant`, `Masked`, `Chunked` or `Interleave` into that encoding
  over `Narrow` children. Dictionary codes that are a `Narrow` use its child directly.
- Depends on: NR4. FX3 is merged (`dcb4f50ae2`). Soft: NR2. Blocks: WR3.
- Decisions: D25, D15, D23, D38.
- Branch: `ct/narrow-structural-rules`, based on NR4, with `gh stack add` on the top branch.
- Scope: Seven pull-up rules, the dictionary-codes rule `TakeIndicesReduce` (reference:
  `af4b663059`), and the case of the `Narrow` regression test from `7e2e5e4d7a`.
- Out of scope: The chunked and interleave merge rules of `af4b663059` (`narrow/merge.rs`). They
  put `Narrow` above `Chunked` and `Interleave`, so they would loop against the new rules. The same
  pull-up rule for every other encoding, which a follow-up issue tracks. Decimal tests (CD4).

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/narrow/rules.rs` | The pull-up rules for `Dict`, `Constant`, `Masked`, `Chunked` and `Interleave`. `TakeIndicesReduce` for `Dict` codes, from `af4b663059` |
| `vortex-array/src/arrays/narrow/tests/mod.rs` | `mod structural;` |
| `vortex-array/src/arrays/narrow/tests/structural.rs` | New. The pull-up tests, `test_narrowed_take_indices` from `af4b663059`, and `test_fill_null_after_narrowing_take_indices` from `7e2e5e4d7a`, with the module `SESSION` |
| `encodings/runend/src/` | A parent-reduce rule that moves `RunEnd` above a `Narrow` parent, with a test |
| `encodings/sparse/src/` | A parent-reduce rule that moves `Sparse` above a `Narrow` parent, with a test |

Design notes:

| Input | Output |
| --- | --- |
| `Narrow(Dict(codes, values))` | `Dict(codes, Narrow(values))` |
| `Narrow(RunEnd(ends, values))` | `RunEnd(ends, Narrow(values))` |
| `Narrow(Sparse(indices, values, fill))` | `Sparse(indices, Narrow(values), fill)`, with `fill` cast to the parent dtype |
| `Narrow(Constant(value))` | `Constant(value)`, with `value` cast to the parent dtype |
| `Narrow(Masked(child, validity))` | `Masked(Narrow(child), validity)` |
| `Narrow(Chunked(c1, ..., cn))` | `Chunked(Narrow(c1), ..., Narrow(cn))` |
| `Narrow(Interleave(selectors, v1, ..., vn))` | `Interleave(selectors, Narrow(v1), ..., Narrow(vn))` |

- Every output is valid if each new `Narrow` gets the right dtype. The array that it wraps can have
  another nullability than the structural array above it. `Masked` has the nullable dtype of its
  child (`vortex-array/src/arrays/masked/array.rs:84`), so its child is often non-nullable. `Dict`
  values can be non-nullable under nullable codes, and `Sparse` values can be non-nullable under a
  null fill. So each new `Narrow` takes the parent dtype with the nullability of the array that it
  wraps: `parent.with_nullability(inner.dtype().nullability())`. Then `validate_dtypes` holds. A
  cast of a scalar from the child dtype to the parent dtype is exact.
- Reduce rules have no `ExecutionCtx`, but `MaskedArray::try_new`, `RunEndArray::try_new`,
  `RunEndArray::new`, `SparseArray::try_new` and `Patches::new` need one. Use the constructors
  without a context:
  - `Masked`: `MaskedArray::new_unchecked`. Its safety needs a child without nulls. The new
    `Narrow` has the validity of the old child, so it holds.
  - `RunEnd`: `RunEnd::new_unchecked(ends, values, offset, length)`, with the ends, offset and
    length of the input.
  - `Sparse`: `patches.map_values(..)` (`vortex-array/src/patches.rs:972`), then
    `SparseArray::try_new_from_patches`.
  - `Dict`, `Chunked` and `Interleave`: their `try_new` takes no context.
- Where the rules live. `Dict`, `Constant`, `Masked`, `Chunked` and `Interleave` are in
  `vortex-array`, so their rules are `Narrow` reduce rules that match the child encoding. `RunEnd`
  and `Sparse` are in their own crates, which depend on `vortex-array`. So their rules are
  parent-reduce rules on `RunEnd` and `Sparse` with `Narrow` as the parent. Whether the rule
  registry supports both shapes as described is Not verified. Read how `vortex-array/src/arrays/`
  registers parent rules before the first commit.
- No rule moves `Narrow` above a structural encoding. A rule in the other direction would loop.
  NR1 has no `TakeReduce` rule for this reason. Its `TakeExecute` kernel executes the gather of
  the child and wraps the result, so no lazy `Dict` sits under a `Narrow`. So the `Dict` pull-up
  rule and `TakeIndicesReduce` do not loop against the take of NR1.
- Dictionary codes: a `Narrow` codes child is replaced by its child, with the same `DictData`.
  Narrowed codes can make an unreferenced fill index exceed the codes range. FX3 makes the
  dictionary `fill_null` kernel decline in that case, and it is on `develop` (`dcb4f50ae2`).
- The rules never read the child dtype, so they also hold for decimal `Narrow`. CD4 adds the
  decimal tests.
- WR3 can wrap a column root in `Narrow`, because these rules then move a structural encoding back
  to the top. The dict-layout probe and the Arrow and DuckDB exporters then see `Dict` and `RunEnd`
  at the root, as today.

Steps:

1. FX3 is on `develop` (`dcb4f50ae2`), and NR1 to NR4 are based on it. Run
   `gh stack add ct/narrow-structural-rules` on top of NR4.
2. Add the five rules in `vortex-array`, with their tests.
3. Add the `RunEnd` and `Sparse` rules in their crates, with one test each.
4. Write `TakeIndicesReduce`, with `af4b663059` as the reference, and port the cases of
   `test_narrowed_take_indices`.
5. Port the case of `test_fill_null_after_narrowing_take_indices` from `7e2e5e4d7a`.

Tests:

- `test_pull_up_structural`, an `rstest` over the five `vortex-array` shapes. Each result has the
  structural encoding at the root and a `Narrow` below it. Its values equal the input, checked with
  `assert_arrays_eq!`.
- `test_pull_up_nullable`: nullable children, a null constant, `Masked` with nulls over a
  non-nullable child, `Dict` with nullable codes over non-nullable values, and `Sparse` with a null
  fill over non-nullable values.
- `test_pull_up_fixed_point`: `optimize()` on a `Chunked` of `Narrow` chunks returns the same tree.
  So no rule pair loops.
- In `encodings/runend` and `encodings/sparse`: `Narrow` over the encoding optimizes to the encoding
  over `Narrow` values, with equal values.
- `test_narrowed_take_indices`, from `af4b663059`.
- `test_fill_null_after_narrowing_take_indices`, from `7e2e5e4d7a`. It fails without FX3.

Benchmarks: NR2 `chunked[Narrow]` and `interleave[Narrow]` keep their baseline. Their `Narrow`
input already is a `Chunked` or `Interleave` of `Narrow` chunks, so the new rules change nothing
for them. Kernels push work into each chunk.

Acceptance checks:

- `cargo clippy -p vortex-array -p vortex-runend -p vortex-sparse --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-array -p vortex-runend -p vortex-sparse`
- `cargo nextest run -p vortex-array narrow`, `cargo nextest run -p vortex-runend` and
  `cargo nextest run -p vortex-sparse`
- After `optimize()`, no `Narrow` sits directly above one of the seven encodings.

Review focus: The fill and constant casts to the parent dtype. Nullability of each new `Narrow`.
No rule pair loops. The dictionary rule copies `DictData`. Each `unsafe` constructor has a SAFETY
comment that names why its precondition holds.

Shares files with CD4: both edit `vortex-array/src/arrays/narrow/rules.rs`. Do not run NR5 and CD4
at the same time.

Size: Medium. Seven small rules in three crates, and their tests.

Open questions: None.

### NR6: The existing narrowing call sites move to `encode_values`

- Goal: Replace `PrimitiveArrayExt::narrow` with the deprecated `NarrowArray::encode_values` of NR1
  at every call site. Each call site carries `#[expect(deprecated)]` with a reason, so every
  narrowing that changes a dtype is marked for a later decision (D28). Internal children keep
  their signedness.
- Depends on: NR1. Soft: NR5 (merge after the stack, D35). Blocks: None.
- Decisions: D28, D22, D23, D11.
- Branch: `ct/narrow-encode-values`, based on `develop` after NR1 merges.
- Scope: The call sites that part 1 of `adc1e83b9d` changed, with that commit and `0d94e44a91` as
  references. The
  removal of `PrimitiveArrayExt::narrow`, with its tests ported. Regenerated golden snapshots. A
  check that the latest release reads files that this PR writes.
- Out of scope: `encode_values` itself (NR1). Part 2 of `adc1e83b9d` (WR1 to WR3). Which arrays a
  compressor wraps in a `Narrow` (deferred, D27).

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/narrow/tests/encoding.rs` | Port the narrowing tests of `PrimitiveArrayExt::narrow` to `encode_values` |
| `vortex-array/src/arrays/primitive/array/mod.rs:157-241` | Remove `PrimitiveArrayExt::narrow` and imports that become unused |
| `vortex-array/src/arrays/primitive/array/cast.rs:38-213` | Remove the narrowing tests, which move to `narrow/tests/encoding.rs` |
| `vortex-array/src/builders/dict/mod.rs:19,87` | Dictionary codes |
| `encodings/runend/src/compress.rs:17,91` | Run ends |
| `encodings/fastlanes/src/rle/array/mod.rs:128,442` | Test call site |
| `vortex-btrblocks/src/schemes/patches.rs:9,21,42` | Patch indices and chunk offsets |
| `vortex-btrblocks/src/schemes/binary/varbin.rs:21,87` | VarBin offsets |
| `vortex-btrblocks/src/schemes/float/sparse.rs:13,101` | Sparse indices |
| `vortex-btrblocks/src/schemes/integer/rle.rs:13,65,78` | RLE indices and offsets |
| `vortex-btrblocks/src/schemes/integer/sparse.rs:15,192` | Sparse indices |
| `vortex-btrblocks/src/schemes/string/fsst.rs:17,86,100` | FSST lengths and code offsets |
| `vortex-btrblocks/src/schemes/string/onpair.rs:13,145` | OnPair children |
| `vortex-btrblocks/src/schemes/string/sparse.rs:13,109` | Sparse indices |
| `vortex-btrblocks/src/schemes/temporal.rs:16,95,105,115` | DateTimeParts days, seconds, subseconds |
| `vortex-compressor/src/builtins/dict/binary.rs:20,121` | Dictionary codes |
| `vortex-compressor/src/builtins/dict/float.rs:22,139` | Dictionary codes |
| `vortex-compressor/src/builtins/dict/integer.rs:22,131` | Dictionary codes |
| `vortex-compressor/src/builtins/dict/string.rs:20,121` | Dictionary codes |
| `vortex-compressor/src/compressor/structural.rs:19,54,87,99` | List offsets, ListView offsets and sizes |
| `vortex-btrblocks/tests/narrow.rs` | New. Only `test_internal_signed_buffers_roundtrip` from `0d94e44a91`, with the core edition and the module `SESSION` |
| `vortex-btrblocks/tests/snapshots/*.snap` | Regenerated |

The first number in each call-site row is the `PrimitiveArrayExt` import. Remove an import only
when it becomes unused: `vortex-btrblocks/src/schemes/integer/bitpacking.rs:16`,
`float/alp.rs:18` and `float/alprd.rs:16` import the trait for other methods.

`07fafc2a5b` pins dictionary compression in `test_array_stream_no_double_dict_encode`. `develop`
reworked that test with far-apart values (`vortex-file/src/tests.rs:1599-1623`). Leave
`07fafc2a5b` out unless that test fails with NR6.

Design notes:

- Each call site follows this shape. The reason names the open question, so a search finds every
  site later:

  ```rust
  #[expect(
      deprecated,
      reason = "TODO(connor): decide whether this child narrows its dtype or becomes a Narrow"
  )]
  let offsets = NarrowArray::encode_values(offsets, ctx)?;
  ```

- What "internal buffers keep their signedness" changes in file contents. The old
  `PrimitiveArrayExt::narrow` (`vortex-array/src/arrays/primitive/array/mod.rs:157-241`) picks
  `U8`, `U16` or `U32` whenever no value is negative (`:189`), whatever the input signedness. It
  also returns a zeroed `u8` buffer for an empty or all-null input (`:163-167`). `encode_values`
  keeps the signedness:
  - A signed child with only non-negative values gets a signed type. It is the same width when the
    max fits the signed type. For example, `u16` becomes `i16` for list offsets below 32,768. It is
    the next width when the max is between the signed and the unsigned limit. For example, values
    128 to 255 become `i16` instead of `u8`.
  - A 1-byte signed child stays `i8` instead of becoming `u8`.
  - An empty or all-null signed child becomes `i8` instead of `u8`.
  - Unsigned children, children with negative values, and `u64` values above `i64::MAX` do not
    change.
  - Signed sources in practice: list offsets, ListView offsets and sizes, VarBin offsets, FSST
    lengths and code offsets, OnPair children, and the DateTimeParts parts. Unsigned sources do not
    change: patch and sparse indices, RLE indices and offsets, dictionary codes and run ends.
  - The encoding IDs and metadata formats do not change. The dtype of these children changes in new
    files, and an uncompressed child can grow by one width.
- Readers on `develop` accept signed children. List, ListView, VarBin, FSST, DateTimeParts and
  OnPair require only `is_int()` for these children
  (`vortex-array/src/arrays/list/array.rs:211-215`, `listview/array.rs:239-246`,
  `varbin/array.rs:238-239`, `encodings/fsst/src/array.rs:842-848`,
  `encodings/datetime-parts/src/array.rs:318-328`, `encodings/onpair/src/array.rs:419-426`).
  ListView reinterprets signed offsets and sizes as unsigned before it uses them
  (`listview/array.rs:270-280`). Whether earlier releases behave the same is Not verified, so an
  acceptance check runs the latest release on the new files.
- The closed #10333 changed five snapshots this way. List offsets went from `u16` to `i16`. FSST
  lengths and code offsets went from `u8` to `i8` and from `u32` to `i32`. Their byte counts did
  not change. `develop` now writes `fastlanes.bitpacked.v2` and has more snapshot variants, so the
  snapshots must be regenerated, not copied.

Steps:

1. Create `ct/narrow-encode-values` from `develop` after NR1 merges.
2. Replace every call site in the table, each with the `#[expect(deprecated)]` shape above. Remove
   imports that become unused.
3. Remove `PrimitiveArrayExt::narrow` and port its tests.
4. Add `vortex-btrblocks/tests/narrow.rs` with the round-trip test only.
5. Regenerate the golden snapshots with
   `INSTA_UPDATE=always cargo test -p vortex-btrblocks --all-features --test golden`.
6. Commit as `feat(compress): keep integer signedness when narrowing internal children`, with the
   sign-off line. The PR body must say that `PrimitiveArrayExt::narrow` is removed from the public
   API.

Tests:

- Port each case of `vortex-array/src/arrays/primitive/array/cast.rs:58-212` to an `rstest` on
  `encode_values` in `narrow/tests/encoding.rs`. Expectations change where signedness changes:
  - `[0i64, 127]` gives `I8` (was `U8`).
  - `[Some(0i32), None, Some(127)]` gives nullable `I8` (was `U8`).
  - An empty `i32` or `i64` gives `I8` (was `U8`).
  - Unchanged: all-invalid `u32` gives `U8`, `[-1i32, 200]` gives `I16`, `[-100i16, 0, 100]` gives
    `I8`, `[0u64, u64::MAX]` stays `U64`, and floats stay unchanged.
  - Validity is kept in every case.
- `vortex-btrblocks/tests/narrow.rs::test_internal_signed_buffers_roundtrip`: lists and strings
  compress with the compact preset, serialize, decode and match the input, with no `vortex.narrow`
  in the array context.
- Regenerated `vortex-btrblocks/tests/snapshots/*.snap`. Only signed internal children change.

Benchmarks: Expect changes in the btrblocks `compress` and `compress_listview` benchmarks, because
child dtypes change. The closed #10333 reported `take_map` regressions of 22% and 11% in Simulation
mode. The plan to investigate them:

- `take_map` is in `vortex-array/benches/take_patches.rs:110-129`. It builds `Patches` with `u64`
  indices and `u64` values directly (`:131-140`), draws random `u64` take indices, and times
  `Patches::take_map` (`vortex-array/src/patches.rs:919-970`). No narrowing code runs.
- Step 1: compare the `take_map` cases on NR1, NR3 and NR4. These PRs add session registrations
  and kernels but no narrowing code. A move there points at registry growth or code layout.
- Step 2: compare `take_map` on NR6 with its base. NR6 changes none of the code that `take_map`
  runs, so a move there also points at code layout.
- Step 3: for a case that moves, open the CodSpeed flame graph diff. Find the symbols whose cost
  changed, for example `hashbrown` inserts and lookups, or allocator calls.
- Step 4: check the `take_map` history on `develop` for variance. `HashMap` is `hashbrown` with
  `DefaultHashBuilder` (`vortex-utils/src/aliases/hash_map.rs:5-9`). If that builder seeds per
  process, the hash layout differs between runs. Not verified.
- Report the result in the NR6 PR. Do not change `take_map` in NR6.

Acceptance checks:

- The commands below pass:

  ```bash
  cargo clippy -p vortex-array -p vortex-btrblocks -p vortex-compressor -p vortex-runend \
    -p vortex-fastlanes --all-targets --all-features -- -D warnings
  cargo +nightly-2026-09-10 fmt -p vortex-array -p vortex-btrblocks -p vortex-compressor \
    -p vortex-runend -p vortex-fastlanes
  cargo nextest run -p vortex-array narrow
  cargo nextest run -p vortex-btrblocks --all-features
  cargo nextest run -p vortex-compressor -p vortex-runend -p vortex-fastlanes
  ```

- `git grep -n "\.narrow(" -- '*.rs'` finds no call of `PrimitiveArrayExt::narrow`.
- Every call of `encode_values` carries `#[expect(deprecated)]` with the `TODO(connor)` reason.
- Snapshot diffs show only dtype labels of signed internal children, plus byte counts where a
  value range needs the next signed width.
- The latest release reads files that this PR writes:
  1. With this PR, run `cargo run -p vortex-compat --release -- generate --output <dir>`.
  2. In a separate worktree at the latest release tag, build `vortex-compat` and run its `check`
     on `<dir>`. Read the modes in `vortex-test/compat-gen/src/check.rs:21-30` first, and use the
     one that skips fixtures that only one side has. `compat.py --git-ref` does not build from the
     ref (`vortex-test/compat-gen/scripts/compat.py:922-925`), so build the binary by hand.

Review focus: Every byte-count change in the snapshot diff. Parents that receive a signed child
where they received an unsigned one before. The public API removal. Null payloads in the all-null
case, where the old code wrote zeros and the new code casts.

Size: Medium. About 20 call sites, ported tests and snapshots.

Open questions: None.

## Notes from planning

1. The claim that "the only integer-only check is the private `validate_dtypes`" holds only for
   explicit dtype checks. Opening it for decimals without other changes gives a panic in the
   compare kernel for two `Narrow` operands, wrong compare and between results for negative
   constants below the child range, and early overflow in decimal `Sum` and `SumV2`. The table
   "Integer-only sites" lists every site, and CD4 owns each fix.
2. `Narrow::execute` yields `values.cast(dtype)`. The decimal cast of D15 returns a `Narrow` for a
   widening cast, so a decimal `Narrow` would execute to itself. CD4 widens decimals explicitly.
   Resolved for integers in NR1 to NR4: `execute` casts the child, which is never a `Narrow` once
   flattened, and a nested `Narrow` that is not yet flattened still converges. The hazard stays
   for CD4 only.
3. The D15 text said that a cast of a `Narrow` passes to the child with the same target. For a
   wider integer target, the rewrite returns a new `Narrow` over the child. The values are equal
   and the data stays narrow. D15 now says this.
4. The rewrite flattens nested `Narrow` arrays only in `try_new`. Physical rewrites check only the
   slot dtype (`vortex-array/src/array/erased.rs:595-625`), so NR1 adds a `VTable::reduce` flatten
   rule.
5. The tests of the rewrite call `array_session()`, which `develop` removed. New tests use the
   module session, so port their cases, not their setup.
6. The report cites `narrow/mod.rs:81-84` for the width rule. That matches the tip `af4b663059`.
   In `f7408481ce` the same lines are `:76-79`.
7. `ct/narrow-encoding` (#10223), `ct/narrow-pushdown` (#10332) and `ct/narrow-compression`
   (#10333) still exist on `origin`. The new branch names avoid all three.
8. `Narrow` is registered in `ArraySession::default`, while the in-memory `Filter`, `Slice` and
   `Interleave` are not. `PiecewiseSequence` is registered and also has no wire form, so the
   rewrite follows a precedent.
9. Part 2 of #10333 wraps every integer primitive that reaches `compress_canonical` in a `Narrow`,
   including internal children. It also declares a preview edition with only `vortex.narrow`. WR2
   and WR3 must not reuse either as they are.
