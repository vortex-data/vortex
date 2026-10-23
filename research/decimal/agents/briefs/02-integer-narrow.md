<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Integer `Narrow` PR briefs: NR1 to NR6

These briefs cover the integer `Narrow` track. Each PR is written from scratch from its brief
(D38). A local rewrite of the closed PRs #10223 and #10332 exists in
`/Users/connor/spiral/vortex-data/vortex3`. It is a reference, not a base: read its code and tests
for edge cases, reuse its test cases, and do not cherry-pick it. The rewrite predates D25, D26, D28
and D36, so its structure differs from these briefs. The one exception is FX3, which ports the
dictionary fill fix of the rewrite as it is.

No build, test or benchmark ran for these briefs. Every claim about code comes from reading it,
and it is marked "Not verified" where it matters.

## Shared context for the track

### The reference rewrite

| Commit | Subject | Use it for | Do not copy |
| --- | --- | --- | --- |
| `f7408481ce` | feat(array): add in-memory Narrow integer encoding | NR1: the width selection of `encode`, `validate_dtypes`, the compare constant shortcut, the selection rules, and the cases of `narrow/tests/{encoding,compute}.rs` | The integer-only parts of the compare kernel (see below). The serialization stub |
| `8d548646ee` | feat(array): forward Narrow integer aggregates | NR3: the list of supported aggregates, and the cases of `narrow/tests/aggregates.rs` and `encodings/fastlanes/tests/narrow.rs` | The accumulator for the child dtype. NR3 uses the accumulator of the parent dtype (D36) |
| `74aa17644e` | feat(array): push arithmetic and scalar operations through Narrow | NR4: the interval arithmetic, the cast, fill-null, zip and between rules, and the cases of `narrow/tests/{numeric,pushdown}.rs` | The replacement of the NR1 cast and fill-null rules. NR1 writes those rules so that NR4 extends them |
| `af4b663059` | feat(array): retain Narrow storage through merges and indices | NR5: `TakeIndicesReduce` and `test_narrowed_take_indices` | The chunked and interleave merge rules in `narrow/merge.rs`, which put `Narrow` above `Chunked` and `Interleave` (D25) |
| `7e2e5e4d7a` | fix(array): avoid truncating dictionary fill codes | FX3 ports the fix as it is. NR5 reuses `test_fill_null_after_narrowing_take_indices` | Nothing |
| `adc1e83b9d` | feat(compress): retain narrow integer children through compression | NR1: `encode_values`. NR6: the list of call sites. WR1 to WR3: background | The `Narrow` wrappers for internal children, and the preview edition with only `vortex.narrow` |

All these commits are reachable from every local worktree, because the worktrees share one bare
repository. The five rewrite commits are not on `origin`. Only the local branches
`ct/narrow-pushdown` (checked out in `vortex3`) and `ct/narrow-encoding` hold them, and both names
also exist on `origin` with the older commits of the closed PRs. Do not push, reset or delete
these two local branches. A remote agent cannot read the reference until it is pushed to a new
backup branch. `adc1e83b9d`, `0d94e44a91` and `07fafc2a5b` are on `origin/ct/narrow-compression`.

### Changes on `develop` since the rewrite

The rewrite is based on `61ff28d970`. These changes on `develop` matter when an agent reads it:

| Change on `develop` | Commit | What new code does |
| --- | --- | --- |
| `vortex_array::array_session()` removed | `d834a11faa` (#10681) | Every test module uses a module `static SESSION`, with the pattern below. The tests of the rewrite call `array_session()`, so port their cases, not their setup |
| Interleave uses named slots | `96ac9f2cc5` (#10688) | `InterleaveArray::try_new(values, array_indices, row_indices)` keeps its signature. The slot order is the same: selectors at 0 and 1, values from 2. `InterleaveSlots` is `pub(super)` (`vortex-array/src/arrays/interleave/array.rs:24`). The NR5 pull-up rule builds a new array with `InterleaveArray::try_new` |
| Storage pushdown in the aggregate session | `d7562aa0e6` (#10566), `3bf14ba647` (#10616) | `AggregateFnSession::default` has `register_storage_ordered` calls (`vortex-array/src/aggregate_fn/session.rs:127-129`). Register the NR3 kernel after the `Dict` kernels. `register_aggregate_kernel` and the fallback lookup are unchanged (`:199-224`) |
| `fill_null` on an empty input | `633c52deab` (#10627) | `FillNull` returns an empty array for a non-constant fill on an empty input (`vortex-array/src/scalar_fn/fns/fill_null/mod.rs:145`). The `Narrow` rule receives a `Scalar`, so it is not affected |
| Kernel for constant dictionary codes | `a169338930` (#10643) | An execute-parent kernel handles `Constant` codes (`vortex-array/src/arrays/dict/take.rs:181-205`). The NR5 rule for `Narrow` codes is a reduce rule, so the two do not overlap |
| Deprecated APIs and `legacy_session` removed | `85ae34d5f0` (#10595) | None of the removed APIs is needed |
| Execution contexts in validation and patches | `c48457b572`, `7216cb5cb8`, `673fd2e064`, `b549eb5a97` | `ListViewArray::try_new`, `Patches::new`, `MaskedArray::try_new`, `RunEndArray::try_new` and `SparseArray::try_new` take a context. Reduce rules have none, so the NR5 brief names constructors without a context |

`VTable`, `OperationsVTable`, `ValidityChild`, the kernel registry, `CompareKernel`,
`execute_compare`, `execute_numeric`, the reduce adaptors, `ChunkedSlots::CHUNKS_OFFSET`,
`DictSlots` and `BitBuffer::full_in` keep the signatures that the rewrite uses at `b70206818f`.
The execute-parent registry keeps a list for each key
(`vortex-array/src/optimizer/kernels.rs:263-276`). So the compare kernel of NR1 and the numeric
kernel of NR4 can both register for `Binary` on `Narrow`.

#### Test session pattern

PR #10681 gave each test module one session. `VortexSession::empty()` creates the vortex-array
session variables on first use, and the default `KernelSession` runs `arrays::initialize`
(`vortex-array/src/optimizer/kernels.rs:336-349`), so the `Narrow` kernels are present.

```rust
// Module level, as in vortex-array/src/arrays/dict/compute/fill_null.rs:109 on develop.
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
| NR1 | `ct/narrow-in-memory` | `develop` |
| NR2 | `ct/narrow-benchmarks` | NR1 |
| NR3 | `ct/narrow-aggregate-kernels` | NR2 |
| NR4 | `ct/narrow-arithmetic-kernels` | NR3 |
| NR5 | `ct/narrow-structural-rules` | NR4, after FX3 is on `develop` |
| NR6 | `ct/narrow-encode-values` | `develop` after NR1 merges. Merge it after NR5, so that the stack lands first (D35) |

NR1 to NR5 form one linear stack (D35). NR3 needs no code from NR2, but it sits on NR2 so that the
benchmarks exist before the kernels change. NR6 needs only NR1. CodSpeed baselines come from pushes
to `develop` (`.github/workflows/codspeed.yml:13-17`). How CodSpeed picks a base for a PR whose base
branch is another PR branch is not verified. The safe order is to merge NR1 and NR2, then rebase NR3
to NR5 on `develop` before their CodSpeed reports are read.

`develop` squash-merges PRs. After a PR of the stack merges, a plain `git rebase develop` on the
next branch replays the commits of the merged PR again and conflicts. Rebase with
`git rebase --onto develop <old tip of the merged branch>`, or with `gh stack`.

### Write the kernels so that decimal support is additive

CD4 opens `Narrow` for decimals. The rewrite has integer-only assumptions that break then (table
below). The new NR code avoids them from the start where that costs little:

- No `as_ptype()` and no `DType::Primitive` on a path that a decimal child can reach once
  `validate_dtypes` accepts decimals. For example, the compare of two `Narrow` operands picks the
  wider child dtype without `as_ptype()`.
- The out-of-range compare decides the direction by comparing the constant with the bounds of the
  child dtype, not through `i64::try_from` and `is_signed_int()`.
- Aggregates use the accumulator of the parent dtype (D36), so a decimal sum needs no special case.
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
- Decisions: D8 (one encoding with ID `vortex.narrow`), D15, D18, D22, D23, D26, D28, D38.
- Branch: `ct/narrow-in-memory`, based on `develop` (`b70206818f` or later).
- Scope:
  - Write the encoding, `validate_dtypes`, the compare kernel, and the selection, cast and
    fill-null rules. `f7408481ce` is the reference (D38).
  - Add `NarrowArray::encode_values`, marked `#[deprecated]` (D28). `adc1e83b9d` has a reference
    version.
  - Add the serialization workaround of D26: a plugin hook that lowers an array before it is
    serialized, which `Narrow` implements by executing itself to its canonical array.
  - Add a reduce rule that flattens a `Narrow` child, so D15 holds after physical rewrites.
- Out of scope: Aggregate kernels (NR3). Arithmetic, between, zip, float and cross-signedness casts
  (NR4). The rules that move structural encodings above `Narrow`, and the dictionary-codes rule
  (NR5). The switch of the existing narrowing call sites to `encode_values` (NR6). The
  `vortex.narrow` wire form and editions (WR1, WR2). Compressor changes (WR3). Decimals (CD4). The
  dictionary fill-code fix (FX3).

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/narrow/mod.rs` | New. Reference: `f7408481ce`. `Narrow`, `NarrowArray`, `NarrowSlots`, `try_new`, `validate_dtypes`, `initialize` |
| `vortex-array/src/arrays/narrow/vtable.rs` | New. Reference: `f7408481ce`. With `VTable::reduce` that flattens a `Narrow` child, and the lowering hook |
| `vortex-array/src/arrays/narrow/encoding.rs` | New. `NarrowArray::encode` and `encode_values`. References: `f7408481ce` and `adc1e83b9d:vortex-array/src/arrays/narrow/encoding.rs:45-90` |
| `vortex-array/src/arrays/narrow/compare.rs` | New. Reference: `f7408481ce`. Compare kernel |
| `vortex-array/src/arrays/narrow/rules.rs` | New. Reference: `f7408481ce`. Slice, filter, take, mask, cast and fill-null rules |
| `vortex-array/src/arrays/narrow/tests/mod.rs` | New |
| `vortex-array/src/arrays/narrow/tests/encoding.rs` | New. Port the cases of the reference tests, and add the flatten, `encode_values` and serialization tests |
| `vortex-array/src/arrays/narrow/tests/compute.rs` | New. Port the cases of the reference tests |
| `vortex-array/src/array/plugin.rs:113-158` | Add the lowering hook to `ArrayPlugin`, with a default of `Ok(None)`. The blanket impl for `VTable` (`:165-205`) forwards it |
| `vortex-array/src/serde.rs:169-190` | `ArraySerializationTree::try_new` asks the hook when `array_serialize` returns `None` |
| `vortex-array/src/arrays/mod.rs:85-91` | Add `pub mod narrow;` with `pub use narrow::Narrow;` and `pub use narrow::NarrowArray;` between `masked` and `null` |
| `vortex-array/src/arrays/mod.rs:144-145` | Add `narrow::initialize(session);` between `map` and `null` |
| `vortex-array/src/session/mod.rs:31-32` | Import `crate::arrays::Narrow` |
| `vortex-array/src/session/mod.rs:107-108` | `this.register(Narrow);` after `Masked` |

Design notes:

```rust
pub struct Narrow;
pub type NarrowArray = Array<Narrow>;
#[array_slots(Narrow)]
pub struct NarrowSlots { pub values: ArrayRef }

impl NarrowArray {
    pub fn try_new(values: ArrayRef, dtype: DType) -> VortexResult<Self>;

    /// Stores `array` as a `Narrow` over a child of the smallest integer type with the same
    /// signedness, or returns it unchanged when no narrower type holds its values.
    pub fn encode(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef>;

    /// Returns `array` at the smallest integer type with the same signedness, without a `Narrow`.
    /// Floats return unchanged.
    #[deprecated(note = "changes the dtype; a later change decides whether each caller needs a Narrow")]
    pub fn encode_values(array: PrimitiveArray, ctx: &mut ExecutionCtx)
        -> VortexResult<PrimitiveArray>;
}
pub(super) fn validate_dtypes(storage: &DType, logical: &DType) -> VortexResult<()>;
```

- `validate_dtypes` implements D22: both dtypes are integer primitives, the signedness is equal,
  the child byte width is strictly smaller, and the nullability is equal. `VTable::validate` and
  `try_new` both call it.
- `encode` and `encode_values` share one private, non-deprecated function, so `encode` itself needs
  no `#[expect(deprecated)]`. That function takes `min_max` of the valid values, picks the smallest
  type of the same signedness for the min and for the max, and takes the wider of the two. An
  empty or all-null input gets the 1-byte type of the same signedness. A 1-byte input returns
  unchanged. `encode` fails on a float input, and `encode_values` returns a float unchanged. An
  `I64` column with values 0 to 200 gets an `I16` child (D22).
- `encode_values` has no caller in this PR. CD7 (the DecimalByteParts MSP) and NR6 (the existing
  call sites) call it, each with `#[expect(deprecated, reason = "...")]` (D28).
- `try_new` flattens a `Narrow` child. Physical rewrites keep only the slot dtype
  (`vortex-array/src/array/erased.rs:595-620`), so a rewrite can replace a child with a `Narrow` of
  the same dtype. The new `reduce` returns `NarrowArray::try_new(child.values().clone(), dtype)`
  when the child is a `Narrow`, and `None` otherwise.
- `execute` yields `values.cast(dtype)`, so the child can fuse decoding and widening. `scalar_at`
  casts the child scalar. Validity comes from the child. CD4 changes `execute` for decimals,
  because the decimal cast of D15 would loop.
- The serialization workaround (D26):
  - Today `ArraySerializationTree::try_new` fails with "Array … does not support serialization"
    when `array_serialize` returns `None` (`vortex-array/src/serde.rs:171-175`).
  - A plugin cannot return the wire form of another encoding. `array_serialize` checks that the
    returned ID is one of the plugin's `serialized_ids` (`vortex-array/src/session/mod.rs:144-151`).
  - So the fallback lives in the serializer. A new `ArrayPlugin` method, for example
    `lower_for_serialization(&self, array, session) -> VortexResult<Option<ArrayRef>>`, returns
    `Ok(None)` by default. When `array_serialize` returns `None`, `try_new` calls it, and it builds
    the tree from the returned array. Without a lowered array, it fails as today.
  - `Narrow` lowers itself by executing to its canonical array with
    `session.create_execution_ctx()`. No other encoding implements the hook.
  - The doc of the hook and a comment in the `Narrow` impl say that this is a workaround. It stays
    until `vortex.narrow` is in `core` (LT1). WR1 also uses it when the selected editions do not
    permit `vortex.narrow`.
- `deserialize` fails with "Narrow deserialization is not supported", because no wire form exists
  yet. `ArraySession::default` registers `Narrow` like `PiecewiseSequence`
  (`vortex-array/src/session/mod.rs:108`). No compressor produces a `Narrow` in this PR.
- Compare kernel: two `Narrow` operands compare at the wider child width. A constant that casts to
  the child dtype compares against the child. A constant outside the child range gives a constant
  answer with the child validity. `Eq` is false and `NotEq` is true. `Lt`, `Lte`, `Gt` and `Gte`
  follow from whether the constant is below the range. A null constant gives all nulls. Follow
  "Write the kernels so that decimal support is additive" above.
- Cast rule in this PR: take an integer target of the same signedness. If it is no wider than the
  child, the cast goes to the child with the same target. If it is wider, the rule returns a
  `Narrow` over the child cast to the target nullability (D15 as amended). Other targets fall
  back. Write the cast and fill-null rules so that NR4 can add target kinds without replacing them.
- Fill null declines when the fill value does not fit the child dtype, so canonical execution
  handles it.

Steps:

1. Create `ct/narrow-in-memory` from `develop` in a new worktree. Do not reuse the `vortex3`
   worktree or its branch.
2. Read `f7408481ce` and its tests with `git show f7408481ce`.
3. Write the module: the encoding, `validate_dtypes`, the vtable, the compare kernel and the rules.
   Register it in `arrays/mod.rs` and `session/mod.rs`.
4. Add `encode_values` and the shared private function.
5. Add the lowering hook in `plugin.rs` and `serde.rs`, and the `Narrow` impl.
6. Add `VTable::reduce` and the flatten rule.
7. Add the tests, with the module `SESSION`. Port the cases of the reference tests.
8. Commit in reviewable steps. Each commit has the line
   `Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>` and no co-author line.

Tests:

- Port the cases of these tests in `f7408481ce:vortex-array/src/arrays/narrow/tests/encoding.rs`:
  `test_signed_encoding` and
  `test_unsigned_encoding` (`rstest` width boundaries), `test_null_payloads_and_empty`,
  `test_reject_invalid_dtypes`, `test_validate_slots`, `test_flatten_and_scalar_dtype`,
  `test_encoded_child_execution`, `test_smallest_width_unchanged`, `test_reject_float_encoding`,
  `test_deserialization_unsupported`.
- `test_serialization_unsupported` becomes `test_serialization_writes_canonical`. Serialize a
  nullable and a non-nullable `Narrow`, and decode the bytes. The result is a canonical primitive
  array with the logical dtype and equal values, and its wire ID is the primitive ID.
- New, in `narrow/tests/encoding.rs`:
  - `test_encode_values_keeps_signedness`, an `rstest`. `[0i64, 127]` gives `I8`, `[0i64, 200]`
    gives `I16`, `[-1i32, 200]` gives `I16`, an empty `i32` gives `I8`, all-invalid `u32` gives
    `U8`, and `[0u64, u64::MAX]` stays `U64`. Validity is kept in every case.
  - `test_encode_values_preserves_floats`, from `adc1e83b9d`.
  - `test_reduce_flattens_nested_narrow`. Build a `Narrow` with `try_from_parts` over a `Narrow`
    child, and call `optimize()`. Check that the result is one `Narrow` over the inner child, with
    the same values.
- In `vortex-array/src/serde.rs` tests: an encoding without a wire form and without the hook still
  fails with "does not support serialization".
- Port the cases of these tests in `f7408481ce:vortex-array/src/arrays/narrow/tests/compute.rs`:
  `test_selection_and_fill_null`,
  `test_cast_preserves_checked_nullability`, `test_fill_null_preserves_result_dtype`,
  `test_arithmetic_uses_logical_width`, and `test_comparison`. The last one is an `rstest` over six
  operators. It covers constants on both sides of the child range, both operand orders, and a
  second `Narrow`.

Benchmarks: None in this PR. NR2 adds them. No change on existing benchmarks is expected. Record
the `take_patches` results for the NR6 investigation.

Acceptance checks:

- `cargo clippy -p vortex-array --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-array`
- `cargo nextest run -p vortex-array narrow` and `cargo nextest run -p vortex-array serde`
- Serializing a `Narrow` writes its canonical array, and the read-back array has equal values.
- No file writer or compressor produces `vortex.narrow`.
- `encode_values` carries `#[deprecated]` and has no caller yet.

Review focus: The constant shortcut of the compare kernel for both operand orders and null
constants. Checked nullability in the cast rule. Nullability of `take` and `mask` results, which
follow the child. The fill-null fallback. The new `reduce` rule. The lowering hook: it runs only
when an array has no wire form, and only `Narrow` implements it. Registration of the compare kernel
through `arrays::initialize`.

Size: Large. The reference commit has about 550 lines of source and 435 lines of tests.
`encode_values`, the hook and their tests come on top.

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

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-array/benches/narrow.rs` | New |
| `vortex-array/Cargo.toml:335-336` | Add `[[bench]]` with `name = "narrow"` and `harness = false` after the `probe` entry |

Design notes:

- Follow the conventions of `vortex-array/benches/compare.rs:22-76` and
  `vortex-array/benches/filter_fixed_width.rs:101-126`:
  - An SPDX header and a module doc.
  - The `MiMalloc` global allocator.
  - A `main` that forces `SESSION` and then calls `divan::main()`.
  - A `SESSION` with `ArraySession`, `KernelSession`, `DTypeSession`, `ScalarFnSession`,
    `AggregateFnSession`, `MemorySession` and `StatsSession`.
  - Each result wrapped in `divan::black_box`.
- Do not tag any case with `#[vortex_bench_support::cpu_features]`. These kernels are dispatch
  work, so the cases stay in simulation.
- The name `narrow` matches no prefix in `.github/workflows/codspeed.yml:92-104`, so the cases run
  in the "builders" shard. No workflow change is needed.
- Data: `LEN = 8_192` rows, so the wide input holds 64 KiB, as `filter_fixed_width.rs:57-64`
  recommends. Values are `i16` from `StdRng::seed_from_u64`, uniform in `-10_000..=10_000`. The
  `Narrow` input has logical dtype non-nullable `I64` over that `i16` child. The wide input holds
  the same values as `i64`.
- Build every input inside `with_inputs`, so setup is not timed. Build fresh arrays for each
  iteration. `sum` caches its result on the array
  (`vortex-array/src/aggregate_fn/fns/sum/mod.rs:63-65`), and the NR4 arithmetic kernel calls
  `min_max` on both operands, so reused inputs hide work.

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

#[divan::bench(args = INPUTS)]
fn compare_in_range(bencher: Bencher, input: Input) { /* ... */ }
```

Workloads. Each one ends with `execute::<Canonical>`, or returns the aggregate scalar:

| Bench | Operation | PR whose kernel changes the `Narrow` case |
| --- | --- | --- |
| `compare_in_range` | `binary(Constant(0i64), Lt)` | NR1 (child compare) |
| `compare_out_of_range` | `binary(Constant(100_000i64), Lt)`, above `i16::MAX` | NR1 (constant answer) |
| `between_in_range` | `between(Constant(-1_000i64), Constant(1_000i64))`, both bounds non-strict | NR4 |
| `filter` | `filter(mask)`, random mask with density 0.5 | NR1 |
| `take` | `take(indices)`, `LEN / 4` random `u32` indices | NR1 |
| `slice` | `slice(LEN / 4..3 * LEN / 4)` | NR1 |
| `sum` | `sum(&array, ctx)` | NR3 |
| `min_max` | `min_max(&array, ctx, NumericalAggregateOpts::default())` | NR3 |
| `add_constant` | `binary(Constant(1i64), Add)` | NR4 |
| `add_narrow` | `Narrow + Narrow`, second operand from another seed | NR4 |
| `add_wide` | `Narrow + wide i64 array`, only `Narrow` and `Decoded` inputs | NR4 |
| `chunked` | `ChunkedArray` of four inputs of `LEN / 4` rows | None. It guards the chunk kernels, which push work into each `Narrow` chunk |
| `interleave` | `InterleaveArray` of two inputs of `LEN / 2` rows, `array_indices = i % 2` as `u8`, `row_indices = i / 2` as `u32` | None. It guards the interleave kernels |

- `Decoded` for `chunked` and `interleave` decodes each input, then builds the parent.
- `Wide` for `add_narrow` adds two `i64` arrays. `add_wide` omits `Wide`, because it is the same
  case.

Steps:

1. Create `ct/narrow-benchmarks` on top of NR1.
2. Add `vortex-array/benches/narrow.rs` and the `[[bench]]` entry.
3. Commit as `bench(array): add Narrow compute benchmarks`, with the sign-off line.

Tests: None. The benchmark compiles under `cargo clippy --all-targets`.

Benchmarks: This PR only adds cases, so CodSpeed reports them as new. Qualitative expectations for
later PRs:

- NR3: `sum[Narrow]` and `min_max[Narrow]` move from about the `Decoded` level toward or below
  `Wide`.
- NR4: `add_constant`, `add_narrow`, `add_wide` and `between_in_range` change for `Narrow`. The
  direction for `add_wide` is not known, because the NR4 kernel scans the wide operand for bounds.
- `chunked` and `interleave` stay flat. NR5 changes no tree that they build.
- `Wide` and `Decoded` cases must stay flat in NR3 to NR5. A move there points at noise or at a
  side effect of session registration.

Acceptance checks:

- `cargo clippy -p vortex-array --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-array`
- `cargo bench -p vortex-array --bench narrow` runs every case.
- Each case stays under 1 ms in CodSpeed simulation.

Review focus: The three inputs hold the same logical values. Inputs are built outside the timed
closure and fresh for each iteration. `Decoded` decodes inside the timed closure.

Size: Medium. One file with about 13 workloads.

Open questions: None.

### NR3: Aggregates forward to the child

- Goal: Register one fallback aggregate kernel for `Narrow`. It feeds the child to an accumulator
  of the parent dtype (D36), so every result and every overflow is the same as on the parent.
- Depends on: NR1, NR2 (stack, D35: the benchmarks exist before the kernels change). Soft: None.
  Blocks: NR4 (stack), CD4.
- Decisions: D15, D36, D38.
- Branch: `ct/narrow-aggregate-kernels`, based on NR2.
- Scope: Write the kernel. `8d548646ee` is the reference for the list of aggregates and for test
  cases.
- Out of scope: Decimal aggregates (CD4). Pushdown of sum, min and max into the MSP (LT3).

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/narrow/aggregates.rs` | New. Reference: `8d548646ee` |
| `vortex-array/src/arrays/narrow/mod.rs` | `mod aggregates;` and `pub(crate) use aggregates::register_aggregate_kernels;`, module doc names aggregates |
| `vortex-array/src/arrays/narrow/tests/mod.rs` | `mod aggregates;` |
| `vortex-array/src/arrays/narrow/tests/aggregates.rs` | New. Port the cases of the reference tests |
| `vortex-array/src/aggregate_fn/session.rs:51-52` | `use crate::arrays::narrow;` |
| `vortex-array/src/aggregate_fn/session.rs:135-136` | `narrow::register_aggregate_kernels(&this);` after the `Dict` kernels |
| `encodings/fastlanes/tests/narrow.rs` | New integration test with a `BitPacked` child, with the closure session pattern |

Design notes:

```rust
pub(crate) fn register_aggregate_kernels(session: &AggregateFnSession);
struct NarrowAggregateKernel; // impl DynAggregateKernel
```

- The kernel registers with aggregate id `None`, so it is the fallback for `Narrow`. A kernel for an
  exact aggregate wins over it (`vortex-array/src/aggregate_fn/session.rs:199-224`).
- Supported: `MinMax`, `Min`, `Max`, `Sum`, `SumV2`, `IsConstant`, `IsSorted`, `First`, `Last`,
  `Count`, `NullCount`, `AllNull`, `AllNonNull`. Every other aggregate returns `None` and uses the
  default path. `UncompressedSizeInBytes` must stay on the default path, because it depends on the
  logical width.
- Algorithm (D36): build the accumulator for the parent dtype, and feed it the child batch. First
  confirm that each supported accumulator accepts a batch of the narrower child dtype. Not
  verified. An aggregate whose accumulator rejects it leaves the supported list and uses the
  default path.
- Integer correctness: exact widening keeps values and order. The integer `Sum` state is `I64` for
  every signed input and `U64` for every unsigned input
  (`vortex-array/src/aggregate_fn/fns/sum/mod.rs:114-122`). With the parent accumulator, overflow
  matches the parent by construction.
- A decimal sum state depends on the input precision (`sum/mod.rs:81-86`). Because the
  accumulator comes from the parent dtype, a decimal sum is correct once CD4 opens
  `validate_dtypes`. CD4 only adds the decimal tests.

Steps:

1. Create `ct/narrow-aggregate-kernels` on the NR2 branch.
2. Write `narrow/aggregates.rs` and its registration in `aggregate_fn/session.rs`.
3. Port the test cases of `8d548646ee`, including the fastlanes integration test, with the module
   sessions. That integration test needs no `#![cfg(test)]`.
4. Commit with the sign-off line.

Tests:

- `test_aggregate_partial_states` (`rstest`): six batch shapes (signed, unsigned, descending
  boundary, equal boundary, empty, all null) times 14 bound aggregates. It checks the kernel
  partial, the running partial and the final value against the canonical input.
- `test_unsupported_aggregate_keeps_logical_width`: `UncompressedSizeInBytes` returns `None` from
  the kernel and reports the logical width.
- `encodings/fastlanes/tests/narrow.rs::test_narrow_bitpacked_child`: filter, take, compare, sum and
  execution on a `BitPacked` child.

Benchmarks: NR2 `sum` and `min_max` with the `Narrow` input are expected to improve. All other
cases stay flat.

Acceptance checks:

- `cargo clippy -p vortex-array -p vortex-fastlanes --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-array -p vortex-fastlanes`
- `cargo nextest run -p vortex-array narrow`
- `cargo nextest run -p vortex-fastlanes --test narrow`

Review focus: The supported list. Each listed aggregate must give the same result on the child as
on the widened values. The cast of partial states that hold extrema or boundary values. The
fallback registration does not shadow kernels for exact aggregates.

Size: Small. The reference has about 100 lines of source and 235 lines of tests.

Open questions: None.

### NR4: Arithmetic and scalar operations forward to the child

- Goal: Execute checked add, subtract, multiply and divide at the smallest width that holds both
  operands and the result interval. Push `between`, `zip`, `fill_null` and more casts through
  `Narrow`.
- Depends on: NR3 (stack). Soft: NR2. Blocks: NR5.
- Decisions: D15, D22, D38.
- Branch: `ct/narrow-arithmetic-kernels`, based on NR3.
- Scope: Write the numeric kernel and the rules. `74aa17644e` is the reference for the interval
  arithmetic and for test cases. Extend the cast and fill-null rules of NR1 instead of replacing
  them.
- Out of scope: Structural rules and the dictionary-codes rule (NR5). Decimal arithmetic (D20).
  Decimal casts (CD4).

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/narrow/numeric.rs` | New. Reference: `74aa17644e`. Execute-parent kernel for `Binary` |
| `vortex-array/src/arrays/narrow/rules.rs` | Extend the cast and fill-null rules. Add `ZipReduce`, `ZipFalseReduce` and `BetweenReduce` |
| `vortex-array/src/arrays/narrow/encoding.rs` | `integer_types` helper, used by `encode` and `numeric.rs` |
| `vortex-array/src/arrays/narrow/mod.rs` | `mod numeric;`, `initialize` registers the numeric kernel |
| `vortex-array/src/arrays/narrow/tests/mod.rs` | `mod numeric;` and `mod pushdown;` |
| `vortex-array/src/arrays/narrow/tests/numeric.rs` | New. Port the cases of the reference tests |
| `vortex-array/src/arrays/narrow/tests/pushdown.rs` | New. Port the cases of the reference tests |

Design notes:

- Numeric kernel: for `Add`, `Sub`, `Mul` and `Div`, unwrap `Narrow` operands to their children.
  Take `min_max` bounds of both operands as `i128`. Compute the result interval. Pick the smallest
  type of the same signedness and no wider than the logical type that holds both operands and the
  interval. Cast both operands to it, run the checked `execute_numeric`, and wrap the result with
  `wrap_values`. When no narrower type holds the interval, use the logical type, so the checked
  kernel reports a real overflow exactly as canonical execution does. An empty parent gives an
  empty canonical array. An all-null operand gives a null constant.
- Cast rule (D15): a float target goes to the child with the same target. An integer target
  first picks a storage type. It is the child type with the signedness of the target. For an
  unsigned child and a signed target, it is the next wider signed type. A target no wider than the
  storage type goes to the child with the same target. A wider target returns a `Narrow` over the
  child cast to the storage type. Non-primitive targets fall back.
- Fill null: `integer_storage_type` picks the child type, or the smallest type that holds the fill
  value, without reading values. The child is cast to it before the fill. A fill value that needs
  the logical width gives a plain array.
- Zip: both branches. The other branch is a `Narrow` or a constant. Otherwise fall back.
- Between: when both bounds are constants that fit the child, run `between` on the child.
  Otherwise combine two compares with `And`, so each compare uses the NR1 kernel.

Steps:

1. Create `ct/narrow-arithmetic-kernels` on the NR3 branch.
2. Write the numeric kernel, extend the cast and fill-null rules, and add the zip and between
   rules.
3. Port the test cases of `74aa17644e`, with the module `SESSION`.
4. Commit in reviewable steps, with the sign-off line.

Tests:

- `narrow/tests/numeric.rs`:
  - `test_signed_arithmetic`: `rstest` over width growth, skipped widths, full width, division at
    `i8::MIN` and across zero, and an overestimated interval.
  - `test_signed_domain`: every `i8` value against small divisors, for all four operators.
  - `test_logical_errors`: overflow and division by zero match the canonical errors.
  - `test_unsigned_arithmetic`, `test_unsigned_underflow`, `test_nulls_and_empty`,
    `test_encoded_child_and_operand_order`, `test_overflow_at_each_logical_width`,
    `test_logical_overflow_behind_null`.
- `narrow/tests/pushdown.rs`:
  - `test_between_bounds`: `rstest` over bounds inside and outside the child range, and both
    strictness values.
  - `test_cast_child`: unsigned to signed, signed to unsigned, and float.
  - `test_fill_width`, `test_zip_width`, `test_between_nullable_bounds`,
    `test_zip_false_branch_and_null_mask`, `test_signedness_cast_checks_values_and_nulls`.

Benchmarks: NR2 `add_constant`, `add_narrow`, `add_wide` and `between_in_range` change for the
`Narrow` input. If `add_wide[Narrow]` is slower than `add_wide[Decoded]`, report it. A possible
fix is to decline operands that are neither `Narrow` nor constant.

Acceptance checks:

- `cargo clippy -p vortex-array --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-array`
- `cargo nextest run -p vortex-array narrow`
- Every arithmetic result equals the canonical result, including errors and nulls.

Review focus: Interval arithmetic for division across zero and for `i64::MIN / -1`. Overflow
checks stay with the checked kernel at the logical width. The cast rule for unsigned children and
signed targets. The `min_max` pass on a wide operand. Null payloads outside the chosen width.

Size: Large. The reference has about 450 lines of source and 425 lines of tests.

Open questions: None.

### NR5: Structural encodings move above `Narrow`, and dictionary codes use the child

- Goal: Keep generic structural encodings above `Narrow` (D25). Reduce rules turn a `Narrow` over
  `Dict`, `RunEnd`, `Sparse`, `Constant`, `Masked`, `Chunked` or `Interleave` into that encoding
  over `Narrow` children. Dictionary codes that are a `Narrow` use its child directly.
- Depends on: NR4 and FX3. Soft: NR2. Blocks: WR3.
- Decisions: D25, D15, D23, D38.
- Branch: `ct/narrow-structural-rules`, based on NR4, after FX3 is on `develop`.
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
- Dictionary codes: a `Narrow` codes child is replaced by its child, with the same `DictData`.
  Narrowed codes can make an unreferenced fill index exceed the codes range. FX3 makes the
  dictionary `fill_null` kernel decline in that case, so FX3 must be on `develop` first.
- The rules never read the child dtype, so they also hold for decimal `Narrow`. CD4 adds the
  decimal tests.
- WR3 can wrap a column root in `Narrow`, because these rules then move a structural encoding back
  to the top. The dict-layout probe and the Arrow and DuckDB exporters then see `Dict` and `RunEnd`
  at the root, as today.

Steps:

1. Confirm that FX3 is on `develop`. Rebase NR1 to NR4 if needed.
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
   Any later integer cast rule that returns a `Narrow` has the same hazard.
3. The D15 text said that a cast of a `Narrow` passes to the child with the same target. For a
   wider integer target, the rewrite returns a new `Narrow` over the child. The values are equal
   and the data stays narrow. D15 now says this.
4. The rewrite flattens nested `Narrow` arrays only in `try_new`. Physical rewrites check only the
   slot dtype (`vortex-array/src/array/erased.rs:595-620`), so NR1 adds a `VTable::reduce` flatten
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
