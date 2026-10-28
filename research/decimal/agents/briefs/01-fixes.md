<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Independent fixes: FX1, FX2 and FX3

These three PRs depend on no other PR. FX1 blocks CD6, and FX3 blocks WD5. Each one starts from
`develop` at `b70206818f`. Every `file:line` below points at that checkout, unless a commit is
named.

FX2 is dropped (see its section). The PR lists in #10701 and #10733 track the state of FX1 and
FX3.

Rules for all three PRs:

- Each commit ends with `Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>`. Add no AI
  co-author line.
- Agents can run any command that the work needs, including the acceptance checks. The checks are
  scoped to the affected crates. The pinned nightly is `nightly-2026-09-10`
  (`.github/workflows/ci.yml:25`).
- Tests return `VortexResult<()>` where they can and use named `rstest` cases. Tests that need a
  session share the module `static SESSION: LazyLock<VortexSession>` that PR #10681 introduced.
- Every "before the fix" result in these briefs comes from reading the code, not from a run.

### FX1: Strict decoding of stored `DecimalType` and `PType` values

- Goal: A reader returns an error for an unknown stored `DecimalType` or `PType` value. Today the
  prost getters decode such a value silently as `I8` or `U8`. CD6 detects old decimal nodes by the
  stored width, so that field must be strict first.
- Depends on: None. Soft: None. Blocks: CD6.
- Decisions: D34 (one PR for all three commits). FX1 is the prerequisite of the D7 detection in
  CD6. It follows the read rule of D14: a file read never panics. It is the first half of node N10
  in the roadmap.
- Branch: `ct/strict-enum-decoding`, based on `develop`.
- Scope: Commit 1 makes the decimal `values_type` strict. Commit 2 makes the protobuf dtype `PType`
  strict, and it adds a test for the flatbuffer path, which is already strict. Commit 3 makes the
  eleven `PType` fields of array and layout metadata strict.
- Out of scope: New `PType` values and the protobuf `oneof` for wide integers (D5, D20). Edition
  gates for core dtypes (the second half of N10). The detection of old decimal nodes (CD6). Paths
  that are already strict (listed below). The renaming of the test modules named `test`. The
  unused `DictLayoutMetadata` in `vortex-layout/src/layouts/dict/writer.rs:537-550` (see Notes).

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/decimal/vtable/mod.rs` | `DecimalMetadata::get_values_type`, the strict decode in `deserialize` (`:151-186`), tests |
| `vortex-array/src/dtype/serde/proto.rs` | Strict `PType` decode at `:36`, tests |
| `vortex-array/src/dtype/serde/flatbuffers.rs` | The raw value in the error at `:559`, a test |
| `vortex-array/src/arrays/dict/array.rs` | `DictMetadata::get_codes_ptype` beside the struct (`:29-43`), a test module |
| `vortex-array/src/arrays/dict/vtable/mod.rs` | Call `get_codes_ptype` instead of the getter at `:180` |
| `vortex-array/src/arrays/varbin/vtable/mod.rs` | `VarBinMetadata::get_offsets_ptype` (`:60-64`), call at `:180`, a test module |
| `vortex-array/src/arrays/list/vtable/mod.rs` | `ListMetadata::get_offset_ptype` (`:48-54`), call at `:178`, a test module |
| `vortex-array/src/arrays/listview/vtable/mod.rs` | `ListViewMetadata::get_offset_ptype` and `get_size_ptype` (`:56-64`), calls at `:198` and `:205`, a test module |
| `encodings/runend/src/array.rs` | `RunEndMetadata::get_ends_ptype` (`:62-70`), call at `:155`, a test in `mod tests` (`:574`) |
| `encodings/fastlanes/src/rle/vtable/mod.rs` | `RLEMetadata::get_indices_ptype` and `get_values_idx_offsets_ptype` (`:48-62`), calls at `:178` and `:185`, tests in `mod tests` (`:329`) |
| `encodings/alp/src/alp_rd/array.rs` | `ALPRDMetadata::get_left_parts_ptype` (`:61-73`), call at `:182`, a test in `mod test` (`:563`) |
| `vortex-layout/src/layouts/dict/mod.rs` | `DictLayoutMetadata::get_codes_ptype` (`:163-178`), call at `:83`, a test module |
| `vortex-layout/src/layouts/list/mod.rs` | `ListLayoutMetadata::get_offsets_ptype` (`:228-240`), one call that serves `:87` and `:105`, a test module |

Design notes:

The cause. prost-derive 0.14.4, the version in `Cargo.lock`, generates each enumeration getter as
`TryFrom::try_from(raw).unwrap_or(default)` (`prost-derive-0.14.4/src/field/scalar.rs:291-300` in
the cargo registry). The default is the first variant (`src/lib.rs:308`). `DecimalType` and `PType`
both derive `prost::Enumeration` (`vortex-array/src/dtype/decimal/types.rs:25`,
`vortex-array/src/dtype/ptype.rs:33`). So an unknown value reads as `I8` or `U8`.

Silent fallbacks on `develop`, all fixed by FX1:

| Read site | Field | Commit |
| --- | --- | --- |
| `vortex-array/src/arrays/decimal/vtable/mod.rs:170`, `:177` | `DecimalMetadata.values_type` | 1 |
| `vortex-array/src/dtype/serde/proto.rs:36` | `pb::Primitive.r#type` | 2 |
| `vortex-array/src/arrays/dict/vtable/mod.rs:180` | `DictMetadata.codes_ptype` | 3 |
| `vortex-array/src/arrays/varbin/vtable/mod.rs:180` | `VarBinMetadata.offsets_ptype` | 3 |
| `vortex-array/src/arrays/list/vtable/mod.rs:178` | `ListMetadata.offset_ptype` | 3 |
| `vortex-array/src/arrays/listview/vtable/mod.rs:198`, `:205` | `ListViewMetadata.offset_ptype`, `size_ptype` | 3 |
| `encodings/runend/src/array.rs:155` | `RunEndMetadata.ends_ptype` | 3 |
| `encodings/fastlanes/src/rle/vtable/mod.rs:178`, `:185` | `RLEMetadata.indices_ptype`, `values_idx_offsets_ptype` | 3 |
| `encodings/alp/src/alp_rd/array.rs:182` | `ALPRDMetadata.left_parts_ptype` | 3 |
| `vortex-layout/src/layouts/dict/mod.rs:83` | `DictLayoutMetadata.codes_ptype` | 3 |
| `vortex-layout/src/layouts/list/mod.rs:87`, `:105` | `ListLayoutMetadata.offsets_ptype` | 3 |

Already strict, no change:

| Path | Item |
| --- | --- |
| `vortex-array/src/dtype/serde/flatbuffers.rs:543-562` | Flatbuffer `PType` |
| `vortex-array/src/patches.rs:103-124` | `PatchesMetadata` |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/plugin/v1.rs:71` | DecimalByteParts v1 |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/plugin/v2.rs:83`, `:93` | DecimalByteParts v2 |
| `encodings/fastlanes/src/bitpacking/plugin/v2.rs:98` | BitPacked v2 |
| `encodings/datetime-parts/src/array.rs:74-88` | DateTimeParts |
| `encodings/fsst/src/array.rs:85-88`, `:269` | FSST |
| `encodings/onpair/src/array.rs:554-563` | OnPair |
| `vortex-array/src/scalar_fn/fns/operators.rs:84` | Expression `BinaryOp` |

Variant metadata. `VariantMetadataProto.shredded_dtype`
(`vortex-array/src/arrays/variant/vtable/mod.rs:51-55`, decoded at `:154-159`) and
`ParquetVariantMetadataProto.typed_value_dtype` (`encodings/parquet-variant/src/vtable.rs:66-77`,
decoded at `:212-216`) store protobuf dtypes on disk. Neither one holds an enum field directly. Both
call `DType::from_proto`, so commit 2 covers them. Commit 2 also covers the other callers of
`DType::from_proto`: scalars (`vortex-array/src/scalar/proto.rs:198`), cast options
(`vortex-array/src/scalar_fn/fns/cast/mod.rs:91`), `variant_get` options
(`vortex-array/src/scalar_fn/fns/variant_get/mod.rs:88`), the indexed layout
(`vortex-layout/src/layouts/indexed/mod.rs:393`), zstd buffers
(`encodings/zstd/src/zstd_buffers.rs:578`), `vortex-tensor` and `vortex-json`.

Rules for the change:

- Each metadata struct gets one `get_<field>()` accessor per enum field. It returns
  `VortexResult<DecimalType>` or `VortexResult<PType>`. This follows
  `DateTimePartsMetadata::get_days_ptype` and `FSSTMetadata::get_uncompressed_lengths_ptype`. The
  plain field name is taken by the prost getter.
- Put each accessor in an `impl` block right below its struct. Give it the smallest visibility that
  its caller needs.
- Each `deserialize` calls the accessor right after it decodes the metadata. The call comes before
  any other check and before any buffer or child read. So an unknown value always fails first.
- Keep the `enumeration` attributes. They document the wire type, and the setters such as
  `set_codes_ptype` use them. The generated `pb::Primitive::r#type()` getter also stays, because
  generated files are never edited by hand.
- Decode the protobuf field with `pb::PType::try_from(p.r#type)`, then convert with the existing
  `From<pb::PType> for PType` (`proto.rs:229-246`). Do not use `PType::try_from(i32)` here. It ties
  the wire numbers to the discriminants of the Rust enum. The numbers match today
  (`vortex-array/proto/dtype.proto:11-23`, `ptype.rs:37-60`), but the two enums are separate.
- proto3 omits a field that holds 0, and a missing field decodes as 0. Zero is `U8` or `I8`, which
  are valid values. So a file from a current writer reads the same way after FX1.
- Every error states the requirement first and ends with the raw value. Since #10080
  (2026-10-26), `VortexError` has no categories, so `vortex_err!` takes the message only, and a
  test checks the message only.

| Site | Message |
| --- | --- |
| Decimal metadata | `DecimalMetadata values_type must be a known DecimalType, got {raw}` |
| Protobuf dtype | `protobuf Primitive type must be a known PType, got {raw}` |
| Flatbuffer dtype | `flatbuffer Primitive ptype must be a known PType, got {raw}` |
| Array and layout metadata | `{Struct} {field} must be a known PType, got {raw}`, for example `DictMetadata codes_ptype must be a known PType, got 11` |

```rust
impl DecimalMetadata {
    fn get_values_type(&self) -> VortexResult<DecimalType> {
        DecimalType::try_from(self.values_type).map_err(|_| {
            vortex_err!(
                "DecimalMetadata values_type must be a known DecimalType, got {}",
                self.values_type
            )
        })
    }
}

// In `DType::from_proto`.
DtypeType::Primitive(p) => {
    let ptype = pb::PType::try_from(p.r#type).map_err(|_| {
        vortex_err!("protobuf Primitive type must be a known PType, got {}", p.r#type)
    })?;
    Ok(Self::Primitive(ptype.into(), p.nullable.into()))
}
```

Steps:

1. `fix(array): decode decimal values_type strictly`. Add `DecimalMetadata::get_values_type`. In
   `Decimal::deserialize`, call it right after `DecimalMetadata::decode` at `:160`. Use the result
   at `:170` and `:177`. Add the decimal tests.
2. `fix(array): decode protobuf PType strictly`. Change `proto.rs:36` as sketched. Put the raw value
   into the error at `flatbuffers.rs:559`. Add the protobuf and flatbuffer tests.
3. `fix: decode PType metadata fields strictly`. Add the accessors of commit 3 in the table above.
   Call each one at the top of its `deserialize`. In `vortex-layout/src/layouts/list/mod.rs`, bind
   the result once and use it at `:87` and `:105`. Add the accessor tests.

Tests:

- `vortex-array/src/arrays/decimal/vtable/mod.rs`, `mod tests` (`:220`):
  - `test_deserialize_rejects_unknown_values_type`, with named cases `first_unused` (6),
    `negative` (-1) and `max` (`i32::MAX`). Encode `DecimalMetadata { values_type: raw }`. Call
    `Decimal.deserialize` with one host buffer of 4 bytes, the dtype `Decimal(2, 0)` non-nullable,
    length 4 and no children. Assert an error that contains `must be a known DecimalType`. Before
    the fix, case `first_unused` returns `Ok` with an `I8` array.
  - `test_serde_each_values_type`, with one named case for each width at a precision with that
    canonical width: `i8` at `Decimal(2, 0)`, `i16` at `(4, 0)`, `i32` at `(9, 2)`, `i64` at
    `(18, 2)`, `i128` at `(38, 2)` and `i256` at `(76, 2)`. Round-trip as `test_array_serde` does
    (`:241-273`). Assert the decoded `values_type()` and use `assert_arrays_eq!`. Every case uses
    the canonical width, so CD8a keeps all of them valid.
- `vortex-array/src/dtype/serde/proto.rs`, `mod tests` (`:286`):
  - `test_from_proto_rejects_unknown_ptype`, with named cases `first_unused` (11), `negative` (-1)
    and `max` (`i32::MAX`). Assert an error that contains `must be a known PType`. Before the fix,
    every case returns `Primitive(U8)`.
  - `test_from_proto_rejects_unknown_ptype_in_struct_field`. A struct dtype with one field whose
    `PType` is 11. This is the shape of a shredded Variant dtype. Assert an error.
- `vortex-array/src/dtype/serde/flatbuffers.rs`, `mod test` (`:565`):
  `test_unknown_ptype_from_flatbuffer`. Assert that `PType::try_from(fb::PType(11))` returns an
  error that contains `got 11`. The path is strict today, so this test covers the message only.
- Commit 3: one test per struct, named `test_get_<field>_rejects_unknown_ptype`. Use named cases
  where a struct has two fields (`ListViewMetadata`, `RLEMetadata`). Each test builds the struct
  with the field set to 11 and asserts an error that names the struct and the field. One more case
  with a valid value asserts `Ok`. Put the test in the file that defines the struct. Add a
  `#[cfg(test)] mod tests` at the bottom of `dict/array.rs`, `varbin/vtable/mod.rs`,
  `list/vtable/mod.rs`, `listview/vtable/mod.rs`, `vortex-layout/src/layouts/dict/mod.rs` and
  `vortex-layout/src/layouts/list/mod.rs`.
- No existing test changes.

Benchmarks: None. Each read adds one integer match per enum field. CodSpeed is expected to show
no change.

Acceptance checks:

- `cargo clippy -p vortex-array -p vortex-layout -p vortex-runend -p vortex-fastlanes -p vortex-alp --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-array -p vortex-layout -p vortex-runend -p vortex-fastlanes -p vortex-alp`
- `cargo nextest run -p vortex-array -p vortex-layout -p vortex-runend -p vortex-fastlanes -p vortex-alp -p vortex-parquet-variant`
- `uv run vortex-test/compat-gen/scripts/compat.py check --mode all`. It reads every published
  fixture, so it shows that no published file holds a value outside the enums.
- An unknown stored value returns an error. No read path panics on it.
- No read path in the table above calls a prost enumeration getter.

Review focus:

- Every read site in the table calls its accessor, and the call comes before any other check.
- `rg -n 'metadata\.\w+_ptype\(\)|metadata\.values_type\(\)' | rg -v '\.get_'` finds no read
  path.
- The error messages follow the table, with the raw value last.
- The wire format and the setters do not change.
- An old file with a value outside the enum now fails. Before FX1 it read as `U8` or `I8`. Whether
  such a file exists is Not verified. The compat check covers the published fixtures.
- CD6 keeps `get_values_type()` as the first step of its detection.

Size: medium. Thirteen fields in five crates, each with a small test.

Open questions: None. CD6 needs only commit 1, but the maintainer chose one PR for all three
commits (D34).

### FX2: Dropped

FX2 added the `trivially_cast_nullability` guard and a `CastKernel` to the DecimalByteParts cast
rule (#10746). It closed without a merge on 2026-10-26. The rule pushes the nullability cast into
the MSP, which holds the validity of the array, and the MSP cast fails on a null row. So no result
was wrong. #10798 documents the two valid forms of a cast rule on `CastReduce`:

- A rule that builds a new `Validity` returns `Ok(None)` when `trivially_cast_nullability` does,
  and the encoding implements `CastKernel` for that case.
- A rule that pushes the cast into the children that hold its validity needs neither, as long as
  every null in those children reaches a row of the array. `Dict` is the exception, because its
  values can hold nulls that no code references.

Issue #10750 listed seven more push-down rules, and it closed as not planned for the same reason.

### FX3: Dictionary `fill_null` no longer truncates the fill code

- Goal: The dictionary `fill_null` kernel stops truncating the index of the fill value into the
  codes type. When the index does not fit the codes dtype, the kernel declines, and the generic
  path fills the decoded array.
- Depends on: None. Soft: None. Blocks: WD5. With WD5, `optimize()` unwraps a `Widen` over
  dictionary codes to a narrow child
  (`af4b663059:vortex-array/src/arrays/narrow/tests/merge.rs:29-41`). This makes the bug easy to
  reach. WD5 also carries the `Widen` regression test of this fix.
- Decisions: None by ID. `decisions.md`, section "The `Widen` pull requests", makes this fix its
  own PR.
- Branch: `ct/dict-fill-code`, based on `develop`.
- Scope: The fix of local commit `7e2e5e4d7a` to
  `vortex-array/src/arrays/dict/compute/fill_null.rs`, and its `Widen`-free regression test,
  adapted to the current test module.
- Out of scope: The `Widen` test
  `7e2e5e4d7a:vortex-array/src/arrays/narrow/tests/merge.rs:45-62`, which WD5 owns. Any change to
  the "no fill value found" branch at `fill_null.rs:45-55`.

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/dict/compute/fill_null.rs` | Checked fill code at `:57-76`, two unused imports removed (`:19`, `:21`), a regression test in `mod tests` (`:90-131`) |

Design notes:

How the truncation happens on `develop`, without `Widen`:

1. The kernel compares every dictionary value with the fill value (`fill_null.rs:33-40`). This
   includes values that no code references.
2. It takes the first match as a `usize` index (`:43-44`).
3. It converts the index with `as P` into the integer type of the codes (`:61-69`). The
   `#[expect(clippy::cast_possible_truncation)]` at `:63-66` assumes that the index fits.
4. A dictionary can hold more values than its codes type can index. `DictArray::try_new` and
   `Dict::validate` check only the codes type, the lengths and the dtypes
   (`dict/array.rs:303-315`, `dict/vtable/mod.rs:97-116`).
5. So `u8` codes `[0, null]` over the 301 values `0..=300` form a valid dictionary. With fill value
   300, `300 as u8` is 44, and the null row reads value 44. With fill value 256, it reads value 0.
6. With `i8` codes and index 128, the code becomes -128, and the read panics. A run of the FX3
   test without the fix showed the panic on 2026-10-23.

Whether a producer on `develop` other than direct construction builds such a dictionary is Not
verified.

The fix, as in `7e2e5e4d7a`:

```rust
let codes = array.codes();
// An unreferenced dictionary entry can exceed the codes' range.
let Ok(fill_code) =
    Scalar::from(existing_fill_value_index).cast(&codes.dtype().as_nonnullable())
else {
    return Ok(None);
};

// Fill nulls in both the codes and the values. ...
let codes = codes.clone().fill_null(fill_code)?;
```

- `Scalar::from(usize)` builds a `u64` scalar
  (`vortex-array/src/scalar/convert/primitive.rs:199-203`). The cast to an integer dtype is
  checked. `PValue::cast` returns an error for a value that does not fit
  (`vortex-array/src/scalar/typed_view/primitive/pvalue.rs:287-303`).
- `Ok(None)` declines the kernel. `FillNull::execute` then decodes the dictionary and fills the
  result (`vortex-array/src/scalar_fn/fns/fill_null/mod.rs:153-163`).
- The first match has the smallest index. If it does not fit, no later match fits. So declining on
  the first match loses no fast path.
- Remove the imports `match_each_integer_ptype` (`:19`) and `ScalarValue` (`:21`). Only the
  replaced code uses them.

Steps:

1. `fix(array): avoid truncating dictionary fill codes`. Apply the kernel change and the test in
   one commit, as `7e2e5e4d7a` does.

Tests:

- Port `fill_unreferenced_value` from `7e2e5e4d7a:vortex-array/src/arrays/dict/compute/fill_null.rs`
  into `mod tests` at `fill_null.rs:90-131`. PR #10681 rewrote this module (commit `d834a11faa`),
  so adapt the test:
  - Use the module `SESSION` (`:109`) with `SESSION.create_execution_ctx()`, not `array_session()`.
  - Drop the explicit `.execute::<PrimitiveArray>` before `assert_arrays_eq!`. The macro executes
    the lazy result (`AGENTS.md`, Tests), and PR #10458 removed the same pattern from this module.
  - Add `use rstest::rstest;` and `use vortex_error::VortexResult;`.
- The test reaches the dictionary kernel. `VortexSession::empty()` installs the built-in kernels on
  first use (`vortex-array/src/optimizer/kernels.rs:336-349`), including this one
  (`dict/vtable/kernel.rs:25`). The codes are partially valid, so `short_circuit` does not answer
  first (`vortex-array/src/scalar_fn/fns/fill_null/kernel.rs:67-93`).
- Cases. Each dictionary has the codes `[0, null]` and the values `0..=fill`. The results before the
  fix come from reading the code, not from a run.

| Case | Codes type | Fill | Before the fix | After the fix |
| --- | --- | --- | --- | --- |
| `u8_max` | `u8` | 255 | `[0, 255]`, passes | `[0, 255]` |
| `u8_overflow` | `u8` | 256 | `[0, 0]`, fails | `[0, 256]` |
| `u8_truncation` | `u8` | 300 | `[0, 44]`, fails | `[0, 300]` |
| `i8_max` | `i8` | 127 | `[0, 127]`, passes | `[0, 127]` |
| `i8_overflow` | `i8` | 128 | Code -128, panics | `[0, 128]` |

- `nullable_codes_fill_in_values` stays as it is.

Benchmarks: None. The change adds one checked scalar cast per kernel call.

Acceptance checks:

- `cargo clippy -p vortex-array --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-array`
- `cargo nextest run -p vortex-array dict::compute::fill_null`
- With the kernel change reverted, `u8_overflow` and `u8_truncation` fail.

Review focus:

- No `as` conversion of the index remains, and the `#[expect(clippy::cast_possible_truncation)]` is
  gone.
- The declined case gives the same result as the "no fill value found" branch.
- The test style matches the module after PR #10681.

Size: small. One kernel branch and one parameterized test.

Open questions: None.

## Notes from planning

These facts came up while the briefs were drafted. They do not change the scope of the PRs.

1. FX1 decodes the protobuf field with `pb::PType::try_from` and the existing
   `From<pb::PType> for PType`, not with `PType::try_from(i32)`. The wire numbers then stay tied to
   the protobuf enum. The numbers of the two enums match today.
2. The flatbuffer `PType` path is already strict
   (`vortex-array/src/dtype/serde/flatbuffers.rs:559`). FX1 only adds the raw value to its message
   and a test.
3. No Variant metadata field holds an enum. Variant and Parquet-variant metadata store protobuf
   dtypes, so the protobuf fix covers them.
4. `vortex-layout/src/layouts/dict/writer.rs:537-550` defines a second `DictLayoutMetadata` that no
   code uses. Its comment says to read the field with the generated getter, which FX1 makes wrong
   advice. Its removal belongs in a separate change.
5. FX2 was dropped. Its section gives the reason.
6. Commit `7e2e5e4d7a` has two tests. `fill_unreferenced_value` already avoids `Widen`, and FX3
   ports it. The `Widen` test belongs to WD5, so WD5 lands after FX3.
7. On `develop`, the FX3 bug is shown only for a dictionary built directly. Whether the compressor
   or a file read builds a dictionary with more values than its codes type can index is Not
   verified.
8. The test modules in `vortex-array/src/dtype/serde/flatbuffers.rs:565` and
   `encodings/alp/src/alp_rd/array.rs:563` are named `test`, which `AGENTS.md` discourages. FX1
   adds tests to them but does not rename them.
