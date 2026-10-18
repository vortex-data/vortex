<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Extension type design notes (agent reference)

This file is the detailed design behind `../03-design.md` and `../05-implementation.md`. It is a
proposal, not an accepted design. "Open" marks a decision that is not made. Finding IDs (`EXT-*`)
refer to `findings.md`. Paths and line numbers are for `develop` at `5fc5cb8`.

## 1. Invariants

The design holds these invariants. Every change in the plan must keep them.

- **I1.** A semantic operation with an extension operand is rejected, unless the extension type
  declares a derive for it or provides a rewrite for it.
- **I2.** A structural operation on an extension value is always allowed and keeps the extension
  dtype.
- **I3.** The binder is the only code that decides extension semantics. It runs in
  `Expression::bind` (`vortex-array/src/expr/bound_expression.rs:304`) and when a scalar function
  array is built (`vortex-array/src/arrays/scalar_fn/array.rs:79`).
- **I4.** After binding, no semantic operation has an extension operand. Kernels, encodings,
  statistics, pruning and compression see storage dtypes only.
- **I5.** The only ways to move between an extension dtype and another dtype are `ext_storage`,
  `ext_wrap`, and cast hooks that the extension type owns.
- **I6.** The row path and the array path evaluate the same bound expression, so they return the
  same result (EXT-P1).
- **I7.** Reading a value never panics. A value that the type rejects produces an error.
- **I8.** A nullability change never fails and never validates the type again.
- **I9.** An unknown extension type (`ForeignExtDType`) declares no derives and no rewrites.

## 2. Operation classes

Structural operations do not read values. Semantic operations depend on what values mean.

| Operation | Class | Note |
| --- | --- | --- |
| slice, take, filter, mask | Structural | Run on storage, keep the wrapper |
| `is_null`, `is_not_null`, validity, `null_count`, `count` | Structural | |
| `fill_null` with a value of the same dtype | Structural | Selects between existing values |
| `zip`, `case_when` with branches of the same dtype | Structural | Selects between existing values |
| `first`, `last` | Structural | Selects an existing value |
| `uncompressed_size_in_bytes` | Structural | Size of the storage |
| `pack`, `merge`, `select`, `get_item` on a struct that holds extension fields | Structural | The extension field stays opaque |
| `scalar_at`, serialization, compression of storage | Structural | |
| `ext_storage` | Structural | Explicit unwrap |
| `=`, `!=` | Semantic | Needs `EQ` or a rewrite |
| `<`, `<=`, `>`, `>=`, `between` | Semantic | Needs `ORD` or a rewrite |
| `min`, `max`, `min_max`, `bounded_min`, `bounded_max`, `is_sorted`, sort keys | Semantic | Needs `ORD` or a rewrite |
| `is_constant`, `all_non_distinct`, `list_contains`, distinct, group by, join keys | Semantic | Needs `EQ` or a rewrite |
| Arithmetic, `sum`, `sum_v2`, `mean` | Semantic | Needs a rewrite, because the result type usually changes |
| `cast` other than a nullability change | Semantic | Needs a cast hook |
| `is_nan`, `nan_count`, `like`, `byte_length` | Semantic | Needs a rewrite |
| Zero and default values | Semantic | Needs a `zero_value` hook |
| Export to Arrow, DuckDB, DataFusion, Python, JNI | Semantic | Needs an export plugin |

## 3. Derives

### Definition

```rust
/// Storage behavior that two values of one extension type share.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Derives {
    /// Storage equality is value equality, in both directions. The storage hash is then a valid
    /// hash for value equality.
    pub eq: bool,
    /// Storage order is value order. `ord` requires `eq`.
    pub ord: bool,
}

pub trait ExtVTable {
    // The existing methods stay.

    /// Returns the storage behavior that values of `lhs` and `rhs` share.
    ///
    /// Both dtypes have this extension type. The default shares nothing.
    fn derives(lhs: &ExtDType<Self>, rhs: &ExtDType<Self>) -> Derives {
        Derives::default()
    }
}
```

`ExtDTypeRef` gets an erased forwarder: `fn derives(&self, other: &ExtDTypeRef) -> Derives`. It
returns `Derives::default()` when the two extension IDs differ.

### Rules

- **D1.** A derive applies only when the two storage dtypes are equal, ignoring nullability. For
  other pairs, the type must provide a rewrite, for example a cast of one storage to the other.
- **D2.** `derives(a, b)` must equal `derives(b, a)`. The conformance harness checks this.
- **D3.** `eq` requires canonical storage: each value has exactly one storage form under both
  metadata values.
- **D4.** `ord` requires `eq`, and the storage order must be a total order on values.
- **D5.** A type with `eq` must not let two different metadata values give different meanings to
  the same storage value. If they do, `derives` returns no `eq` for that pair.

### Table for the types in the repository

| Type | Pair | `eq` | `ord` | Reason |
| --- | --- | --- | --- | --- |
| `Timestamp` | Same unit, both with a time zone | Yes | Yes | Both store UTC instants |
| `Timestamp` | Same unit, both without a time zone | Yes | Yes | Both store wall-clock counts |
| `Timestamp` | One with and one without a time zone | No | No | Different meanings |
| `Timestamp` | Different units | No | No | Needs a cast first |
| `Date`, `Time` | Same unit | Yes | Yes | |
| `Uuid` | Any version metadata | Yes | Open | Byte order is the Postgres order |
| `Json` | Any | No | No | Text is not canonical |
| `Vector` | Same dimension and element type | Yes | No | No natural order |
| `FixedShapeTensor` | Equal metadata | Yes | No | Shape and permutation change the meaning |
| WKB and native geometry types | Any | No | No | Two encodings can hold one geometry |
| `ForeignExtDType` | Any | No | No | I9 |

The float element types of `Vector` give storage equality IEEE semantics (`NaN != NaN`). That is
correct for a derive, because the derive inherits the storage semantics on purpose.

## 4. Binder

### Resolution order

For a scalar function call that has at least one extension operand:

1. If the function is structural (section 2), bind it without a change.
2. The function can be a comparison, `between`, or an order-based aggregate. If the extension type
   declares the matching derive for the operand pair, rewrite the call to the same operation on
   `ext_storage` of each operand.
3. If the extension type provides a rewrite (`ExtVTable::rewrite`), use it.
4. If the function itself handles the extension type (for example `replace_time_zone` for
   `Timestamp`), bind it without a change. The function owner then owns the semantics.
5. Otherwise, return an error that names the function, the extension ID, and the missing derive.

For a cast, the binder tries these steps in order:

1. A nullability change of the same dtype.
2. `cast_to` of the source type.
3. `cast_from` of the target type.
4. An error.

### Proposed hooks

```rust
pub trait ExtVTable {
    /// Rewrites a cast from this type into an expression, or returns `None`.
    fn cast_to(
        ext: &ExtDType<Self>,
        input: Expression,
        target: &DType,
    ) -> VortexResult<Option<Expression>> {
        Ok(None)
    }

    /// Rewrites a cast into this type into an expression, or returns `None`.
    fn cast_from(
        ext: &ExtDType<Self>,
        input: Expression,
        source: &DType,
    ) -> VortexResult<Option<Expression>> {
        Ok(None)
    }

    /// Rewrites a scalar function call that has an operand of this type, or returns `None`.
    fn rewrite(ext: &ExtDType<Self>, call: &ScalarFnCall) -> VortexResult<Option<Expression>> {
        Ok(None)
    }
}
```

`ScalarFnCall` is a new view of the function, its options and its bound arguments. Aggregates
need the same pair of hooks (`rewrite_aggregate`). Open: whether a session registry keyed by
`(function ID, extension ID)` also exists, so that a function owner can support a type that it does
not own.

### Examples

```text
Timestamp(ms) a  <  Timestamp(ms) b        =>  ext_storage(a) < ext_storage(b)
Timestamp(ms) a  <  lit(Timestamp(ms) 5)   =>  ext_storage(a) < lit(5i64)
min(Timestamp(ms) a)                       =>  ext_wrap(min(ext_storage(a)), Timestamp(ms), unchecked)
cast(Timestamp(ms) a, Timestamp(ns))       =>  ext_wrap(checked_mul(ext_storage(a), 1_000_000),
                                                        Timestamp(ns), unchecked)
cast(Timestamp(ns) a, Timestamp(ms))       =>  rejected unless the type defines rounding
DecimalLike(10, 1) a  <  DecimalLike(10, 2) b
                                           =>  checked_mul(ext_storage(a), 10) < ext_storage(b)
                                               (a rewrite that the type owns)
Json a  =  Json b                          =>  rejected
```

`ext_wrap(..., unchecked)` is correct in these examples, because `min` returns an existing value and
the unit cast keeps the range check in `checked_mul`.

### Array API

The array API builds `ScalarFnArray` values directly. The binder must run in
`ScalarFnArray::try_new` (or in the typed constructors such as `Binary::try_new`). It returns the
rewritten tree, so the new array holds `ext_storage` children and an `ext_wrap` parent where needed.
The row path (`vortex-array/src/arrays/scalar_fn/vtable/operations.rs:30-42`) then evaluates the
rewritten tree, and I6 holds without other changes.

Open: some rewrites can need session state. The hooks above use only the `ExtDType`, which travels
with the dtype, so the binder does not need a session.

### Literals

- `ext_storage(lit(s))` simplifies to `lit(s.to_storage_scalar())`.
- `ext_wrap(ext_storage(x), dtype(x))` simplifies to `x`.
- `ext_storage(ext_wrap(y, d))` simplifies to `y`.

## 5. `ext_wrap`

- Options: the target `ExtDTypeRef`, and a flag that is `checked` or `unchecked`.
- `return_dtype`: the target dtype, with the nullability of the input.
- Argument: must have the storage dtype of the target, ignoring nullability.
- Checked execution: calls `validate_array` (section 8), then builds the wrapper.
- Unchecked execution: builds the wrapper. Only the binder and kernels that keep the invariants of
  the type use it.
- Row path: `Scalar::try_new` for checked, `Scalar::new_unchecked` for unchecked.

## 6. Normal form

Only the wrapper (`vortex.ext`) carries an extension dtype. These reduce rules move the wrapper to
the top:

| From | To |
| --- | --- |
| `Constant(extension scalar s)` | `Extension(Constant(s.to_storage_scalar()))` |
| `Chunked[Extension(d, a_i)]` | `Extension(d, Chunked[a_i])` |
| `Dict(codes, Extension(d, values))` | `Extension(d, Dict(codes, values))` |
| `Masked(Extension(d, x), m)` | `Extension(d with the result nullability, Masked(x, m))` |
| `Sparse` with extension values | `Extension(Sparse` over storage`)` |

Consequences:

- `ExtensionConstantRule` (`vortex-array/src/arrays/extension/compute/rules.rs:28-46`) reverses
  direction.
- `ArrayRef::as_constant()` must look through the wrapper, or a new helper does it. Many kernels
  call `as_constant()` on an operand.
- `encodings/sparse/src/canonical.rs:360-362` no longer needs to reject extension arrays.
- `DateTimeParts` becomes an `i64` encoding with the time unit in its metadata. Today its dtype is
  `Timestamp` (`encodings/datetime-parts/src/canonical.rs:28`). Readers convert the old form on
  read. An edition gates writes of the new form. `vortex-btrblocks/src/schemes/temporal.rs` chooses
  the encoding from the extension metadata, and compresses the storage child.
- The same applies to any future encoding for one extension type.

Open: the direction of the normal form. The alternative is "wrapper below generic encodings":
`Constant(extension scalar)`, `Dict(codes, Extension(d, values))` and `Chunked[Extension(d, a_i)]`
stay as they are, and only value-reading encodings must sit under the wrapper. That matches
`ExtensionConstantRule` and the legacy decimal wire form. It has two costs. `ext_storage` needs a
pushdown rule for each generic encoding, or a fallback that executes the encoding to the canonical
wrapper. Also, each generic encoding keeps an extension arm in its canonicalization. The wrapper on
top removes those arms. The Decimal report (`../../decimal/agents/report.md`, sections 3.3 and
3.5) uses the wrapper on top in memory, and its wire plugin writes the older shapes.

## 7. Statistics

- **S1.** Zone maps and file statistics for an extension column store statistics of the storage,
  with the storage dtype. The fallback in `vortex-layout/src/layouts/zoned/schema.rs:112-121`
  becomes the rule.
- **S2.** A new stats rewrite rule maps `stat(ext_storage(x), s)` to the storage statistic `s` of
  `x`. Bound predicates reference `ext_storage(col)` after binding, so pruning needs only this rule.
- **S3.** Readers convert existing extension-typed min and max to storage scalars with
  `to_storage_scalar()`. The values are already storage values.
- **S4.** Engine statistics export (DuckDB `column_statistics.rs`, DataFusion `convert/stats`, JNI
  write summaries) goes through the export plugin of the type. A type without `ord` exports no min
  and max.
- **S5.** Bounded min and max (truncation) apply to the storage dtype, so they also apply to
  extensions over `Utf8` and `Binary`.

## 8. Validation

```rust
pub trait ExtVTable {
    /// Checks that every non-null value of `storage` is valid for this type.
    ///
    /// The default accepts every value. A type with value-level rules overrides it.
    fn validate_array(
        ext: &ExtDType<Self>,
        storage: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        Ok(())
    }
}
```

- Callers: `ExtensionArray::try_new`, `ext_wrap` (checked), Arrow import, and file reads when a read
  option enables it.
- Kernels that keep the invariants use `ExtensionArray::new_unchecked`.
- A row read from an array uses `Scalar::new_unchecked`. The per-row validation in the current read
  path (`vortex-array/src/arrays/extension/vtable/operations.rs:29`) then goes away.
- Examples:
  - `Timestamp` computes the storage min and max, then checks only those two values with `jiff`.
  - `Uuid` checks the version nibble of each value against the metadata.
  - `NormalizedVector` checks the norm of each row within a tolerance.
- Open: is the default a no-op, or a check of each value through `validate_scalar_value`?

## 9. Nullability

- **N1, short term.** `ExtDType::try_with_vtable` calls `validate_dtype` for both nullabilities of
  the storage dtype, and fails if either call fails. The documentation states that the result of
  `validate_dtype` must not depend on nullability. `with_nullability` then never validates again. A
  type that breaks the rule fails when its dtype is built, not later in a kernel. Calling
  `validate_dtype` only with the non-nullable form is the alternative, but it hides such a type.
- **N2, long term.** `ExtDType` stores a non-nullable storage dtype and a `Nullability`, like every
  other `DType` variant. The storage child has the dtype `storage_dtype.with_nullability(n)`. The
  wire format does not change, because the serialized storage dtype already carries nullability.

## 10. Zero values

```rust
pub trait ExtVTable {
    /// Returns the zero of this type, or `None` if the type has no zero.
    fn zero_value(ext: &ExtDType<Self>) -> Option<ScalarValue> {
        None
    }
}
```

- `Scalar::zero_value` and `ScalarValue::try_zero_value` use this hook for extension dtypes.
- Open: builders pad a non-nullable child under a null parent
  (`vortex-array/src/builders/fixed_size_list.rs:288` and `:303`). A type with no zero has no valid
  padding value. One option is that `validate_array` checks only rows that are valid in every
  ancestor. Another option is that such a child must be nullable.

## 11. Scalar identity

`Scalar` implements `PartialEq` and `Hash` for Rust code: tests, maps of constants, and
deduplication. Keep these structural: two scalars are equal when the dtypes are equal (ignoring
nullability) and the storage values are equal. Document this as identity, not as SQL equality.
`PartialOrd` for extension scalars must follow `derives`, because ordering is semantic. The
compare paths (`scalar_cmp`) must go through the binder rules, not through `PartialOrd`.

## 12. Files to change, by phase

| Phase | Files |
| --- | --- |
| 0, U1 | `vortex-array/src/scalar/cast.rs`, `vortex-array/src/arrays/extension/compute/cast.rs`, `vortex-arrow/src/executor/{primitive,byte_view,struct_}.rs` |
| 0, U2 | `vortex-array/src/scalar_fn/fns/binary/mod.rs`, `vortex-array/src/scalar_fn/fns/binary/compare/mod.rs`, `encodings/datetime-parts/src/compute/compare.rs` |
| 0, U3 | `vortex-array/src/arrays/extension/vtable/operations.rs`, `vortex-array/src/arrays/extension/compute/rules.rs`, `vortex-array/src/scalar/constructor.rs` |
| 0, U4 | `vortex-array/src/dtype/extension/typed.rs`, `vortex-array/src/dtype/extension/vtable.rs` |
| 0, U5 | `vortex-array/src/extension/datetime/{date,time,timestamp}.rs` |
| 0, U6 | `vortex-array/src/arrays/extension/vtable/mod.rs` |
| 0, U7 | `vortex-array/src/scalar/scalar_impl.rs`, `vortex-array/src/scalar/scalar_value.rs` |
| 0, U8 | `vortex-array/src/dtype/extension/erased.rs` |
| 2 | `vortex-array/src/dtype/extension/{vtable,typed,erased}.rs`, `vortex-array/src/expr/bound_expression.rs`, `vortex-array/src/arrays/scalar_fn/array.rs`, `vortex-array/src/scalar_fn/fns/{binary,between}/`, `vortex-array/src/aggregate_fn/fns/{min,max,min_max,bounded_min,bounded_max,is_sorted}/`, `vortex-array/src/arrays/extension/compute/compare.rs`, `vortex-array/src/scalar/typed_view/extension/mod.rs`, `vortex-row/src/encoder.rs`, `vortex-duckdb/src/table_function.rs` |
| 3 | `vortex-array/src/scalar_fn/fns/cast/mod.rs`, `vortex-array/src/extension/datetime/`, `vortex-duckdb/src/convert/expr.rs`, `vortex-datafusion/src/convert/exprs/mod.rs` |
| 4 | `vortex-array/src/arrays/extension/compute/rules.rs`, `vortex-array/src/arrays/{constant,chunked,dict,masked}/`, `encodings/sparse/src/canonical.rs`, `encodings/datetime-parts/`, `vortex-btrblocks/src/schemes/temporal.rs`, `vortex-layout/src/layouts/zoned/`, `vortex-layout/src/layouts/file_stats.rs`, `vortex-file/src/pruning.rs`, `vortex-array/src/stats/rewrite/`, `vortex-edition/` |
| 5 | `vortex-array/src/arrays/extension/array.rs`, `vortex-array/src/builders/extension.rs`, `vortex-array/src/dtype/extension/` |
| 6 | `vortex-arrow/src/session.rs`, `vortex-duckdb/src/convert/`, `vortex-duckdb/src/exporter/`, `vortex-datafusion/src/convert/`, `vortex-python/src/scalar/`, `docs/developer-guide/extending/extension-dtypes.md` |

## 13. Tests

- **Path consistency.** A helper in `vortex-array/src/compute/conformance/` takes an expression
  builder and inputs. It evaluates the expression on the array path, on the row path, and with
  constant-encoded inputs. It asserts equal values or an error on all three.
- **Reference types.** `vortex-array/src/extension/tests/` already has `DivisibleInt`. Add:
  - `Opaque` over `i64`, with no derives. Every semantic operation fails when it is bound.
  - `DecimalLike` over `i64`, with `eq` and `ord` for the same scale and a rescale rewrite. A
    differential test runs each operation on `DecimalLike` and on the built-in `Decimal`, and
    compares values or errors.
  - `Refined` over `i64` (for example, positive values only), with every derive and a
    `validate_array` check.
- **Conformance harness for `ExtVTable`.** Checks the metadata round trip, that `validate_dtype`
  ignores nullability, rules D1 to D4, cast round trips, and path consistency.
- **Engines.** DuckDB and DataFusion pushdown tests with extension literals for `Timestamp`, `Date`,
  `Uuid` and `Json` columns. A `Json` comparison must not be pushed down.
