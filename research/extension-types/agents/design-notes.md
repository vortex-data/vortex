<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Extension type design notes (agent reference)

This file is the detailed design behind `../03-design.md` and `../05-implementation.md`. It is a
proposal, not an accepted design. "Open" marks a decision that is not made. Finding IDs (`EXT-*`)
refer to `findings.md`, which covers current code only. Probe IDs (R1 to R10) refer to appendix A
of `findings.md`. Paths and line numbers are for `develop` at `5fc5cb8`.

## 1. Invariants

The design holds these invariants. Every change in the plan must keep them.

- **I1.** A semantic operation on an operand that contains an extension dtype is rejected by
  default. A declared derive, a type-owned rewrite, or a declaration of the function can allow it
  (sections 3 and 4).
- **I2.** A structural operation on an extension value is always allowed and keeps the extension
  dtype.
- **I3.** The binder is the only code that decides how a generic operation treats extension
  values. It runs at three kinds of sites:
  - `Expression::bind` (`vortex-array/src/expr/bound_expression.rs:304`).
  - The construction of a scalar function array (`vortex-array/src/arrays/scalar_fn/array.rs:79`).
  - Every site that binds an aggregate to an input dtype, for example the zone-map writer
    (`vortex-layout/src/layouts/zoned/writer.rs:217-247`).
- **I4.** After binding, a generic operation applies storage semantics to an extension value only
  where the binder checked a declared derive (section 4.3). The kernels of generic semantic
  operations, statistics and pruning see storage dtypes only. Whether generic encodings and
  compression carry extension dtypes depends on the physical form (section 6). A type-aware
  function sees the extension types that it declares. `DateTimeParts` stays an exception until
  phase 4 (EXT-S2).
- **I5.** The only ways to move between an extension dtype and another dtype are `ext_storage`,
  `ext_wrap`, and cast hooks that the extension type owns.
- **I6.** The row path and the array path evaluate the same bound expression, so they return the
  same result (EXT-P1). This also applies to an unvalidated array with invalid values.
- **I7.** Reading a value never panics. A read from an unvalidated array checks the value. A value
  that the type rejects produces an error (section 8).
- **I8.** A nullability change never fails and never validates the type again.
- **I9.** An unknown extension type (`ForeignExtDType`) declares no derives and no rewrites, and
  no function declaration matches it.

## 2. Operation classes

Structural operations do not read values. Semantic operations depend on what values mean. A
type-aware function implements the semantics of the extension types that it declares.

| Operation | Class | Note |
| --- | --- | --- |
| slice, take, filter, mask | Structural | Run on storage, keep the wrapper |
| `is_null`, `is_not_null`, validity, `null_count`, `count` | Structural | |
| `fill_null` with a value of the same dtype | Structural | Selects between existing values |
| `zip`, `case_when` with branches of the same dtype | Structural | Selects between existing values |
| `first`, `last` | Structural | Selects an existing value |
| `uncompressed_size_in_bytes` | Structural | Size of the storage |
| `pack`, `merge`, `select`, `get_item` on a struct that holds extension fields | Structural | The extension field stays opaque |
| `cast` to the same dtype with another nullability | Structural | |
| `scalar_at`, serialization, compression of storage | Structural | Constant detection for compression compares storage |
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
| `replace_time_zone`, tensor functions, spatial functions | Type-aware | The function owner declares the types (section 4.2) |

## 3. Derives

### 3.1 Definition

A derive is a declaration on one extension dtype. It says that a semantic operation on values of
that dtype is the same operation on their storage values.

```rust
/// Storage behavior that values of one extension dtype inherit, like `#[derive]` on a newtype.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Derives {
    /// Storage equality is value equality for values of this dtype. The storage hash is then a
    /// valid hash for value equality.
    pub eq: bool,
    /// Storage order is value order for values of this dtype. `ord` requires `eq`.
    pub ord: bool,
}

pub trait ExtVTable {
    // The existing methods stay.

    /// Returns the storage behavior that values of `ext` inherit.
    ///
    /// The default inherits nothing, so every semantic operation on this dtype is rejected.
    fn derives(ext: &ExtDType<Self>) -> Derives {
        Derives::default()
    }
}
```

`ExtDTypeRef` gets an erased forwarder: `fn derives(&self) -> Derives`. The binder reads the flags
from the non-nullable form of the dtype. So a type that breaks D4 still gives one answer.

### 3.2 Rules

- **D1.** A derive applies only when every operand in a derived position has the same extension
  dtype, ignoring nullability. A lone extension operand never gets a derive, so `ts < 5i64` stays
  rejected. Operations across dtypes need an explicit rule (section 3.5).
- **D2.** `eq` requires canonical storage: each value of the dtype has exactly one storage form.
- **D3.** `ord` requires `eq`, and storage order must equal value order.
- **D4.** The flags do not depend on nullability.

The conformance harness checks D4, and that `ord` implies `eq`. It checks D2 and D3 only when a
type supplies an optional reference order on native values for tests. `NativeValue` requires only
`Display` (`vortex-array/src/dtype/extension/vtable.rs:29-32`), so the harness has no other oracle.

### 3.3 Use by operation

| Operation | Operands in derived positions | Flag |
| --- | --- | --- |
| `=`, `!=` | Both operands | `eq` |
| `<`, `<=`, `>`, `>=` | Both operands | `ord` |
| `between` | All three operands. `Between::return_dtype` already requires one dtype (`vortex-array/src/scalar_fn/fns/between/mod.rs:282-305`) | `ord` |
| `min`, `max`, `min_max`, `bounded_min`, `bounded_max`, `is_sorted`, sort keys | The one input | `ord` |
| `is_constant`, `all_non_distinct`, distinct, group by | The one input | `eq` |
| `list_contains` | The list element and the needle | `eq` |
| Join keys, hash partitioning | The key columns on both sides | `eq` |

An aggregate binds to one input dtype (`AggregateFnVTable::return_dtype`,
`vortex-array/src/aggregate_fn/vtable.rs:170`). So a unary operation reads the flags of its one
input, and it never needs a second dtype. A function outside this table gets no derive. It uses a
declaration or a type-owned rewrite (section 4).

### 3.4 Containers

A struct, list, fixed-size list or map dtype that holds extension values has `eq` or `ord` only if
every nested extension dtype declares it. An order comparison of two `struct{ts: Timestamp(ms)}`
values needs `ord` on `Timestamp(ms)`. Any comparison of two `struct{doc: Json}` values is
rejected. The nested comparator reads nested extension values by storage
(`vortex-array/src/scalar_fn/fns/binary/compare/nested.rs:229-233`). It does so only after the
binder checked these flags.

### 3.5 Operations across dtypes

Derives never apply across dtypes. Two dtypes with different metadata are different dtypes, for
example `Timestamp(ms)` and `Timestamp(ns)`, or two time zones. The binder rejects such an
operation by default. A type can support one only through an explicit rule that the type owns:

- **Compatibility.** The type states that two of its dtypes compare by storage as they are. An
  example is two zoned timestamps with one unit, because both store UTC instants.
- **Coercion.** The binder casts one operand to the dtype of the other with a type-owned cast
  hook. Then the derive of that dtype applies.
- **Rewrite.** The type rewrites the whole call into a storage expression, for example a rescale
  before a decimal comparison (section 4.4).

Open: whether Vortex coerces at all, or engines coerce before they send an expression. Rust and the
built-in `Decimal` reject such operations today. SQL engines coerce them.

### 3.6 Table for the types in the repository

| Type | `eq` | `ord` | Reason |
| --- | --- | --- | --- |
| `Timestamp` | Yes | Yes | One dtype fixes the unit and the time zone, so storage counts one unit from one epoch |
| `Date`, `Time` | Yes | Yes | |
| `Uuid` | Yes | Open | Byte order is the Postgres order |
| `Json` | No | No | Text is not canonical |
| `Vector` | Yes | No | No natural order |
| `FixedShapeTensor` | Yes | No | No natural order |
| WKB and native geometry types | No | No | Two encodings can hold one geometry |
| `ForeignExtDType` | No | No | I9 |

Vortex compares floats with a total order. Equality is bitwise, so `NaN` equals a `NaN` with the
same bits, and `-0.0` differs from `+0.0`
(`vortex-array/src/scalar_fn/fns/binary/compare/primitive.rs:31-34`, `nested.rs:258`). Probe R3
shows this for primitive and fixed-size list compares. A `Vector` derive inherits these storage
semantics on purpose.

## 4. Binder

### 4.1 Resolution order

The binder handles every call that has at least one operand whose dtype contains an extension
dtype. It tries these steps in order:

1. The function declares the call structural for these operands. Bind it without a change.
2. The function declares itself type-aware for every extension type in the operands. Bind it
   without a change. The function owner owns the semantics.
3. The function declares a derive, and the operands meet D1 and section 3.4. If every operand in
   a derived position is a top-level extension operand, rewrite each one to `ext_storage`. A
   scalar result with the extension dtype, such as the result of `min`, gets a checked
   `ext_wrap`. Otherwise, bind the call without a change. An example is `list_contains`, where the
   list holds the extension values. Its kernel then compares nested values by storage, which the
   derive check authorizes.
4. An extension type of an operand provides a rewrite for the call. Use it. The binder asks the
   types in argument order.
5. Otherwise, return an error that names the function, the extension ID, and the missing
   declaration.

For a cast, the binder tries these steps in order:

1. A nullability change of the same dtype.
2. `cast_to` of the source type.
3. `cast_from` of the target type.
4. An error.

### 4.2 Function declarations

```rust
/// How a function uses an operand whose dtype contains an extension dtype.
pub enum ExtensionUse {
    /// The function moves or selects values and never reads them.
    Structural,
    /// The function applies storage equality or order. The binder checks these flags (D1).
    Derive(Derives),
    /// The function implements the semantics of the matched extension type itself.
    TypeAware,
    /// The function has no semantics for extension values. The binder rewrites or rejects it.
    None,
}

pub trait ScalarFnVTable {
    /// Declares how this function uses the operand at `arg`, which contains `ext`.
    ///
    /// The default is `ExtensionUse::None`.
    fn extension_use(
        &self,
        options: &Self::Options,
        arg: usize,
        ext: &ExtDTypeRef,
    ) -> ExtensionUse {
        ExtensionUse::None
    }
}
```

`AggregateFnVTable` gets the same method without `arg`, because an aggregate has one input.

- Only the function owner declares, in the definition of the function. The core grants no
  type-aware declaration by default.
- A declaration can depend on the options. `Binary` declares `Derive` with `eq` or `ord` by
  operator. `cast` is structural only for a nullability change. `stat` depends on its aggregate
  (`vortex-array/src/scalar_fn/fns/stat.rs:112-114`).
- A `TypeAware` declaration matches an extension type by its vtable, for example
  `ext.is::<Timestamp>()`, never by ID string. A `ForeignExtDType` keeps the real ID
  (`vortex-array/src/dtype/extension/foreign.rs:28-56`). So a match by ID also covers a
  placeholder, which breaks I9.
- `replace_time_zone` declares `TypeAware` for `Timestamp`. Today its `return_dtype` accepts only
  timestamps (`vortex-array/src/scalar_fn/fns/replace_time_zone.rs:131-148`). The tensor and
  spatial functions declare their own types in the same way.
- The binder never infers a declaration from `return_dtype`. `Binary::return_dtype` accepts any
  extension operand in a comparison today (`vortex-array/src/scalar_fn/fns/binary/mod.rs:162-168`,
  EXT-Q1).
- A value that a function keeps in its options counts as an operand. `DynamicComparison` keeps its
  right operand in its options and calls `Binary` directly
  (`vortex-array/src/scalar_fn/fns/dynamic.rs:80-111`). Its declaration covers that value, and a
  derive rewrite converts the value to a storage scalar.

### 4.3 Nested extension values

An operand counts as an extension operand if its dtype contains an extension dtype at any depth.
Examples are `list_contains` on `List<ext>`
(`vortex-array/src/scalar_fn/fns/list_contains/mod.rs:178-194`) and the spatial `collect` on
`List<Point>` (`vortex-spatial/src/scalar_fn/collect.rs:60-75`).
`ext_storage` unwraps only the top level (`vortex-array/src/scalar_fn/fns/ext_storage.rs:58-63`).
So a derive rewrite happens only when every derived operand is a top-level extension operand
(section 4.1). Otherwise the call binds unchanged. The kernel then reads nested members by storage,
which the container rule of section 3.4 authorizes.

### 4.4 Proposed hooks on the extension type

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
    ///
    /// The call holds every argument, so a rewrite handles any arity and operands of other dtypes.
    fn rewrite(ext: &ExtDType<Self>, call: &ScalarFnCall) -> VortexResult<Option<Expression>> {
        Ok(None)
    }
}
```

`ScalarFnCall` is a new view of the function, its options and its bound arguments. Aggregates need
the same pair of hooks (`rewrite_aggregate`). Open: whether a session registry keyed by function
and extension type also exists. It lets a crate add support for a function and a type that it
does not own.

### 4.5 Examples

```text
Timestamp(ms) a  <  Timestamp(ms) b        =>  ext_storage(a) < ext_storage(b)
Timestamp(ms) a  <  lit(Timestamp(ms) 5)   =>  ext_storage(a) < lit(5i64)
Timestamp(ms) a  <  lit(5i64)              =>  rejected (D1)
min(Timestamp(ms) a)                       =>  ext_wrap(min(ext_storage(a)), Timestamp(ms), checked)
cast(Timestamp(ms) a, Timestamp(ns))       =>  ext_wrap(checked_mul(ext_storage(a), 1_000_000),
                                                        Timestamp(ns), unchecked)
cast(Timestamp(ns) a, Timestamp(ms))       =>  rejected unless the type defines rounding
DecimalLike(10, 1) a  <  DecimalLike(10, 2) b
                                           =>  checked_mul(ext_storage(a), 10) < ext_storage(b)
                                               (a rewrite that the type owns)
Json a  =  Json b                          =>  rejected
```

`min` uses a checked wrap, because its result is a scalar and every scalar is valid (section 8).
The unit cast uses an unchecked wrap. Its result stays unvalidated, so a read of a row still
checks the value.

### 4.6 Array API

The array API builds `ScalarFnArray` values directly. The binder must run in
`ScalarFnArray::try_new` (or in the typed constructors such as `Binary::try_new`). It returns the
rewritten tree, so the new array holds `ext_storage` children and an `ext_wrap` parent where
needed. The row path (`vortex-array/src/arrays/scalar_fn/vtable/operations.rs:30-42`) then
evaluates the rewritten tree.

Open: some rewrites can need session state. The hooks above use only the `ExtDType`, which travels
with the dtype, so the binder does not need a session.

### 4.7 Simplification of wraps

`ext_storage` is infallible today (`vortex-array/src/scalar_fn/fns/ext_storage.rs:87-93`). A
checked `ext_wrap` is fallible in the sense of `ScalarFnVTable::is_infallible`
(`vortex-array/src/scalar_fn/vtable.rs:262-279`). It fails when a row that it evaluates holds a
value that the type rejects. A simplification must not remove such a failure. It must also not add
a failure for a row that the original expression does not evaluate.

| Rule | Valid when | Reason |
| --- | --- | --- |
| `ext_storage(lit(s))` to `lit(s.to_storage_scalar())` | Always | Both read the storage of one valid scalar, and the dtypes match |
| `ext_storage(ext_wrap(y, d, unchecked))` to `y` | `y` has the storage dtype of `d`, ignoring nullability | Both steps are infallible and keep the values. The result dtype is the dtype of `y` |
| `ext_storage(ext_wrap(y, d, checked))` to `y` | Never | The wrap fails on a value that `d` rejects, and `y` alone does not fail |
| `ext_wrap(ext_storage(x), d, unchecked)` to `x` | `d` equals the dtype of `x`, ignoring nullability | Same values and dtype. The result keeps the validation state of `x`, which came from a real check |
| `ext_wrap(ext_storage(x), d, checked)` to `x` | Never at bind time | `x` can be unvalidated, and then the check can fail |
| `ext_wrap(lit(v), d, mode)` to a literal | `Scalar::try_new(d.with_nullability(n), v)` succeeds, where `n` is the nullability of `v` | A literal is a scalar, and every scalar is valid |

A counterexample for the third rule uses `NonZero` over `u64` and `y = lit(0u64)`. The original
expression fails, because `NonZero` rejects zero. The simplified expression returns `0`. Probe R4
shows that the check fails: `Scalar::try_new` with `NonZero` and storage `0` returns an error.
`ext_wrap` itself does not exist yet.

Binding checks a literal argument of `ext_wrap` in both modes (section 5). A literal can also
appear after binding. `Expression::bind` folds nothing
(`vortex-array/src/expr/bound_expression.rs:309-323`), but simplification turns `cast(lit)` into a
literal (`vortex-array/src/scalar_fn/fns/cast/mod.rs:170-181`). So
`ext_wrap(cast(lit(0i64), u64), NonZero, checked)` binds, and a later fold meets an invalid literal.
When the fold check fails, the wrap stays in the tree, as a failing cast does. Then a checked wrap
fails only for rows that it evaluates, and an unchecked wrap leaves the check to reads. The binder
emits an unchecked wrap only where the values are valid by construction.

## 5. `ext_wrap`

- Options: the target `ExtDTypeRef`, and a mode that is `checked` or `unchecked`.
- `return_dtype`: the target dtype, with the nullability of the input.
- Argument: must have the storage dtype of the target, ignoring nullability. Otherwise binding
  fails.
- Literal argument: binding checks it with `Scalar::try_new`, in both modes, and an invalid
  literal is a bind error. A literal that simplification makes later follows the fold rule of
  section 4.7.
- Checked execution: calls `validate_array` (section 8), then builds a validated wrapper. It is
  fallible.
- Unchecked execution: builds the wrapper without a check. The result is unvalidated. It is
  infallible.
- Row path: both modes build the scalar with `Scalar::try_new`. `Scalar::new_unchecked` checks
  values only in debug builds, and it panics there on an invalid value
  (`vortex-array/src/scalar/scalar_impl.rs:97-104`).

## 6. Physical form (open)

Opacity is a semantic rule. It does not decide where the wrapper sits relative to generic
encodings such as `Constant`, `Dict`, `Chunked`, `Masked` and `Sparse`. The binder works with
either placement, because `ext_storage` works with either. This section compares the two. No
maintainer decision exists.

### 6.1 Current code

- The compressor writes `Constant(extension scalar)` for a constant extension column, and
  `Masked(Constant(extension scalar))` when some rows are null
  (`vortex-compressor/src/compressor/cascade.rs:272-273`,
  `vortex-compressor/src/compressor/constant/mod.rs:175-193`). For other columns, it compares two
  results and keeps the smaller one (`cascade.rs:175-205`). The first comes from a scheme that
  matches the extension type. For example, `TemporalScheme` writes `DateTimeParts` for a
  `Timestamp` (`vortex-btrblocks/src/schemes/temporal.rs:45-56`, `:124`). The second is
  `vortex.ext` over the compressed storage. Probe R8 shows the constant, masked and `vortex.ext`
  shapes. Its compressor had no temporal scheme.
- `ExtensionConstantRule` rewrites `Extension(Constant(storage))` to `Constant(extension scalar)`
  (`vortex-array/src/arrays/extension/compute/rules.rs:28-46`). Probe R5 shows the rewrite.
- `ArrayRef::as_constant()` matches only the `Constant` encoding
  (`vortex-array/src/array/erased.rs:485-487`). Probe R5 shows that it misses
  `Extension(Constant)`.
- `ext_storage` reads a constant input directly. For other inputs, it executes the input to a
  canonical `ExtensionArray` (`vortex-array/src/scalar_fn/fns/ext_storage.rs:78-84`).
- Generic rules push infallible scalar functions into dictionary values and into chunks
  (`vortex-array/src/arrays/dict/compute/rules.rs:96-138`,
  `vortex-array/src/arrays/chunked/compute/rules.rs:35-36`). Probe R6 shows that
  `ext_storage` keeps both encodings. Over `Masked`, it decodes the input. Over `Sparse` with
  extension values, it fails, because sparse canonicalization rejects extension arrays
  (`encodings/sparse/src/canonical.rs:360-362`).
- Constant canonicalization and masked execution have extension arms
  (`vortex-array/src/arrays/constant/vtable/canonical.rs:204-219`,
  `vortex-array/src/arrays/masked/execute.rs:178-193`). So does dictionary execution
  (`vortex-array/src/arrays/dict/execute.rs:182-193`).
- `vortex.ext`, `vortex.constant`, `vortex.dict` and `vortex.chunked` are in core edition 2025.05
  (`vortex-edition/src/declarations/core/v2025_05.rs:27-33`). `vortex.masked` is in 2025.10
  (`v2025_10.rs:24`).

### 6.2 Comparison

| Aspect | Wrapper above generic encodings | Generic encodings above the wrapper |
| --- | --- | --- |
| Shapes of an extension column | One after reduction: `Extension(child)` | Several: `Extension(x)`, `Constant(ext s)`, `Masked(Constant(ext s))`, `Dict(codes, Extension(v))`, `Chunked[...]` |
| Constructors | Code that builds a `Constant`, `Dict` or `Chunked` with an extension dtype makes a form that reduce rules lift. A constructor can also build the lifted form directly. `ExtensionConstantRule` reverses | No change. `Constant` already sits on top |
| Constant detection | `as_constant()` misses `Extension(Constant)`, so callers need a helper that looks through the wrapper. The compressor writes `Extension(Constant(storage))` | `as_constant()` works as today |
| Storage extraction, array path | `ext_storage` reads the child | `ext_storage` keeps `Dict` and `Chunked` through existing rules. `Masked` decodes, and `Sparse` fails (R6) |
| Storage extraction, row path | One reduce rule, `ext_storage(Extension(s))` to `s`, makes the row path read storage. No such rule fired in probe R9 | The same rule, plus a pushdown for `Masked` and `Sparse`. `Dict` and `Chunked` already push `ext_storage` down (R6) |
| Extension arms in generic encodings | None. Sparse works over storage | `Constant`, `Masked` and `Dict` keep theirs. `Sparse` needs one |
| Statistics | Child statistics are storage statistics | `Constant(ext s)` holds an extension scalar. A reader converts it |
| Wire compatibility | Readers read every old form, and reduce rules normalize it. Old readers read `vortex.ext` over `vortex.constant`. `Decimal` needs a translation on write, because `vortex.decimal` is a leaf | The memory form and the wire form match for both types |

### 6.3 Recommendation

These notes recommend the wrapper above generic encodings. This is a recommendation, not a
decision. The reasons:

- One shape removes EXT-S1, and generic encodings lose their extension arms.
- The compressor already writes this shape for non-constant extension columns, unless a
  type-specific scheme such as `DateTimeParts` wins (R8).
- Extension columns need no new wire form, because old readers read `vortex.ext` over
  `vortex.constant`.

The costs:

- `ExtensionConstantRule` reverses. Constant detection in the compressor and in kernels must look
  through the wrapper.
- The change lands best after the binder. Before the binder, a kernel that reads a constant
  extension operand with `as_constant()` misses `Extension(Constant)`. The `DateTimeParts` compare
  kernel then declines (`encodings/datetime-parts/src/compute/compare.rs:31-33`). The extension
  compare kernel still compares the storage
  (`vortex-array/src/arrays/extension/compute/compare.rs:31-51`).
- The `Decimal` plugin translates the wrapper form to the legacy decimal forms on write
  (`../../decimal/agents/report.md`, section 3.5).

### 6.4 Type-specific encodings

Type-specific encodings are a separate question. `DateTimeParts` carries the `Timestamp` dtype
today (`encodings/datetime-parts/src/canonical.rs:28`, EXT-S2). Under the binder, a compare
reaches it as `ext_storage(column) < lit(i64)`. To keep its compare kernel, `DateTimeParts`
becomes an `i64` encoding with the time unit in its metadata. Readers convert the old form on
read. An edition gates writes of the new form. `vortex-btrblocks/src/schemes/temporal.rs` chooses
the encoding from the extension metadata, and compresses the storage child. The same applies to
any future encoding for one extension type.

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

## 8. Validation and trust

### 8.1 Three kinds of validation

| Kind | What it checks | When | Cost |
| --- | --- | --- | --- |
| Dtype validation | `validate_dtype`: the metadata and the storage dtype | When a dtype is built or read, for both nullabilities (section 9) | Once for each dtype |
| Storage-shape validation | The storage child has the length of the array, which `Extension::validate` checks today, and the storage dtype (U6). A scalar value has the storage dtype (EXT-V7) | When an array or a scalar is built or read | Constant |
| Value validation | `validate_array`: each checked value is a valid value of the type | At the trust boundaries in section 8.3 | One pass over the values |

### 8.2 Validation state

Each extension array is validated or unvalidated.

- An array is validated only after a value check succeeds for all its checked values, for example
  `validate_array`. The result of a structural operation is validated only if all its extension
  inputs are validated.
- In this design, a `Scalar` is always valid, because `Scalar::try_new` runs the value check.
  Today two gaps exist. A scalar can hold a value of the wrong storage type (EXT-V7, fixed by U6).
  `Scalar::zero_value` skips the check (EXT-V3, fixed by U7).
- A constant array checks its scalar when it is decoded
  (`vortex-array/src/arrays/constant/vtable/mod.rs:160-161`). So a constant extension array is
  validated.
- Every other array is unvalidated. This includes file reads without a value check, unchecked
  wraps, and constructors that skip the value check.
- No constructor marks an array validated without a check. A skipped check is never a proof of
  validity.

### 8.3 Where value validation runs

- Checked constructors and checked `ext_wrap`. The result is validated.
- Imports from Arrow, DuckDB, FFI and Python. The result is validated.
- `Scalar::try_new`, for every extension scalar, as today.
- File reads, when a read option asks for it. The result is validated. Open: the default of this
  option.
- Row reads from an unvalidated array check each read value, and return an error for an invalid
  value (U3). Row reads from a validated array skip the check.
- A consumer that needs every value to be valid, such as an export, validates an unvalidated array
  first.

`ext_storage` never checks values, on either path. Today the row path checks them. It reads each
child as an extension scalar (`vortex-array/src/arrays/scalar_fn/vtable/operations.rs:35-39`), and
that read runs the per-row check (`vortex-array/src/arrays/extension/vtable/operations.rs:19-30`).
The array path reads the storage without a check
(`vortex-array/src/scalar_fn/fns/ext_storage.rs:83-84`). Probe R9 shows the difference: the array
path returns `[0, 1]` for `NonZero [0, 1]`, and the row path panics. The row path must read the
storage child directly.

### 8.4 The array validator and the scalar validator

```rust
pub trait ExtVTable {
    /// Checks every value of `storage` that is valid in `storage` and in `ancestors`.
    ///
    /// The default applies `validate_scalar_value` to each of those values. An override must
    /// accept exactly the arrays that the default accepts.
    fn validate_array(
        ext: &ExtDType<Self>,
        storage: &ArrayRef,
        ancestors: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        validate_each_value::<Self>(ext, storage, ancestors, ctx)
    }
}
```

- `validate_scalar_value` stays the rule for one value. Its default calls `unpack_native`
  (`vortex-array/src/dtype/extension/vtable.rs:53-66`).
- The default `validate_array` is not a no-op. It applies `validate_scalar_value` to each value
  that is valid in the array and in every ancestor.
- The caller passes the ancestor mask. Builders pad the rows under a null parent with default
  values (`vortex-array/src/builders/fixed_size_list.rs:292-305`). `vortex-spatial` treats
  storage behind a null row as arbitrary (`vortex-spatial/src/scalar_fn/row.rs:77-78`). So a
  padding value never has to be valid.
- A type can override `validate_array` with a faster check. The conformance harness compares the
  override with the default on generated inputs.
- Examples of overrides: `Timestamp` checks the storage minimum and maximum against its valid
  range. `Uuid` checks the version nibble of each value against the metadata. A hypothetical
  `NormalizedVector` type checks the norm of each row within a tolerance.

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
- Builders pad a non-nullable child under a null parent
  (`vortex-array/src/builders/fixed_size_list.rs:303`, `vortex-array/src/builders/mod.rs:156-166`).
  Value validation skips those rows (section 8.4), so a type with no zero still works as such a
  child. The padding needs only a storage value.

## 11. Scalar identity and order

### 11.1 Current code

- `Scalar` derives `Eq` (`vortex-array/src/scalar/mod.rs:41`). `PartialEq` and `Hash` compare the
  dtype without nullability and the value (`vortex-array/src/scalar/scalar_impl.rs:336-356`).
- `PartialOrd` returns `None` for different dtypes. Otherwise it compares the stored values
  (`scalar_impl.rs:358-401`), so extension values compare by storage for every type (EXT-Q2).
  Probe R2 shows `Some(Less)` for two `Opaque` values.
- Floats use bitwise equality and `total_cmp` (`vortex-array/src/dtype/ptype.rs:328-336`).
- `scalar_cmp` computes every comparison operator from `partial_cmp`
  (`vortex-array/src/scalar_fn/fns/binary/compare/mod.rs:241-267`). Constant inputs and the row
  path use it.
- A `Variant` value has no order, so `partial_cmp` returns `None` even when `==` holds
  (`scalar_impl.rs:449-451`).
- An extension scalar can hold a value of the wrong storage type. Then `==` can panic (EXT-V7,
  probe R2).
- A valid `F16` scalar can hold a `U64` value, for compatibility with an old serialized form
  (`vortex-array/src/scalar/validate.rs:147-157`). Then `==` panics in one direction and returns
  `false` in the other (probe R10).

### 11.2 The Rust rule

`PartialOrd` must agree with `PartialEq`: `a == b` holds exactly when `partial_cmp(a, b)` returns
`Some(Equal)`. Suppose `PartialEq` stays structural and `PartialOrd` follows the derives. Then the
rule breaks. For an `Opaque` scalar `x` without `ord`, `x == x` is true, but `partial_cmp(x, x)`
returns `None`.

### 11.3 Contract

- `PartialEq`, `Eq` and `Hash` describe identity, as today: the same dtype without nullability,
  and equal stored values. Integer and decimal values of different widths compare by value
  (`vortex-array/src/scalar/typed_view/primitive/pvalue.rs:122-176`,
  `vortex-array/src/scalar/typed_view/decimal/dvalue.rs:255-281`).
- `PartialOrd` follows three steps:
  1. If `a == b`, return `Some(Equal)`.
  2. If both have one ordered dtype, ignoring nullability, return the order of the stored values.
     Nulls sort first.
  3. Otherwise, return `None`.
- A built-in dtype keeps its current order. An extension dtype is ordered only if it declares
  `ord`. A container is ordered only if all its members are ordered. The flags come from the
  non-nullable dtype.
- `Ord` is not implemented, as today.
- A valid `Eq` needs values whose variant matches the dtype. So U6 checks the storage dtype of
  extension scalars (EXT-V7). Scalar construction also converts the `U64` form of an `F16` value
  to `F16` (probe R10).

These steps keep the Rust rule. Step 1 returns `Some(Equal)` exactly when `==` holds. Under the
stored-value order, two values that are not equal never compare as `Equal`, because float
equality is bitwise and float order is `total_cmp`. The steps also fix the `Variant` case.

### 11.4 Relation to expression semantics

- Identity serves Rust code: maps of constants, deduplication, tests and caches. It never decides
  an expression result for extension values.
- Identity implies semantic equality, because one dtype and one storage value give one value.
  The reverse does not hold. Without `eq`, or with a rule across dtypes, two values that are not
  identical can be equal.
- Expression comparisons follow the binder. After binding, `scalar_cmp` sees storage scalars at
  the top level. It must compute `=` and `!=` with `==`, and the order operators with
  `partial_cmp`. Then `=` works on a container with an `eq`-only member, such as
  `struct{v: Vector}`.
- The nested comparator follows the container rule of section 3.4.

### 11.5 Order of work

Change `PartialOrd` only together with, or after, the phase 2 change that rejects `min` and `max`
for extension types without `ord`. Today `min` and `max` merge partial results with
`partial_min(..).vortex_expect` (`vortex-array/src/aggregate_fn/fns/min/mod.rs:59`,
`vortex-array/src/aggregate_fn/fns/min_max/mod.rs:247`, `:252`). The zone-map writer binds them
for every extension column (`vortex-layout/src/layouts/zoned/writer.rs:217-231`). With the new
order, a `Json` zone that merges two different maxima panics. The merge of several partial results
in one zone is an inference from the code, not a run.

## 12. Files to change, by phase

| Phase | Files |
| --- | --- |
| 0, U1 | `vortex-array/src/scalar/cast.rs`, `vortex-array/src/scalar/typed_view/extension/mod.rs`, `vortex-array/src/arrays/extension/compute/cast.rs`, `vortex-arrow/src/executor/{primitive,byte_view,struct_}.rs` |
| 0, U2 | `vortex-array/src/scalar_fn/fns/binary/mod.rs`, `vortex-array/src/scalar_fn/fns/binary/compare/mod.rs`, `encodings/datetime-parts/src/compute/compare.rs` |
| 0, U3 | `vortex-array/src/arrays/extension/vtable/operations.rs`, `vortex-array/src/arrays/extension/compute/rules.rs`, `vortex-array/src/scalar/constructor.rs` |
| 0, U4 | `vortex-array/src/dtype/extension/typed.rs`, `vortex-array/src/dtype/extension/vtable.rs` |
| 0, U5 | `vortex-array/src/extension/datetime/{date,time,timestamp}.rs` |
| 0, U6 | `vortex-array/src/arrays/extension/vtable/mod.rs`, `vortex-array/src/scalar/validate.rs`, `vortex-array/src/scalar/scalar_impl.rs` |
| 0, U7 | `vortex-array/src/scalar/scalar_impl.rs`, `vortex-array/src/scalar/scalar_value.rs` |
| 0, U8 | `vortex-array/src/dtype/extension/erased.rs` |
| 2 | `vortex-array/src/dtype/extension/{vtable,typed,erased}.rs`, `vortex-array/src/scalar_fn/vtable.rs`, `vortex-array/src/aggregate_fn/vtable.rs`, `vortex-array/src/expr/bound_expression.rs`, `vortex-array/src/arrays/scalar_fn/array.rs`, `vortex-array/src/scalar_fn/fns/{binary,between,dynamic.rs,ext_storage.rs}`, `vortex-array/src/aggregate_fn/fns/{min,max,min_max,bounded_min,bounded_max,is_sorted}/`, `vortex-array/src/arrays/extension/compute/compare.rs`, `vortex-array/src/scalar/scalar_impl.rs`, `vortex-array/src/scalar/typed_view/extension/mod.rs`, `vortex-layout/src/layouts/zoned/writer.rs`, `vortex-row/src/encoder.rs`, `vortex-duckdb/src/table_function.rs` |
| 3 | `vortex-array/src/scalar_fn/fns/cast/mod.rs`, `vortex-array/src/extension/datetime/`, `vortex-duckdb/src/convert/expr.rs`, `vortex-datafusion/src/convert/exprs/mod.rs` |
| 4 | The files depend on the chosen physical form (section 6). Both forms change `encodings/datetime-parts/`, `vortex-btrblocks/src/schemes/temporal.rs`, `vortex-layout/src/layouts/zoned/`, `vortex-layout/src/layouts/file_stats.rs`, `vortex-file/src/pruning.rs`, `vortex-array/src/stats/rewrite/` and `vortex-edition/`. The wrapper on top also changes `vortex-array/src/arrays/extension/compute/rules.rs`, `vortex-array/src/arrays/{constant,chunked,dict,masked}/` and `vortex-compressor/src/compressor/constant/`. Generic encodings on top also change `encodings/sparse/src/canonical.rs` |
| 5 | `vortex-array/src/arrays/extension/array.rs`, `vortex-array/src/arrays/extension/vtable/operations.rs`, `vortex-array/src/builders/extension.rs`, `vortex-array/src/dtype/extension/` |
| 6 | `vortex-arrow/src/session.rs`, `vortex-duckdb/src/convert/`, `vortex-duckdb/src/exporter/`, `vortex-datafusion/src/convert/`, `vortex-python/src/scalar/`, `docs/developer-guide/extending/extension-dtypes.md` |

## 13. Tests

- **Path consistency.** A helper in `vortex-array/src/compute/conformance/` takes an expression
  builder and inputs. It evaluates the expression on the array path, on the row path, and with
  constant-encoded inputs. It asserts equal values or an error on all three. The inputs include
  unvalidated arrays with invalid values.
- **Reference types.** `vortex-array/src/extension/tests/` already has `DivisibleInt`. Add:
  - `Opaque` over `i64`, with no derives. Every semantic operation fails when it is bound.
  - `DecimalLike` over `i64`, with `eq` and `ord` for each dtype and a rescale rewrite across
    scales. A differential test runs each operation on `DecimalLike` and on the built-in
    `Decimal`, and compares values or errors.
  - `Refined` over `i64` (for example, positive values only), with every derive and a
    `validate_array` override.
- **Conformance harness for `ExtVTable`.** It checks the metadata round trip, that
  `validate_dtype` ignores nullability, rules D1 to D4, cast round trips, and path consistency. D2
  and D3 need the optional reference order. The harness also compares a `validate_array` override
  with the default.
- **Scalar contract.** For generated scalars, `a == b` holds exactly when `partial_cmp(a, b)`
  returns `Some(Equal)`, and equal scalars have equal hashes.
- **Engines.** DuckDB and DataFusion pushdown tests with extension literals for `Timestamp`, `Date`,
  `Uuid` and `Json` columns. A `Json` comparison must not be pushed down.
