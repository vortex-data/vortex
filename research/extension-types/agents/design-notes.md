<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Extension type design notes (agent reference)

This file is the detailed design behind `../03-design.md` and `../05-implementation.md`. It
follows `../../decisions.md`, which records the decisions of the maintainer. IDs such as X4 and XQ1
refer to that file. "Open" marks a decision that is not made. Finding IDs (`EXT-*`) refer to
`findings.md`, which covers current code only. Probe IDs (R1 to R10) refer to appendix A of
`findings.md`. Paths and line numbers are for `develop` at `5fc5cb8`.

## 1. Invariants

The design holds these invariants. Every change in the plan must keep them.

- **I1.** A semantic operation on an operand that contains an extension dtype is rejected by
  default. A declared derive, a rule that the type owns, or a declaration of the function can allow
  it (sections 3 and 4).
- **I2.** A structural operation on an extension value is always allowed and keeps the extension
  dtype.
- **I3.** Two kinds of rules decide how a generic operation treats extension values. They are the
  derive rules of the core and the rules that each extension type owns (section 4). No other code
  applies storage semantics to an extension value. Every node meets a typing check when code
  builds it (section 4.1). The rules run on every path that optimizes or executes arrays and bound
  expressions (section 4.5).
- **I4.** After the rules run, a generic operation applies storage semantics to an extension value
  only where a derive covers it (sections 3.4 and 4.3). The kernels of generic semantic
  operations, statistics and pruning see storage dtypes only, at every depth. Generic structural
  encodings sit above the extension wrapper, and storage encodings sit below it (section 6). A
  type-aware function sees the extension types that it declares. `DateTimeParts` stays an
  exception until phase 4b (EXT-S2).
- **I5.** The only ways to move between an extension dtype and another dtype are `ext_storage`,
  `ext_storage_deep` and the rules that the extension type owns. A rule that produces extension
  values builds the extension array itself (section 5). No `ext_wrap` expression exists (X4).
- **I6.** The row path and the array path apply the same rules, so they return the same result
  (EXT-P1). Issue [#10313](https://github.com/vortex-data/vortex/issues/10313) owns the separate
  question of whether a validity query raises errors from values that it skips (X9).
- **I7.** Reading a value never panics. Building an extension scalar from an array checks the value
  and returns an error for an illegal value (section 8).
- **I8.** A nullability change of a dtype never fails and never validates the type again.
- **I9.** An unknown extension type (`ForeignExtDType`) declares no derives and owns no rules, and
  no `TypeAware` declaration matches it.

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
| `ext_storage`, `ext_storage_deep` | Structural | Explicit unwrap of the top level, or of every level |
| `=`, `!=` | Semantic | Needs `eq` or a rule of the type |
| `<`, `<=`, `>`, `>=`, `between` | Semantic | Needs `ord` or a rule of the type |
| `min`, `max`, `min_max`, `bounded_min`, `bounded_max`, `is_sorted`, sort keys | Semantic | Needs `ord` or a rule of the type |
| `is_constant`, `all_non_distinct`, `list_contains`, distinct, group by, join keys | Semantic | Needs `eq` or a rule of the type |
| Arithmetic, `sum`, `sum_v2`, `mean` | Semantic | Needs a rule of the type, because the result type usually changes |
| `cast` other than a nullability change | Semantic | Needs a cast rule of the type (section 4.4) |
| `is_nan`, `nan_count`, `like`, `byte_length` | Semantic | Needs a rule of the type |
| Zero and default values | Semantic | Needs a `zero_value` hook |
| Export to Arrow, DuckDB, DataFusion, Python, JNI | Semantic | Needs an export plugin |
| `replace_time_zone`, tensor functions, spatial functions | Type-aware | The function owner declares the types (section 4.2) |

## 3. Derives

### 3.1 Definition

A derive is a declaration on one extension dtype (X2). It says that a semantic operation on values
of that dtype is the same operation on their storage values.

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

`ExtDTypeRef` gets an erased forwarder: `fn derives(&self) -> Derives`. Typing and the derive rules
read the flags from the non-nullable form of the dtype. So a type that breaks DR4 still gives one
answer.

### 3.2 Rules

- **DR1.** A derive applies only when every operand in a derived position has the same extension
  dtype, ignoring nullability. A lone extension operand never gets a derive, so `ts < 5i64` stays
  rejected. Operations across dtypes are rejected (section 3.5).
- **DR2.** `eq` requires canonical storage: each value of the dtype has exactly one storage form.
- **DR3.** `ord` requires `eq`, and storage order must equal value order.
- **DR4.** The flags do not depend on nullability.

The conformance harness checks DR4, and that `ord` implies `eq`. It checks DR2 and DR3 only when a
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
declaration or a rule that the type owns (section 4).

### 3.4 Containers

A struct, list, fixed-size list or map dtype that holds extension values has `eq` or `ord` only if
every nested extension dtype declares it. An extension dtype whose storage holds another extension
dtype follows the same rule. An order comparison of two `struct{ts: Timestamp(ms)}` values needs
`ord` on `Timestamp(ms)`. Any comparison of two `struct{doc: Json}` values is rejected.

After the check, the derive rule removes every extension layer at every depth (section 4.3, X6).
The kernel then sees only storage dtypes. So the extension branch of the nested comparator
(`vortex-array/src/scalar_fn/fns/binary/compare/nested.rs:229-233`) goes away in phase 2.

### 3.5 Operations across dtypes

Derives never apply across dtypes. Two dtypes with different metadata are different dtypes, for
example `Timestamp(ms)` and `Timestamp(ns)`, or two time zones. Vortex rejects such an operation
(X3). An engine converts the operands to one dtype first, as its planner already does for SQL. A
user of the Rust or Python API writes an explicit cast. Rust and the built-in `Decimal` reject such
operations today as well.

### 3.6 Table for the types in the repository

| Type | `eq` | `ord` | Reason |
| --- | --- | --- | --- |
| `Timestamp` | Yes | Yes | One dtype fixes the unit and the time zone, so storage counts one unit from one epoch |
| `Date`, `Time` | Yes | Yes | |
| `Uuid` | Yes | Open | Byte order is the Postgres order (XQ6) |
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

## 4. Typing and rewrite rules

A call with an extension operand passes two steps. Typing checks the call when code builds the
node. Rewrite rules then replace an accepted semantic call with operations on storage. Typing never
rewrites, and the rules never accept a call that typing rejected.

### 4.1 Typing

Every node of a compute graph meets a typing check when code builds it. Today that check is
`return_dtype`, and it runs in four places:

- `BoundExpression::try_new_vec` (`vortex-array/src/expr/bound_expression.rs:129-148`).
  `Expression::bind`, `with_children` in the optimizer and the reduce rules of bound expressions
  all reach it.
- `ScalarFnArray::try_new_from_slots` (`vortex-array/src/arrays/scalar_fn/array.rs:102-114`).
  `try_new`, `try_new_with_len`, `ArrayRef::apply_bound` and the reduce rules of arrays all reach
  it.
- `AggregateDTypes::try_new` (`vortex-array/src/aggregate_fn/vtable.rs:55-75`), where an aggregate
  meets an input dtype. `Accumulator::try_new` reaches it, and DuckDB aggregate pushdown calls
  `Accumulator::try_new` directly (`vortex-duckdb/src/convert/expr.rs:543-552`).
  `Accumulator::from_dtypes` takes dtypes that are already resolved, so it accepts only checked
  dtypes. `return_dtype` and `state_dtype` of `AggregateFnRef` answer dtype questions without an
  accumulator (`vortex-array/src/aggregate_fn/erased.rs:86-95`), so they apply the same check.
- `ScalarFn::validate` (`vortex-array/src/arrays/scalar_fn/vtable/mod.rs:79-121`) checks a scalar
  function array that code builds from parts.

Two changes on `develop` since `5fc5cb8` affect these places. Since #10502, the first two call
`ScalarFnVTable::bind_options` before `return_dtype` (`bound_expression.rs:139` at `1d09649bbd`),
and `ScalarFn::validate` calls only `return_dtype`. Also, `Expression::Variable` and `Lambda`
(#10502) and `list_transform` (#10503) give a function a body that runs on list elements. Typing
must check that body too, because an element can be an extension value.

Typing accepts a call with an extension operand only in these cases, tried in this order:

1. The function declares the call structural for these operands.
2. The function declares itself type-aware for every extension type in the operands.
3. The function declares a derive, and the operands meet DR1 and section 3.4.
4. A rule of an extension type in the operands covers the call. Typing asks the types in argument
   order.

Otherwise typing returns an error that names the function, the extension ID and the missing
declaration. For a cast, typing tries three cases in order. The first is a nullability change of
the same dtype. The second is a rule of the source type, and the third is a rule of the target type.

### 4.2 Function declarations

```rust
/// How a function uses an operand whose dtype contains an extension dtype.
pub enum ExtensionUse {
    /// The function moves or selects values and never reads them.
    Structural,
    /// The function applies storage equality or order. Typing checks these flags (DR1).
    Derive(Derives),
    /// The function implements the semantics of the matched extension type itself.
    TypeAware,
    /// The function has no semantics for extension values. A rule of the type covers the call,
    /// or typing rejects it.
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

An aggregate with a `Derive` declaration also declares how its result relates to its input (XQ5,
open). These notes recommend three kinds:

- A value of the input is one of its stored values. The derive rule builds an extension scalar of
  the input dtype from the storage result. Examples are the results of `min` and `max`, and each
  field of the struct that `min_max` returns.
- A bound on the input stays a storage value (S5). `bounded_min` and `bounded_max` declare a bound
  when they truncate the storage, which happens for `Utf8` and `Binary`. For other storage dtypes,
  they declare a value.
- Any other result stays as it is, for example the `Bool` of `is_sorted`.

`ExtensionStoragePushdown` reads the same declaration to wrap its results (X12). U9 adds a narrow
form of it first, for `min_max`.

A value result can only get its top-level layer back. So if a value result holds extension values
inside a container, for example `min` over `List<Timestamp(ms)>`, typing rejects the call.

More rules for declarations:

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
- Typing never infers a declaration from `return_dtype`. `Binary::return_dtype` accepts any
  extension operand in a comparison today (`vortex-array/src/scalar_fn/fns/binary/mod.rs:162-168`,
  EXT-Q1).
- A value that a function keeps in its options counts as an operand. `DynamicComparison` keeps its
  right operand in its options and calls `Binary` directly
  (`vortex-array/src/scalar_fn/fns/dynamic.rs:80-111`). Its declaration covers that value. The
  value is a function that returns a new `ScalarValue` at each execution (`dynamic.rs:181-186`).
  So the derive rule changes the dtype of the value to the storage dtype, and the values stay as
  they are. After that change, nothing checks a new value against the extension type. So the rule
  keeps one check: each new value goes through `validate_scalar_value`, and an illegal value is an
  error. Today `Rhs::scalar` panics on such a value (`dynamic.rs:189-193`).

### 4.3 Nested extension values

An operand counts as an extension operand if its dtype contains an extension dtype at any depth.
Examples are `list_contains` on `List<ext>`
(`vortex-array/src/scalar_fn/fns/list_contains/mod.rs:178-194`) and the spatial `collect` on
`List<Point>` (`vortex-spatial/src/scalar_fn/collect.rs:60-96`). `ext_storage` unwraps only the
top level, and it rejects an operand whose top-level dtype is not an extension
(`vortex-array/src/scalar_fn/fns/ext_storage.rs:58-63`).

So a derive rule uses a new structural expression, `ext_storage_deep` (X6). It replaces every
extension dtype in the operand with its storage dtype, at every depth. It repeats this while a
storage dtype holds another extension dtype. Typing has already checked section 3.4, so every layer
that it removes declares the flag.

- For an array, it rebuilds each container around the storage of its children. It copies no
  values.
- For a scalar, it changes only the dtype, because an extension scalar already holds its storage
  value.
- For a top-level extension whose storage holds no extension dtype, it equals `ext_storage`. The
  derive rule then emits `ext_storage` itself. So the statistics rule S2, the simplification rules
  of section 4.8 and the pushdown rules of section 6 apply unchanged to the common case. Those
  rules gain a matching case for `ext_storage_deep`, which appears only for nested extension
  values.

For example, `list_contains` over `List<Timestamp(ms)>` and a `Timestamp(ms)` needle becomes
`list_contains` over `List<i64>` and an `i64` needle. Its kernel sees only storage. The spatial
`collect` declares `TypeAware` for `Point`, so no layer is removed there.

### 4.4 Rules that the type owns

A type implements its own semantics as rewrite rules (X4). The rules are registered with the type,
together with its compression scheme and the other parts of the type. A rule matches a call that
has an operand of the type. It replaces the call with operations on storage, or it declines.

- **Cast rules.** A rule converts the type to another dtype, or another dtype into the type. Cast
  rules implement the design of issue [#8570](https://github.com/vortex-data/vortex/issues/8570),
  with rules in place of closures. A cast exists only if a rule covers it (C1).
- **Call rules.** A rule covers a scalar function or an aggregate with an operand of the type. An
  example is `sum` over a decimal-like type, which needs a wider result dtype.

A rule reaches storage through `ext_storage` or `ext_storage_deep`. When its result holds values of
an extension type, it builds the extension array or scalar itself (section 5).

Where the rules live is open (XQ2). These notes recommend a method on `ExtVTable` that returns the
rules of the type. `ExtDTypeRef` holds the vtable, so the rules travel with the dtype and need no
session lookup. Registering the type in a session then registers its rules too.

The existing session registry, `ArrayKernels` (`vortex-array/src/optimizer/kernels.rs`), is the
alternative. It has two problems for extension types:

- It is keyed by encoding ids, and every extension array has the id `vortex.ext`. So each rule
  checks the dtype itself, and the rules of every registered type run on every extension column.
- Only the session-aware optimizer consults it. A grep counts 83 plain `.optimize()` calls outside
  tests, and they use static rules only. Two of them are `ArrayRef::cast`
  (`vortex-array/src/builtins.rs:176`) and `ArrayRef::apply_bound`
  (`vortex-array/src/expression.rs:41`).

Rules written over `ReduceNode` serve both arrays and bound expressions, as `reduce_null` already
does (`vortex-array/src/scalar_fn/fns/is_not_null.rs:50`). Pruning works on bound expressions
(`vortex-file/src/pruning.rs:35-49`). So a rule that serves only arrays cannot rewrite a predicate
onto storage statistics. Since #10790, `ReduceNode::as_constant` returns a borrowed `Scalar`, and a
rule makes a constant with `ReduceNode::new_constant`. A `ScalarFnArray` cannot wrap a `Literal`
(`vortex-array/src/arrays/scalar_fn/array.rs:161-167` at `1d09649bbd`), so an array rule that
returns a literal returns a `ConstantArray` instead.

Open: whether a crate can register a rule for a function and a type that it does not own.

### 4.5 Where the rules run

- The derive rules belong to the functions that declare a derive, such as `Binary`, `Between` and
  the order aggregates. So the static optimizer runs them.
- Under the recommendation of XQ2, the type rules come from the dtype. So the static optimizer
  finds them as well.
- The executor applies the same rules before it dispatches a kernel. A lazy array that nobody
  optimized still gets the rewrite, so the row path and the array path agree (I6).
- After phase 2, no generic kernel applies storage semantics. So a semantic call that reaches a
  generic kernel without a rewrite fails. A missing rule gives an error, never a wrong result.

### 4.6 Examples

```text
Timestamp(ms) a  <  Timestamp(ms) b       =>  ext_storage(a) < ext_storage(b)
Timestamp(ms) a  <  lit(Timestamp(ms) 5)  =>  ext_storage(a) < lit(5i64)
Timestamp(ms) a  <  lit(5i64)             =>  rejected by typing (DR1)
Timestamp(ms) a  <  Timestamp(ns) b       =>  rejected by typing (X3)
min(Timestamp(ms) a)                      =>  min(ext_storage(a)), and the rule builds a
                                              Timestamp(ms) scalar from the result
cast(Timestamp(ms) a, Timestamp(ns))      =>  Extension(Timestamp(ns),
                                                        checked_mul(ext_storage(a), 1_000_000))
cast(Timestamp(ns) a, Timestamp(ms))      =>  rejected unless the type defines rounding
Json a  =  Json b                         =>  rejected by typing
```

The cast rule builds its result with `ExtensionArray::new_unchecked`. Legal input gives legal
output, and `checked_mul` fails on overflow. The result is a wrapper over a pending computation,
which section 6 allows.

### 4.7 Array API and enforcement

Typing runs in `try_new_from_slots` (`vortex-array/src/arrays/scalar_fn/array.rs:102-114`), which
`try_new` and `try_new_with_len` both call. `ArrayRef::apply_bound` uses `try_new_with_len`
(`vortex-array/src/expression.rs:19-42`). Typing never changes the node, so a typed constructor
keeps its return type.

The types make the typing entry points of section 4.1 the only paths. The design accepts the API
changes that this needs (X8):

- `BoundExpression` keeps its variants private. Today they are public, so any code can build a
  node with any dtype (`vortex-array/src/expr/bound_expression.rs:42-59`).
- A constructor for a call that typing can reject becomes fallible. `Cast::new` is the main case.
  It calls `vortex_expect` on `try_new` (`vortex-array/src/scalar_fn/fns/cast/mod.rs:62-65`), and
  typing rejects every cast that no rule covers. So `Cast::new` becomes fallible in phase 2. A
  constructor for a structural call, such as `IsNull::new`
  (`vortex-array/src/scalar_fn/fns/is_null.rs:35-37`), can stay infallible, because typing never
  rejects it.
- Code that runs a function on raw arguments applies typing and the rules first.
  `DynamicComparison::execute` calls `Binary` directly (section 4.2). The row encoder of sort keys
  runs on raw columns, and today it rejects extension dtypes at every depth
  (`vortex-row/src/encoder.rs:112`, `:126-144`). After the derive rule, it receives the output of
  `ext_storage_deep`.

### 4.8 Simplification

`ext_storage` is infallible today (`vortex-array/src/scalar_fn/fns/ext_storage.rs:87-93`), so these
rules never remove or add a failure:

| Rule | Valid when | Reason |
| --- | --- | --- |
| `ext_storage(lit(s))` to `lit(s.to_storage_scalar())` | Always | Both read the storage of one valid scalar, and the dtypes match |
| `ext_storage_deep(lit(s))` to a literal of the storage dtype | Always | The same as the first rule, at every depth |
| `ext_storage(Extension(s))` to `s` | Always | The wrapper holds `s` as its storage. The row path then reads storage directly (section 8.3). No such rule exists today (probe R9) |
| `ext_storage(Constant(extension scalar))` to `Constant(storage scalar)` | Always | The same as the first rule, for a constant array |

`Expression::bind` folds nothing (`vortex-array/src/expr/bound_expression.rs:309-323`), but
simplification turns `cast(lit)` into a literal
(`vortex-array/src/scalar_fn/fns/cast/mod.rs:170-181`). When such a fold fails, the cast stays in
the tree. This behavior needs no change.

## 5. Wrapping storage

No `ext_wrap` expression exists (X4). A rule that produces extension values builds the extension
array or scalar itself.

- A rule uses `ExtensionArray::new_unchecked` when legal input gives legal output. Examples are the
  unit cast of section 4.6 and the result of `min`, which is a stored value.
- A rule uses `ExtensionArray::try_new` for storage values that no check has seen.
- A rule builds an extension scalar with `Scalar::try_new`, which always checks (section 8).
- Wrapping is a Rust call, not an expression. So no user expression and no serialized plan can
  build a wrapper without a check.
- A lazy computation that produces extension values is a wrapper over a pending computation, for
  example the result of the cast in section 4.6. Section 6 keeps pending computations under the
  wrapper.

## 6. Physical form

### 6.1 Decision

A rewrite rule pulls generic structural arrays out from under the extension wrapper (X5). Code
that looks for a constant, a dictionary or runs then finds them at the top, without any knowledge
of extension types. The compressor, the exporters, pruning and `as_constant()` all depend on this.

Which arrays the rule pulls out is open (XQ1). These notes recommend three roles:

| Role | Encodings | Place |
| --- | --- | --- |
| Structural | `Constant`, `Dict`, `RunEnd`, `Sparse`, `Chunked`, `Masked`, `Interleave` | Above the wrapper |
| Storage | The canonical storage arrays. The FastLanes encodings (BitPacked, FoR, Delta, RLE), ALP, ALP-RD, ByteBool, ZigZag, Pco, Zstd, FSST, OnPair, Sequence, PiecewiseSequence and Patched. Also DecimalByteParts, and `DateTimeParts` after phase 4b | Below the wrapper |
| Pending | Slices, filters and scalar functions that wait to run | Below the wrapper, with the storage that they apply to |

A storage encoding computes values from a layout that only built-in dtypes have, so it can never
carry an extension dtype. Patched, for example, accepts only primitive dtypes
(`vortex-array/src/arrays/patched/array.rs:187`). A structural encoding holds values of the column
dtype, whatever it is.

These notes also recommend that a canonical extension array is a wrapper over a canonical storage
array. Then a canonical array never hides a pending computation or a compressed child. Today
`Extension::execute` returns the array unchanged
(`vortex-array/src/arrays/extension/vtable/mod.rs:189-191`), so a canonical extension array can
hold any child.

### 6.2 Current code

- The compressor writes `Constant(extension scalar)` for a constant extension column, and
  `Masked(Constant(extension scalar))` when some rows are null
  (`vortex-compressor/src/compressor/cascade.rs:272-273`,
  `vortex-compressor/src/compressor/constant/mod.rs:175-193`). For other columns, it compares two
  results and keeps the smaller one (`cascade.rs:175-205`). The first comes from a scheme that
  matches the extension type. For example, `TemporalScheme` writes `DateTimeParts` for a
  `Timestamp` (`vortex-btrblocks/src/schemes/temporal.rs:45-56`, `:124`). When no scheme matches,
  the first result is the input itself (`cascade.rs:288-290`). The second is `vortex.ext` over the
  compressed storage, and it wins a tie (`cascade.rs:201`). Probe R8 shows the constant, masked
  and `vortex.ext` shapes. Its compressor had no temporal scheme.
- `ExtensionConstantRule` rewrites `Extension(Constant(storage))` to `Constant(extension scalar)`
  (`vortex-array/src/arrays/extension/compute/rules.rs:28-46`). Probe R5 shows the rewrite. This
  rule is the first case of the decision in section 6.1.
- `ArrayRef::as_constant()` matches only the `Constant` encoding
  (`vortex-array/src/array/erased.rs:485-487`). Probe R5 shows that it misses
  `Extension(Constant)`.
- `ArrayRef::take` wraps its input in a lazy `DictArray`
  (`vortex-array/src/array/erased.rs:262-264`). The extension take rule applies `take` to the
  storage, so it builds `Extension(Dict(indices, storage))`
  (`vortex-array/src/arrays/extension/compute/take.rs`). Under the decision, `take` of an extension
  array gives `Dict(indices, Extension(storage))`, so this rule goes.
- The extension mask rule builds a lazy mask expression over the storage
  (`vortex-array/src/arrays/extension/compute/mask.rs`). That is a pending computation, so it stays
  under the wrapper.
- `ext_storage` reads a constant input directly. For other inputs, it executes the input to a
  canonical `ExtensionArray` (`vortex-array/src/scalar_fn/fns/ext_storage.rs:78-84`).
- Generic rules push scalar functions into dictionary values and into chunks. The dictionary rule
  pushes an infallible function, or any function when every value is referenced
  (`vortex-array/src/arrays/dict/compute/rules.rs:96-208`). The chunked rules push a unary
  function, or a function with constant siblings
  (`vortex-array/src/arrays/chunked/compute/rules.rs:34-115`). Probe R6 shows that
  `ext_storage` keeps both encodings. Over `Masked`, it decodes the input. Over `Sparse` with
  extension values, it fails, because sparse canonicalization rejects extension arrays
  (`encodings/sparse/src/canonical.rs:360-362`).
- Constant canonicalization and masked execution have extension arms
  (`vortex-array/src/arrays/constant/vtable/canonical.rs:204-219`,
  `vortex-array/src/arrays/masked/execute.rs:178-193`). So does dictionary execution
  (`vortex-array/src/arrays/dict/execute.rs:182-193`). Run-end decoding rejects extension values
  (`encodings/runend/src/array.rs:509`).
- `vortex.ext`, `vortex.constant`, `vortex.dict` and `vortex.chunked` are in core edition 2025.05
  (`vortex-edition/src/declarations/core/v2025_05.rs:27-33`). `vortex.masked` is in 2025.10
  (`v2025_10.rs:24`).

### 6.3 Mechanism and costs

- One static rule on the `Extension` encoding moves a structural child above the wrapper. For
  example, `Extension(Dict(codes, values))` becomes `Dict(codes, Extension(values))`. A static rule
  runs in every `.optimize()` call, unlike a session rule (section 4.4).
- Each structural encoding implements one method that rebuilds it with its value children and value
  scalars passed through a function. That method gives three things: the lift rule, the pushdown
  of `ext_storage`, and decoding of a structural encoding with an extension dtype. Decoding strips
  the wrapper, decodes the storage form, and wraps the result again.
- So the extension code that `Constant`, `Masked` and `Dict` have today becomes one shared path.
  `RunEnd` and `Sparse`, which fail on extension values today, get support through the same path.
- `ext_storage` then never computes. It passes through the structural encodings and removes the
  wrapper.
- Readers apply the lift rule, so old files with `vortex.ext` over `vortex.dict` change shape when
  they are read. Old readers decode `Constant`, `Dict`, `Masked` and `Chunked` with extension
  values, but not `RunEnd` or `Sparse`. So writers keep the old shape for those two until an
  edition allows the new one.
- `Constant(extension scalar)` holds an extension scalar. Readers of statistics convert it to a
  storage scalar (S3).

### 6.4 Type-specific encodings

Type-specific encodings are a separate question. `DateTimeParts` carries the `Timestamp` dtype
today (`encodings/datetime-parts/src/canonical.rs:28`, EXT-S2). After the derive rule, a compare
reaches it as `ext_storage(column) < lit(i64)`. To keep its compare kernel, `DateTimeParts`
becomes an `i64` encoding with the time unit in its metadata. Readers convert the old form on
read. An edition gates writes of the new form. `vortex-btrblocks/src/schemes/temporal.rs` chooses
the encoding from the extension metadata, and compresses the storage child. The same applies to
any future encoding for one extension type.

## 7. Statistics

- **S1.** Zone maps and file statistics for an extension column store statistics of the storage,
  with the storage dtype. The storage fallback in `aggregate_state_dtype`
  (`vortex-layout/src/layouts/zoned/schema.rs:177-188`) becomes the rule. Today it applies only
  when an aggregate rejects the extension dtype.
- **S2.** A new stats rewrite rule maps `stat(ext_storage(x), s)` to the storage statistic `s` of
  `x`. After the derive rules, bound predicates reference `ext_storage(col)`, so pruning needs only
  this rule.
- **S3.** Readers convert existing extension-typed min and max to storage scalars with
  `to_storage_scalar()`. The values are already storage values.
- **S4.** Engine statistics export (DuckDB `column_statistics.rs`, DataFusion `convert/stats`, JNI
  write summaries) goes through the export plugin of the type. A type without `ord` exports no min
  and max.
- **S5.** Bounded min and max (truncation) apply to the storage dtype, so they also apply to
  extensions over `Utf8` and `Binary`.

## 8. Value validation

### 8.1 Three kinds of validation

| Kind | What it checks | When | Cost |
| --- | --- | --- | --- |
| Dtype validation | `validate_dtype`: the metadata and the storage dtype | When a dtype is built or read, for both nullabilities (section 9) | Once for each dtype |
| Storage-shape validation | The storage child has the length of the array, which `Extension::validate` checks today, and the storage dtype (U6). A scalar value has the storage dtype (EXT-V7) | When an array or a scalar is built or read | Constant |
| Value validation | `validate_array`: each non-null value is a legal value of the type | When `ExtensionArray::try_new` builds an array, and at the other places in section 8.3 | One pass over the values |

### 8.2 The invariant

Every extension array that Vortex builds holds only legal values in its non-null rows (X7). A
legal value is a value that `validate_scalar_value` accepts. Data from a file holds legal values
only if the read checks them (XQ3).

- `ExtensionArray::try_new` checks every non-null value with `validate_array`, and it returns an
  error for an illegal value. It takes an `ExecutionCtx`, because the check can execute compressed
  storage. `ExtensionArray::new` calls `try_new`. Today `try_new` checks only the storage dtype
  (`vortex-array/src/arrays/extension/array.rs:52-73`, EXT-V1).
- `ExtensionArray::new_unchecked` skips the check. Its caller promises that every non-null value
  is legal, as with other `new_unchecked` constructors. A debug build checks the promise.
- Kernels that keep values, such as slice, take, filter and mask, use `new_unchecked`. So do rules
  that make legal values from legal input, such as a unit cast of `Timestamp`. So does code that
  wraps storage it already holds again, such as canonicalization and the compressor
  (`vortex-compressor/src/compressor/cascade.rs:198`). A grep counts about 100 call sites of
  `ExtensionArray::new` and `try_new` outside test files. Each one moves to the right constructor.
- A `Scalar` is always legal, because `Scalar::try_new` runs the value check. Today two gaps
  exist. A scalar can hold a value of the wrong storage type (EXT-V7, fixed by U6).
  `Scalar::zero_value` skips the check (EXT-V3, fixed by U7).
- Building a scalar from an array checks the value with `Scalar::try_new`, because data from a file
  can hold illegal values (XQ3). The check returns an error, never a panic (I7).

A value check in every call of `ExtensionArray::new` slows common paths. In this design, kernels
use `new_unchecked`, so only the places in section 8.3 pay for the check. Type overrides of
`validate_array` make that check fast.

### 8.3 Where value validation runs

- `ExtensionArray::try_new`, for data that no check has seen.
- Rules that wrap storage values that no check has seen (section 5).
- Imports from Arrow, DuckDB, FFI and Python. They call `try_new`.
- `Scalar::try_new`, for every extension scalar, including a read of one row.
- File reads, if the answer to XQ3 asks for it.

`ext_storage` never checks values, on either path. Today the row path checks them. It reads each
child as an extension scalar (`vortex-array/src/arrays/scalar_fn/vtable/operations.rs:35-39`), and
that read runs the per-row check (`vortex-array/src/arrays/extension/vtable/operations.rs:19-30`).
The array path reads the storage without a check
(`vortex-array/src/scalar_fn/fns/ext_storage.rs:83-84`). Probe R9 shows the difference: the array
path returns `[0, 1]` for `NonZero [0, 1]`, and the row path panics. Under X7,
`ExtensionArray::new` rejects that array, but a file can still hold it. So the row path of
`ext_storage` reads the storage child directly, through the rule `ext_storage(Extension(s))` to
`s` (section 4.8).

### 8.4 The array validator and the scalar validator

```rust
pub trait ExtVTable {
    /// Checks every non-null value of `storage`.
    ///
    /// The default applies `validate_scalar_value` to each of those values. An override must
    /// accept exactly the arrays that the default accepts.
    fn validate_array(
        ext: &ExtDType<Self>,
        storage: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        validate_each_value::<Self>(ext, storage, ctx)
    }
}
```

- `validate_scalar_value` stays the rule for one value. Its default calls `unpack_native`
  (`vortex-array/src/dtype/extension/vtable.rs:53-66`).
- The default `validate_array` is not a no-op. It applies `validate_scalar_value` to each non-null
  value.
- A check returns an error for an illegal value and never panics. Today the `Timestamp` check
  panics on values outside the range of `jiff::Span`, because it calls the panicking setters
  (`vortex-array/src/extension/datetime/timestamp.rs:214-229`, EXT-V8). U5 fixes it.
- A type can override `validate_array` with a faster check. The conformance harness compares the
  override with the default on generated inputs.
- Examples of overrides: `Timestamp` checks the storage minimum and maximum against its valid
  range. `Uuid` checks the version nibble of each value against the metadata. A hypothetical
  `NormalizedVector` type checks the norm of each row within a tolerance.

### 8.5 Open questions

- **File reads (XQ3).** Decoding builds arrays through the encoding vtable, not through `try_new`.
  A reader can check values, or it can trust the file. These notes recommend a general rule for
  Vortex: decoding checks only what memory safety needs, and it trusts semantic constraints.
  Decoding already checks lengths, offsets and UTF-8, because Rust requires valid UTF-8 for `str`
  (`vortex-array/src/arrays/varbinview/array.rs:234-245`). A strict read mode runs
  `validate_array` on every chunk. No operation panics on an illegal value (I7).
  `Constant(extension scalar)` already checks its value on decode
  (`vortex-array/src/arrays/constant/vtable/mod.rs:160-161`).
- **Values in null slots (XQ4).** Builders fill a non-nullable child under a null parent with
  placeholder values (`vortex-array/src/builders/fixed_size_list.rs:294-305`,
  `vortex-array/src/builders/mod.rs:156-166`). Arrow leaves the child values under a null parent
  undefined. A type such as `NonZero` rejects the placeholder 0.
  - These notes recommend that a null slot can hold any value. The value check skips null slots
    and rows under a null parent. Every public accessor that takes a child out of its parent masks
    the child with the validity of the parent, as `get_item` already does
    (`vortex-array/src/scalar_fn/fns/get_item.rs:205-209`).
  - This recommendation keeps Arrow imports free of copies. It also allows a storage encoding to
    store nulls as a reserved value, such as 0 for `NonZero`.
  - The alternative requires a legal value in every slot. Then imports with nulls must copy, and a
    type with no zero needs a nullable child.

## 9. Nullability

- **NU1, short term.** `ExtDType::try_with_vtable` calls `validate_dtype` for both nullabilities of
  the storage dtype, and fails if either call fails. The documentation states that the result of
  `validate_dtype` must not depend on nullability. `with_nullability` then builds the new dtype
  directly and never validates again. Today it calls `try_with_vtable`
  (`vortex-array/src/dtype/extension/typed.rs:91-98`). A type that breaks the rule fails when its
  dtype is built, not later in a kernel. Calling `validate_dtype` only with the non-nullable form
  is the alternative, but it hides such a type.
- **NU2, long term.** `ExtDType` stores a non-nullable storage dtype and a `Nullability`, like every
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
- Builders fill a non-nullable child under a null parent
  (`vortex-array/src/builders/fixed_size_list.rs:303`, `vortex-array/src/builders/mod.rs:156-166`).
  Under the recommendation of XQ4, any value is allowed there. So the builder can keep the storage
  zero, and the hook is not needed for padding.

## 11. Scalar identity and order

### 11.1 Current code

- `Scalar` derives `Eq` (`vortex-array/src/scalar/mod.rs:41`). `PartialEq` and `Hash` compare the
  dtype without nullability and the value (`vortex-array/src/scalar/scalar_impl.rs:336-356`).
- `PartialOrd` returns `None` for different dtypes. Otherwise it compares the stored values
  (`scalar_impl.rs:358-474`), so extension values compare by storage for every type (EXT-Q2).
  Probe R2 shows `Some(Less)` for two `Opaque` values.
- Floats use bitwise equality and `total_cmp` (`vortex-array/src/dtype/ptype.rs:328-336`).
- `scalar_cmp` computes every comparison operator from `partial_cmp`
  (`vortex-array/src/scalar_fn/fns/binary/compare/mod.rs:241-267`). Constant inputs and the row
  path use it.
- A `Variant` value has no order, so `partial_cmp` returns `None` even when `==` holds
  (`scalar_impl.rs:449-451`).
- An extension over `Union` storage also gets `None` even when `==` holds. The union arm needs
  `DType::Union`, but it receives the extension dtype (`scalar_impl.rs:441-443`). The tuple arm
  unwraps extensions (`scalar_impl.rs:469-471`).
- An extension scalar can hold a value of the wrong storage type. Then `==` can panic (EXT-V7,
  probe R2).
- A valid `F16` scalar can hold a `U64` value, for compatibility with an old serialized form
  (`vortex-array/src/scalar/validate.rs:46-50`, `:147-157`). Then `==` panics in one direction
  and returns `false` in the other (probe R10).

### 11.2 The Rust rule

`PartialOrd` must agree with `PartialEq`: `a == b` holds exactly when `partial_cmp(a, b)` returns
`Some(Equal)`. Suppose `PartialEq` stays structural and `PartialOrd` follows the derives. Then the
rule breaks. For an `Opaque` scalar `x` without `ord`, `x == x` is true, but `partial_cmp(x, x)`
returns `None`.

### 11.3 Contract

- `PartialEq`, `Eq` and `Hash` describe identity, as today: the same dtype without nullability,
  and equal stored values. Decimal values of different widths compare by value
  (`vortex-array/src/scalar/typed_view/decimal/dvalue.rs:255-281`). Primitive values of different
  widths compare by value when one converts to the other. Equality panics when the conversion
  fails, for example `U8(5)` against `I8(-1)`
  (`vortex-array/src/scalar/typed_view/primitive/pvalue.rs:122-138`). A valid primitive value has
  the `PType` of its dtype, except the `U64` form of an `F16` value
  (`vortex-array/src/scalar/validate.rs:46-50`). So valid scalars of one dtype never reach the
  panic after U6.
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
equality is bitwise and float order is `total_cmp`. The steps also fix the `Variant` and `Union`
cases.

### 11.4 Relation to expression semantics

- Identity serves Rust code: maps of constants, deduplication, tests and caches. It never decides
  an expression result for extension values.
- Identity implies semantic equality, because one dtype and one storage value give one value.
  The reverse does not hold. Without `eq`, two values that are not identical can be equal.
- Expression comparisons follow the derive rules. After the rules run, `scalar_cmp` and the nested
  comparator see storage dtypes at every depth (section 4.3). So the order of extension scalars
  never decides an expression result, and `scalar_cmp` keeps its current operators. For example,
  `=` on `struct{v: Vector}` compares `struct{v: FixedSizeList<f32>}` storage, where the built-in
  order agrees with `==`.

### 11.5 Order of work

Change `PartialOrd` only together with, or after, the phase 2 change that rejects `min` and `max`
for extension types without `ord`. Today `min`, `max` and their variants merge partial results with
`vortex_expect` on `partial_min` or `partial_max`. The sites are
`vortex-array/src/aggregate_fn/fns/min/mod.rs:59`, `max/mod.rs:59`, `bounded_min/mod.rs:71`,
`bounded_max/mod.rs:86` and `min_max/mod.rs:247`, `:252`. With the new order, a merge of two
different `Json` maxima panics.

- `ChunkedArrayAggregate` merges one partial for each chunk
  (`vortex-array/src/arrays/chunked/compute/aggregate.rs:28-32`). So `max` over a chunked `Json`
  column panics.
- DuckDB aggregate pushdown accumulates every split into one partial
  (`vortex-duckdb/src/file_reader.rs:298-300`). It does not exclude `Json`
  (`vortex-duckdb/src/table_function.rs:525-537`).
- The default zone-map writer is not affected. Each zone is one canonical chunk with its own
  accumulator (`vortex-layout/src/layouts/repartition.rs:158-161`,
  `vortex-layout/src/layouts/zoned/builder.rs:106-118`).

These results come from reading the code, not from a run.

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
| 0, U9 | `vortex-array/src/aggregate_fn/{storage_pushdown,vtable}.rs`, `vortex-array/src/aggregate_fn/fns/min_max/mod.rs` |
| 1 | `vortex-array/src/dtype/extension/vtable.rs`, `vortex-array/src/arrays/extension/array.rs` (`new_unchecked`), `vortex-array/src/compute/conformance/`, `vortex-array/src/extension/tests/` |
| 2 | `vortex-array/src/dtype/extension/{vtable,typed,erased}.rs`, `vortex-array/src/scalar_fn/vtable.rs`, `vortex-array/src/aggregate_fn/{vtable,erased}.rs`, `vortex-array/src/expr/bound_expression.rs`, `vortex-array/src/arrays/scalar_fn/array.rs`, `vortex-array/src/optimizer/`, `vortex-array/src/scalar_fn/fns/{binary,between,cast,dynamic.rs,ext_storage.rs}`, `vortex-array/src/scalar_fn/fns/binary/compare/nested.rs`, `vortex-array/src/aggregate_fn/fns/{min,max,min_max,bounded_min,bounded_max,is_sorted}/`, `vortex-array/src/arrays/extension/compute/compare.rs`, `vortex-array/src/scalar/scalar_impl.rs`, `vortex-array/src/scalar/typed_view/extension/mod.rs`, `vortex-layout/src/layouts/zoned/writer.rs`, `vortex-row/src/encoder.rs`, `vortex-duckdb/src/table_function.rs`, `vortex-array/src/aggregate_fn/{session,storage_pushdown}.rs` and `encodings/uuid/src/lib.rs` (X10) |
| 3 | `vortex-array/src/scalar_fn/fns/cast/mod.rs`, `vortex-array/src/extension/datetime/`, `vortex-duckdb/src/convert/expr.rs`, `vortex-datafusion/src/convert/exprs/mod.rs` |
| 4a | `vortex-array/src/arrays/extension/compute/{rules,take}.rs`, `vortex-array/src/arrays/extension/vtable/mod.rs`, `vortex-array/src/arrays/{constant,chunked,dict,masked,interleave}/`, `encodings/sparse/`, `encodings/runend/`, `vortex-edition/`. The shared mechanism of #10748 also touches `vortex-array/src/arrays/widen/` |
| 4b | `encodings/datetime-parts/`, `vortex-btrblocks/src/schemes/temporal.rs`, `vortex-layout/src/layouts/zoned/`, `vortex-layout/src/layouts/file_stats.rs`, `vortex-file/src/pruning.rs`, `vortex-expr/src/stats/rewrite/` (moved from `vortex-array` by #10734), `vortex-edition/` |
| 5 | `vortex-array/src/arrays/extension/array.rs`, `vortex-array/src/arrays/extension/vtable/operations.rs`, `vortex-array/src/arrays/extension/compute/`, `vortex-array/src/builders/{extension,fixed_size_list,mod}.rs`, `vortex-array/src/dtype/extension/`, `vortex-array/src/extension/datetime/timestamp.rs`, the Arrow, DuckDB, FFI and Python import paths |
| 6 | `vortex-arrow/src/session.rs`, `vortex-duckdb/src/convert/`, `vortex-duckdb/src/exporter/`, `vortex-datafusion/src/convert/`, `vortex-python/src/scalar/`, `docs/developer-guide/extending/extension-dtypes.md` |

## 13. Tests

- **Path consistency.** A helper in `vortex-array/src/compute/conformance/` takes an expression
  builder and inputs. It evaluates the expression on the array path, on the row path, and with
  constant-encoded inputs. It asserts equal values or an error on all three.
- **Rule coverage.** The same expression gives the same result through `optimize`,
  `optimize_ctx`, execution without optimization, and the row path. This test catches a rule that
  one path does not run.
- **Reference types.** `vortex-array/src/extension/tests/` already has `DivisibleInt`. Add:
  - `Opaque` over `i64`, with no derives. Typing rejects every semantic operation.
  - `DecimalLike` over `i64`, with `eq` and `ord` for each dtype. A differential test runs each
    operation on `DecimalLike` and on the built-in `Decimal`, and compares values or errors.
  - `Refined` over `i64` (for example, positive values only), with every derive and a
    `validate_array` override.
- **Conformance harness for `ExtVTable`.** It checks the metadata round trip, that
  `validate_dtype` ignores nullability, rules DR1 to DR4, cast round trips, and path consistency.
  DR2 and DR3 need the optional reference order. The harness also compares a `validate_array`
  override with the default.
- **Scalar contract.** For generated scalars, `a == b` holds exactly when `partial_cmp(a, b)`
  returns `Some(Equal)`, and equal scalars have equal hashes.
- **Engines.** DuckDB and DataFusion pushdown tests with extension literals for `Timestamp`, `Date`,
  `Uuid` and `Json` columns. A `Json` comparison must not be pushed down.
