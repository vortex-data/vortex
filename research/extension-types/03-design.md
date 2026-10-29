<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 3. Design

This part follows [`../decisions.md`](../decisions.md), which records the decisions of the
maintainer. IDs such as X4 and XQ1 refer to that file. A design pass on 2026-10-29 settled the
mechanism: claims on types (X13 to X16), one declaration on each function (X17 to X19), rules on
the type (X20) and a default storage path for aggregates (X21).

## Goal

Extension types behave like Rust newtypes. Structural operations always work. A semantic operation
works only where the extension type makes a claim about its storage, or provides its own rule. A
function written for a few named types can also accept them. No code path applies storage
semantics to an extension value without one of these.

## Rules

1. A semantic operation on an extension type is rejected unless a claim of the type, a rule of the
   type, or a `Specific` declaration of the function covers it (X1).
2. Structural operations are always allowed.
3. Typing checks every node when code builds it. The default storage paths and the rules of the
   type then turn accepted calls into storage operations, on every path.
4. After the rewrite, the kernels of generic semantic operations, statistics and pruning see storage
   dtypes only. A `Specific` function sees the types that it names.
5. The extension type owns every conversion that changes meaning. There are no implicit casts, and
   every cast is exact (C2).
6. An unknown extension type supports structural operations and `ext_storage` only.

## Models considered

| Model | Summary | Verdict |
| --- | --- | --- |
| Transparent | Every operation runs on storage | Rejected: wrong results for most types |
| Refinement for every type | Inherit reads and predicates, return the parent type for new values | Correct only for true subtypes, kept as an opt-in |
| Opaque, flags in each subsystem | Each consumer reads a declaration and decides | Every consumer must check, and new consumers drift |
| Opaque, kernels per extension ID | Look up `(function, extension ID)` before the generic kernel | Computes results, but statistics and pruning learn nothing |
| Type-owned encodings | Each extension type gets its own array encoding | Mixes logical type and encoding, and generic encodings still carry the type |
| Full dtype plugin | The type defines its own arrays, scalars and compute | The most power and the most work, and storage becomes optional |
| **Opaque, claims and rewrite rules** | Typing matches the claims of a type with the declaration of a function, and rewrites accepted calls into storage operations | **Chosen (X1, X4, X13, X17)** |

The deciding property is the number of places that must understand extension semantics. With
flags, every consumer must read the flag correctly. Today, three groups of consumers give three
different answers to "is this type ordered?" (EXT-A1, EXT-B4):

- `compare` and `min_max` treat every extension type as ordered.
- `is_sorted` and `vortex-row` treat no extension type as ordered.
- DuckDB pushdown treats every extension type except `Uuid` as ordered.

With one typing check and rewrite rules, only that check and the rules need to understand
extension semantics. Unlike kernels, rules also rewrite predicates, so statistics and pruning keep
working on storage.

## Proposal

### Typing and rewrite

A call with an extension operand passes two steps:

1. Typing checks the call when code builds the node. It asks the function how it uses extension
   operands (X17), and it checks the claims of the operand types. A call that no answer and no rule
   of the type covers is an error.
2. A rewrite replaces an accepted call with operations on storage. The default storage paths come
   from the core, and the rules of a type come from its vtable.

```text
ts_ms < lit(Timestamp(ms) 1500)  =>  ext_storage(ts_ms) < 1500i64
min(ts_ms)                       =>  min(ext_storage(ts_ms)), and the result mapping gives the
                                     result the dtype Timestamp(ms) again
zip(c, ts_a, ts_b)               =>  an extension array over zip(c, ext_storage(ts_a),
                                     ext_storage(ts_b))
cast(ts_ms, Timestamp(ns))       =>  Extension(Timestamp(ns),
                                               checked_mul(ext_storage(ts_ms), 1_000_000))
json_a = json_b                  =>  rejected, because Json makes no Eq claim
```

The rewrite runs in the optimizer and in the executor, so the row path and the array path apply
the same rewrite. A missing rule gives an error, never a wrong result. Pushdown, pruning and
compression keep working, because the rewritten operations are ordinary storage operations.

### Claims

A claim is a promise that an extension type makes about its storage (X13). It says that one kind of
operation gives the same answer on the values and on their storage values, in both directions.

- `Eq`: two values are equal exactly when their storage values are equal. So `group by`, `distinct`
  and hashing can run on storage.
- `Ord`: the value order is the storage order. `Ord` implies `Eq`.

```rust
#[non_exhaustive]
pub enum Claim {
    Eq,
    Ord,
}

pub trait ExtVTable {
    /// The claims that values of `dtype` make about their storage.
    ///
    /// The default is no claims, so the type is opaque.
    fn claims(dtype: &ExtDType<Self>) -> &'static [Claim] {
        &[]
    }
}
```

`DType::holds(claim)` answers for any dtype. An extension dtype holds the claims in its list, and
the claims that they imply. Built-in dtypes follow a fixed table, which matches what compare
supports today: `Null`, `Bool`, `Primitive`, `Decimal`, `Utf8` and `Binary` hold both claims.
`Struct`, `List`, `FixedSizeList` and `Map` hold a claim when their children hold it. `Union` and
`Variant` hold neither.

Each type lists its claims by hand. Claims are never derived from which `=` or `<` kernels exist.
Kernels are registered per encoding, so a missing kernel would silently remove a claim. A type with
its own `=` rule, such as a JSON-aware equality, still cannot claim `Eq`, because `group by` and
hashing would run on the raw storage.

`ExtDType::try_with_vtable` checks each claim when it builds the dtype (X15). The storage dtype must
hold every claim of the type, and the claims must be the same for both nullabilities. The check
walks the dtype tree once and stops at nested extension dtypes, which were checked when they were
built. So typing needs to check only the top-level dtype of each argument.

A claim covers values of one dtype (X2). A type stored as floats inherits the float semantics of
Vortex: a total order and bitwise equality.

| Type | Claims |
| --- | --- |
| `Timestamp`, `Date`, `Time` | `Eq`, `Ord` |
| `Duration` (open PR #10467) | `Eq`, `Ord` |
| `Uuid`, ordered by its bytes | `Eq`, `Ord` |
| `Vector`, `FixedShapeTensor` | `Eq` |
| `Decimal`, which stays built-in but follows the same rules (D1) | `Eq`, `Ord`, by the built-in table |
| `Json`, WKB, native geometry | None |
| `Interval` (draft PR #10468) | None, until its PR decides what interval equality means |
| Unknown type | None |

The enum is closed on purpose (X14). A claim that a crate outside Vortex defines needs an open form,
such as a struct with one static for each claim. That switch changes interfaces mechanically, and
the logic stays the same.

### Function declarations

Each scalar function and aggregate has one method that typing calls once for each call (X17):

```rust
pub enum ExtensionUse {
    /// The default. Extension operands are rejected, unless a rule of the type covers the call.
    None,
    /// The function only moves values. It runs on storage, and the result gets the dtype that
    /// `return_dtype` gives.
    Structural,
    /// The function handles these extension types itself.
    Specific,
    /// The function runs on storage. `claims[i]` is the claim that argument `i` needs, or `None`
    /// for an argument that is only moved. `result` maps the storage result back.
    Storage { claims: Vec<Option<Claim>>, result: ResultMapping },
}

fn extension_use(&self, options: &Self::Options, arg_dtypes: &[DType]) -> ExtensionUse {
    ExtensionUse::None
}
```

Functions fall into four groups (X18):

1. Functions that move values, such as `zip`, `case_when` and `fill_null`, answer `Structural`.
   Their value operands already need one dtype, ignoring nullability. Functions that never look
   inside an operand, such as `get_item`, `pack` and `is_null`, need nothing at all.
2. Functions that need only equality or order answer `Storage`. Examples are compare, `between`,
   `list_contains` over lists of extension values, `min`, `max`, `min_max`, `is_sorted`, sorting
   and the row encoding.
3. Other functions that read values, such as `byte_length` or `like` on `Json`, are rejected until a
   new claim covers them.
4. Functions written for a few named types answer `Specific` for those types and `None` for the
   rest. Examples are `replace_time_zone` for `Timestamp`, the spatial functions, and
   `json_to_variant`. Each one names its types in its own `return_dtype`.

The default `None` keeps generic functions safe. A function that accepts any dtype, such as a hash
of storage bytes, would otherwise apply storage semantics without a word. `Specific` and `None`
differ only in whether the function's own `return_dtype` sees the extension operand, so their
documentation must make the difference clear.

Claims are per argument. A hypothetical `sort_by(values, keys)` needs `Ord` on `keys` and nothing
on `values`. Every extension type inside one argument must hold the claim of that argument. A part
of an argument that needs another claim becomes a separate argument, for example
`sort_by(payload, get_item(row, "ts"))`.

### Result mappings

`Storage` carries the result mapping, so the compiler forces every function that runs on storage
to give one (X19):

```rust
pub enum ResultMapping {
    /// Wrap the storage result in this extension dtype.
    Rewrap(ExtDTypeRef),
    /// Leave the result as it is.
    Keep,
    /// One mapping for each field of a struct result.
    Fields(Vec<ResultMapping>),
    /// The function maps the result itself, in both directions for partial states.
    Custom(Arc<dyn CustomMapping>),
}
```

- `min` and `max` answer `Rewrap` with the input dtype, and `min_max` answers `Fields` with two
  `Rewrap`s.
- Compare and `is_sorted` answer `Keep`, because they return `Bool`.
- `bounded_min` and `bounded_max` answer `Keep` over `Utf8` and `Binary` storage, where they can
  truncate a value into a bound that is not a legal value. Over other storage they answer `Rewrap`.
- A hypothetical `list_max(List<Timestamp>)` answers `Rewrap(Timestamp)`, so a result that is part
  of an input needs no `Custom`.

When the call is built, the guard of `Rewrap` checks that the storage result dtype equals the
storage dtype of the target, ignoring nullability. The result keeps the nullability of the storage
result. `Rewrap` trusts the values and uses `ExtensionArray::new_unchecked`, and debug builds run
the value check. A claim that allows `Rewrap` must guarantee legal results. `Eq` and `Ord` do,
because their results are input values. A future arithmetic claim can overflow, so it needs a
checked mapping.

### Aggregates

`develop` has a narrower opt-in for aggregates, from
[#10566](https://github.com/vortex-data/vortex/pull/10566) and
[#10616](https://github.com/vortex-data/vortex/pull/10616). `ExtensionStoragePushdown<A, E>` runs
aggregate `A` on the storage of extension type `E`, under a separate aggregate ID such as
`vortex.min_max[vortex.timestamp]`. `register_storage_ordered` registers it by hand for `Date`,
`Time`, `Timestamp` and `Uuid`. Plain `min_max` and `is_sorted` do not consult it, and today only
tests ask for the composite IDs. Its results are storage values (EXT-A3).

Its computation is right, so it becomes the default storage path of aggregate binding (X10, X21).
When `AggregateDTypes::try_new` meets an extension input, it asks the type first:

1. If a rule of the type covers the aggregate, that rule supplies the accumulator. `sum` over a
   decimal-like type takes this path.
2. If the aggregate answers `Storage` and the claims hold, binding runs the body of today's adapter
   under the plain aggregate ID. It unwraps each batch and runs a nested accumulator over the
   storage, so the storage kernels still apply.
3. Otherwise binding fails.

A partial state stays in storage form in memory. It gets the logical form when it leaves the
accumulator: `finalize`, `finalize_scalar` and `to_scalar` wrap, and `partial_from_scalar`
unwraps. So consumers see the same partial dtype as today, and the cost is constant for each
partial. The composite IDs and `register_storage_ordered` go away. U9 adds the result mappings
first and uses them in today's adapter, so its results are right before the move.

### Operations across dtypes

Two dtypes with different metadata are different dtypes, for example `Timestamp(ms)` and
`Timestamp(ns)`. A claim never applies across them, and Vortex rejects such an operation (X3).
An engine converts the operands to one dtype first, as its planner already does for SQL. A user of
the Rust or Python API writes an explicit cast.

### Rules that the type owns

A type implements its own semantics as rewrite rules on arrays (X4). The rules live on `ExtVTable`,
written over `ReduceNode`, so that they also rewrite bound expressions for pruning (X20). They
travel with the dtype, so the static optimizer, the session-aware optimizer and the executor all
find them with no session. Cast rules convert the type to another dtype, or another dtype into the
type. Call rules cover other functions and aggregates, such as `sum` over a decimal-like type. A
rule of the type wins over the default storage path.

Only the crate of a type adds rules for that type, as with the orphan rule of Rust. A crate that
owns neither the function nor the type defines its own function, or its own extension type over
the same storage. A session registry would bring back conflicts between crates and rules that the
plain `.optimize()` calls miss.

A cast exists only if a rule covers it (C1). This is the design of issue
[#8570](https://github.com/vortex-data/vortex/issues/8570), with rules in place of closures. Every
cast that the extension work adds is exact (C2). `Timestamp(ms)` to `Timestamp(ns)` fails on
overflow. `Timestamp(ns)` to `Timestamp(ms)` is not a cast: a named function with an explicit
rounding mode does it.

No `ext_wrap` expression exists. A rule that produces extension values builds the extension array
itself. It uses `ExtensionArray::new_unchecked` when legal input gives legal output, and
`ExtensionArray::try_new` otherwise. So no user expression can build a wrapper without a check.

Two explicit expressions remain:

- `ext_storage(x)` reads the storage. It exists today. It never checks values.
- `ext_storage_deep(x)` removes every extension layer at every depth. It is new. The default storage
  paths use it for operands that hold nested extension values, for example in a comparison of
  `struct{ts: Timestamp}` values.

### Scalar identity

`Scalar` implements `PartialEq`, `Eq`, `Hash` and `PartialOrd`. These traits serve Rust code, such
as maps and tests. Expression results follow the claims, not these traits. What the traits mean
for Vortex is open (XQ8). Any change of `PartialOrd` must land with or after the change that
rejects `min` and `max` for types without `Ord`, because merges of partial results call
`vortex_expect` on the order.

### Physical form

A rewrite rule pulls generic structural arrays, such as `Constant` and `Dict`, out from under the
extension wrapper (X5). Code that looks for a constant or a dictionary then finds it at the top,
without any knowledge of extension types.

Which arrays the rule pulls out is open (XQ1). These notes recommend this split:

- Above the wrapper: `Constant`, `Dict`, `RunEnd`, `Sparse`, `Chunked`, `Masked` and `Interleave`.
- Below the wrapper: the canonical storage, storage encodings such as BitPacked or FSST, and slices,
  filters and scalar functions that wait to run.
- A canonical extension array is a wrapper over canonical storage. So a canonical array never hides
  a pending computation or a compressed child.

Each structural encoding gets one method that rebuilds it with its value children passed through a
function. That method moves the wrapper, pushes `ext_storage` down, and decodes a structural
encoding that holds extension values. `RunEnd` and `Sparse`, which fail on extension values today,
get support from the same method.

The lift rule is the extension case of issue
[#10748](https://github.com/vortex-data/vortex/issues/10748), which keeps generic structural
encodings above every other encoding. Nothing depends on it for correctness. Without it, fast paths
that look only at the root of an array, such as dictionary export and `as_constant`, fall back to
slower paths with the same results. It has low priority.

Type-specific encodings are a separate question. `DateTimeParts` can move under the wrapper as an
`i64` encoding, so that its compare kernel works on storage. The compressor then chooses it from
the extension metadata. Until then, a compare over it decodes, which is slower but correct. This
also has low priority.

[`agents/design-notes.md`](agents/design-notes.md), section 6, has the details.

### Statistics

Zone maps keep logical `Min` and `Max` for extension types that claim `Ord` (X22). Their stored
dtype does not change. A logical min holds its storage min, so pruning can compare it with a
literal of the type, or unwrap it to a storage bound. Both read the same stored value.

Zone maps write no min and max for other extension types, so the storage fallback goes. No legal
filter can use those values, because a compare on such a type is rejected. The null count stays,
and a type can add its own statistics through its own aggregates, as the spatial bounding boxes of
[#10837](https://github.com/vortex-data/vortex/pull/10837) do.

Vortex has stored storage-order min and max for every extension type for at least a year
(EXT-A2). Readers ignore stored min and max of an extension type without `Ord`, in pruning and in
the file statistics that reach engines. They log a warning once for each file and column. A type
that gains `Ord` later can use its old statistics, because `Ord` means storage order.

### Validation

Three kinds of validation exist:

- Dtype validation checks the metadata, the storage dtype and the claims, once for each dtype.
- Storage-shape validation checks that storage has the storage dtype, for arrays and scalars.
- Value validation (`validate_array`) checks that each non-null value is a legal value of the
  type.

Every extension array that Vortex builds holds only legal values (X7). `ExtensionArray::try_new`
checks the values in one batch with `validate_array`. Today it checks only the storage dtype, which
is a bug (EXT-V1). Kernels that keep legal values use a new `ExtensionArray::new_unchecked`, so they
pay nothing.

The default `validate_array` applies the existing scalar check of the type to each value. A type
can override it with a faster check, for example a range check on the minimum and maximum of a
`Timestamp` column.

File reads follow the existing pattern of the repository (X23). Every deserialize ends in
`Array::try_from_parts`, which runs only the structural `VTable::validate`. So the value check lives
in `ExtensionArray::try_new` only, never in `Extension::validate`. An optional strict read mode
checks values. Because files are trusted, building a scalar from an array always checks the value
and returns an error for an illegal one. No read panics.

The extension type adds no rule about null slots. The storage array must be a valid array of its
own encoding, which decides what a null slot can hold. The value check reads only non-null slots.

### Nullability

The short-term rule: the result of `validate_dtype` and the claims must not depend on nullability.
Building a dtype checks both nullabilities, so a later nullability change never validates the type
again. The long-term option: store nullability on the extension dtype, like every other `DType`
variant, and keep the storage dtype non-nullable.

### Authoring and boundaries

The core provides a typed array view, a typed accessor for native values, metadata helpers, and a
conformance test harness for `ExtVTable` implementations.

Every boundary keeps the extension type or fails (X24). Storage leaves only through `ext_storage`.

- Arrow and DataFusion export a type without a plugin as its storage, with `ARROW:extension:name`
  set to the Vortex ID and `ARROW:extension:metadata` set to its metadata. A consumer that knows the
  type rebuilds it. Import recognizes Vortex IDs, so the round trip loses nothing. Today export
  fails without a plugin, but a storage-like target type leaks storage (EXT-B2).
- DuckDB, Python and JNI use a converter for each type, modeled on the Arrow registry, or fail.
- Engine pushdown reads the claims, not the storage type.

The implicit conversions of today cover some types, so new plugins and converters fill the gaps
when those conversions stop.

## Costs and risks

- Typing must cover every path that builds a node. The types enforce it: `BoundExpression` loses
  its public variants, and `Cast::new` becomes fallible in the same step as typing. The design
  accepts these API changes for correctness (X8).
- The rewrite must run on every path: the static optimizer, the session-aware optimizer, and the
  executor. A test checks that all paths give the same result.
- Old readers cannot decode `RunEnd` and `Sparse` over extension values. So writers keep the old
  shape of those two until an edition allows the new one.
- Engines must send extension literals for extension columns. Today the lone-extension compare rule
  can hide literals that have the storage type.
- After the rewrite, plans show storage operations, so plans are harder to read.
- Some conversions that work today through storage stop until a plugin or a converter covers them.
- The aggregate work coordinates with
  [#10769](https://github.com/vortex-data/vortex/pull/10769), which moves the aggregate
  implementations into a new `vortex-aggregates` crate, and with #10837. Statistics work in
  [#10482](https://github.com/vortex-data/vortex/pull/10482) (file `is_sorted` statistics) and
  [#10686](https://github.com/vortex-data/vortex/pull/10686) (comparisons proven true from
  statistics) meets the claims and the compare rewrite.

## Open questions

The full list with recommendations is in [`../decisions.md`](../decisions.md). The questions that
stay open for extension types are:

- XQ1: which arrays the rule of X5 pulls out, and whether canonical form requires canonical storage.
  It has low priority.
- XQ8: what the Rust traits of `Scalar` mean for Vortex.
- Whether a claim beyond `Eq` and `Ord` is needed, for example for text functions on `Json`, or for
  comparisons of a refinement type with plain storage values.
- Whether nullability moves onto the extension dtype.
