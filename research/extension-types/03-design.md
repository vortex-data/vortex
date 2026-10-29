<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 3. Design

This part follows [`../decisions.md`](../decisions.md), which records the decisions of the
maintainer. IDs such as X4 and XQ1 refer to that file.

## Goal

Extension types behave like Rust newtypes. Structural operations always work. A semantic operation
works only where the extension type opts in or provides its own rule. A function that declares that
it understands the type can also allow it. No code path applies storage semantics to an extension
value without that opt-in.

## Rules

1. A semantic operation on an extension type is rejected unless the type or a type-aware function
   opts in (X1).
2. Structural operations are always allowed.
3. Two kinds of rules decide how generic operations treat extension values: the derive rules of the
   core and the rules that each type owns. Every node meets a typing check when code builds it, and
   the rules run on every path.
4. After the rules run, the kernels of generic semantic operations, statistics and pruning see
   storage dtypes only. A function that declares an extension type can see that type.
5. The extension type owns every conversion that changes meaning. There are no implicit casts.
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
| **Opaque, rewrite rules** | Derive rules and rules that the type owns rewrite semantic operations into storage operations | **Chosen (X1, X4)** |

The deciding property is the number of places that must understand extension semantics. With
flags, every consumer must read the flag correctly. Today, three groups of consumers give three
different answers to "is this type ordered?" (EXT-A1, EXT-B4):

- `compare` and `min_max` treat every extension type as ordered.
- `is_sorted` and `vortex-row` treat no extension type as ordered.
- DuckDB pushdown treats every extension type except `Uuid` as ordered.

With rewrite rules, only the rules need to understand extension semantics. Unlike kernels, rules
also rewrite predicates, so statistics and pruning keep working on storage.

## Proposal

### Typing and rewrite rules

A call with an extension operand passes two steps:

1. Typing checks the call when code builds the node. It accepts four kinds of calls. They are
   structural calls, calls of a type-aware function, calls that a derive covers, and calls that a
   rule of the type covers. Everything else is an error.
2. Rewrite rules replace an accepted semantic call with operations on storage.

```text
ts_ms < lit(Timestamp(ms) 1500)  =>  ext_storage(ts_ms) < 1500i64
min(ts_ms)                       =>  min(ext_storage(ts_ms)), and the rule builds a Timestamp(ms)
                                     scalar from the result
cast(ts_ms, Timestamp(ns))       =>  Extension(Timestamp(ns),
                                               checked_mul(ext_storage(ts_ms), 1_000_000))
json_a = json_b                  =>  rejected, because Json derives no equality
```

The rules run in the optimizer and in the executor, so the row path and the array path apply the
same rewrite. A missing rule gives an error, never a wrong result. Pushdown, pruning and
compression keep working, because the rewritten operations are ordinary storage operations.

### Derives for one dtype

A derive says that a semantic operation on values of one extension dtype is the same operation on
their storage values. It is a property of one dtype (X2). The goal is that an author can opt in to
common behavior, such as equality and order, with little code. The mechanism in this section is a
sketch of that goal, not a decision (X11, XQ7):

```rust
/// Storage behavior that values of `dtype` inherit, like `#[derive]` on a newtype.
///
/// The default inherits nothing, so every semantic operation on this dtype is rejected.
fn derives(dtype: &ExtDType<Self>) -> Derives {
    Derives::default()
}
```

`Derives` has two flags. `eq` means that storage equality is value equality, which also makes the
storage hash valid. `ord` means that storage order is value order.

A derive applies only when every operand in a derived position has the same extension dtype:

- A comparison needs two such operands, and `between` needs three.
- A unary aggregate, such as `min` or `is_sorted`, reads the flags of its one input.
- A struct or list that holds extension values inherits a flag only if every nested extension
  type declares it. The derive rule then removes every extension layer at every depth, so the
  kernel sees only storage (X6).

So `ts < 5i64` stays rejected, because only one side is an extension.

| Type | Derives |
| --- | --- |
| `Timestamp` | `eq`, `ord` |
| `Date`, `Time` | `eq`, `ord` |
| `Decimal`, which stays built-in but follows the same rules (D1) | `eq`, `ord` |
| `Uuid` | `eq`, plus `ord` if byte order is the chosen order |
| `Vector`, `FixedShapeTensor` | `eq` |
| `Json`, WKB, native geometry | None |
| Unknown type | None |

### The aggregate opt-in today

`develop` already has an opt-in for aggregates, from
[#10566](https://github.com/vortex-data/vortex/pull/10566) and
[#10616](https://github.com/vortex-data/vortex/pull/10616). `ExtensionStoragePushdown<A, E>` runs
aggregate `A` on the storage of extension type `E`, under a separate aggregate ID such as
`vortex.min_max[vortex.timestamp]`. `register_storage_ordered` registers it for `is_sorted` and
`min_max`. The default session calls it for `Date`, `Time` and `Timestamp`, and
`vortex_uuid::initialize` calls it for `Uuid`.

The adapter fits this design, and it stays (X10). It is how a type with an order opt-in gets
`is_sorted` and `min_max` on its storage. Two parts change:

- Registration. Today a crate calls `register_storage_ordered` by hand, and nothing checks the call.
  The docs of the adapter say that an invalid registration "can silently return incorrect results".
  In this design, a declaration on the type registers the adapter automatically. The type author
  then makes the claim once, as for any other opt-in. XQ7 decides which declaration does this.
- Results. Today the results are storage values. The min of a `Timestamp(ms)` column is a raw
  `i64`, without its time unit or time zone. Nothing wraps the result later, and plain `min_max`
  already returns extension values. The adapter must give each result that is a value of the input
  the extension dtype back, before it returns the result (X12, U9). The `Bool` of `is_sorted` stays
  as it is.

One question stays open. Plain `min_max` and `is_sorted` do not consult the adapter, so a consumer
gets the opt-in only if it asks for the composite ID, and today only tests do. Meanwhile plain
`min_max` uses storage order for every extension type, and plain `is_sorted` rejects every extension
type. These notes recommend that the plain aggregates route to the registered adapter, so that
consumers such as zone maps and engine pushdown need no composite IDs (XQ7).

### Operations across dtypes

Two dtypes with different metadata are different dtypes, for example `Timestamp(ms)` and
`Timestamp(ns)`. A derive never applies across them, and Vortex rejects such an operation (X3).
An engine converts the operands to one dtype first, as its planner already does for SQL. A user of
the Rust or Python API writes an explicit cast.

### Function declarations

Each function declares how it uses an operand that contains an extension type:

- Structural: it moves or selects values, for example `fill_null` or `case_when`.
- Derive: it needs `eq` or `ord`, for example `=` or `<`.
- Type-aware: it implements the semantics of named extension types itself, for example
  `replace_time_zone` for `Timestamp`.
- Nothing: the default. A rule of the type covers the call, or typing rejects it.

Only the function owner declares. Typing never infers a declaration from the input types that a
function accepts, because `Binary` accepts extension operands today.

An aggregate with a derive also declares whether its result is a value of its input, a bound on its
input, or neither (XQ5, open). Only a value is wrapped again. For example, `min` returns a value,
and `bounded_max` over `Utf8` storage returns a truncated bound, which stays a storage value.

### Rules that the type owns

A type implements its own semantics as rewrite rules on arrays (X4). The rules are registered with
the type, together with its compression scheme and the other parts of the type. Cast rules convert
the type to another dtype, or another dtype into the type. Call rules cover other functions and
aggregates, such as `sum` over a decimal-like type.

A cast exists only if a rule covers it (C1). This is the design of issue
[#8570](https://github.com/vortex-data/vortex/issues/8570), with rules in place of closures.

No `ext_wrap` expression exists. A rule that produces extension values builds the extension array
itself. It uses `ExtensionArray::new_unchecked` when legal input gives legal output, and
`ExtensionArray::try_new` otherwise. So no user expression can build a wrapper without a check.

Two explicit expressions remain:

- `ext_storage(x)` reads the storage. It exists today. It never checks values.
- `ext_storage_deep(x)` removes every extension layer at every depth. It is new. The derive rules
  use it for operands that hold nested extension values, for example in a comparison of
  `struct{ts: Timestamp}` values.

Where the rules live is open (XQ2). These notes recommend a method on `ExtVTable`, so that the
rules travel with the dtype and need no session lookup. The existing session registry for array
kernels is keyed by encoding, and every extension array has the same encoding id. Also, 83 plain
`.optimize()` calls outside tests never consult that registry.

### Scalar identity

`Scalar` implements `PartialEq`, `Eq` and `Hash` as identity: the same dtype and the same stored
value. These traits serve Rust code, such as maps and tests, not SQL semantics. `PartialOrd`
returns `Equal` for identical scalars, and an order only for ordered dtypes. An extension dtype is
ordered only if it declares `ord`. This keeps the Rust rule that `==` and `partial_cmp` agree.
Expression comparisons follow the derive rules, not these traits.

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

Type-specific encodings are a separate question. `DateTimeParts` moves under the wrapper as an
`i64` encoding, so its compare kernel works on storage. The compressor chooses it from the
extension metadata.

[`agents/design-notes.md`](agents/design-notes.md), section 6, has the details.

### Statistics

Zone maps and file statistics store storage statistics for extension columns. The derive rules
produce predicates on `ext_storage(col)`, and pruning uses the storage statistics for them.
`vortex-spatial` already prunes in this way, by hand.

### Validation

Three kinds of validation exist:

- Dtype validation checks the metadata and the storage dtype, once for each dtype.
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

Two questions stay open:

- Do file reads check values (XQ3)? These notes recommend that decoding checks only what memory
  safety needs, as it already does for UTF-8, and trusts the rest. A strict read mode checks values.
  Because files are trusted, building a scalar from an array always checks the value and returns an
  error for an illegal one. No read panics.
- Which values can sit in null slots (XQ4)? These notes recommend any value. A child that leaves
  its parent is masked with the validity of the parent. This keeps Arrow imports free of copies,
  and it also allows a storage encoding to store nulls as a reserved value.

### Nullability

The short-term rule: the result of `validate_dtype` must not depend on nullability. Building a dtype
checks both nullabilities, so a later nullability change never validates the type again. The
long-term option: store nullability on the extension dtype, like every other `DType` variant, and
keep the storage dtype non-nullable.

### Authoring and boundaries

The core provides a typed array view, a typed accessor for native values, metadata helpers, and a
conformance test harness for `ExtVTable` implementations. DuckDB, DataFusion and Python get export
plugins like the Arrow registry. Then a new extension type does not need code in each engine crate.

## Costs and risks

- Typing must cover every path that builds a node. The types enforce it: `BoundExpression` loses
  its public variants, and `Cast::new` becomes fallible in the same step as typing. The design
  accepts these API changes for correctness (X8).
- The rules must run on every path: the static optimizer, the session-aware optimizer, and the
  executor. A test checks that all paths give the same result.
- `DateTimeParts` changes from a `Timestamp` encoding to an `i64` encoding. That is a file format
  change for the encoding, and it needs an edition.
- Old readers cannot decode `RunEnd` and `Sparse` over extension values. So writers keep the old
  shape of those two until an edition allows the new one.
- Engines must send extension literals for extension columns. Today the lone-extension compare rule
  can hide literals that have the storage type.
- After the rules run, plans show storage operations, so plans are harder to read.
- The new scalar order must land with or after the change that rejects `min` and `max` without
  `ord`. Otherwise `min` and `max` can panic, for example on a chunked `Json` column. This risk
  comes from reading the code, not from a run.

## Open questions

The full list with recommendations is in [`../decisions.md`](../decisions.md). The questions for
extension types are:

- XQ1: which arrays the rule of X5 pulls out, and whether canonical form requires canonical storage.
- XQ2: where the rules of a type live. A related question is whether a crate can register a rule
  for a function and a type that it does not own.
- XQ3: whether file reads check values.
- XQ4: which values can sit in null slots.
- XQ5: how an aggregate declares its result.
- XQ6: whether `Uuid` orders by bytes, or has no order.

Older questions also stay open:

- Are `eq` and `ord` enough? A refinement over a storage dtype can need a third flag that allows
  comparisons with plain storage values.
- Existing files store min and max for extension columns as extension scalars. Readers can convert
  them to storage values, but the switch needs a compatibility decision.
- Is the nullability contract enough, or does nullability move onto the extension dtype?
