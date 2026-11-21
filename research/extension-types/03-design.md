<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 3. Design

This part follows [`../decisions.md`](../decisions.md), which records the decisions of the
maintainer. IDs such as X4 and XQ1 refer to that file. A design pass on 2026-10-29 settled the
mechanism: claims on types (X13 to X16), one declaration on each function (X17 to X19), rules on
the type (X20) and a default storage path for aggregates (X21). A second pass on 2026-10-30 settled
nullability and the storage declaration (X25 to X30), the range of `Timestamp` (X31) and the
boundary that rewrites must respect (X33, X34). A pass on 2026-11-17 removed the declaration on
each function. Each function now checks the claims in its own `return_dtype`, and its own code runs
accepted calls on storage (X36 to X38).

## Goal

Extension types behave like Rust newtypes. Structural operations always work. A semantic operation
works only where the extension type makes a claim about its storage, or provides its own rule. A
function written for a few named types can also accept them. No code path applies storage
semantics to an extension value without one of these.

## Rules

1. A semantic operation on an extension type is rejected unless a claim of the type, a rule of the
   type, or a function written for that type covers it (X1, X36).
2. Structural operations are always allowed.
3. Typing checks every node when code builds it, in the `return_dtype` of the function (X36). The
   function then runs an accepted call on storage, or a rule of the type rewrites it (X37).
4. Code that reads values without building a node checks the claims directly (X38). This covers
   statistics, pruning, the row encoding and the pushdown decisions of engines.
5. The extension type owns every conversion that changes meaning. There are no implicit casts (C3),
   and every cast is exact (C2).
6. An unknown extension type supports structural operations and `ext_storage` only.

## Models considered

| Model | Summary | Verdict |
| --- | --- | --- |
| Transparent | Every operation runs on storage | Rejected: wrong results for most types |
| Refinement for every type | Inherit reads and predicates, return the parent type for new values | Correct only for true subtypes, kept as an opt-in |
| Opaque, flags in each subsystem | Each consumer reads its own declaration and decides | Every consumer must check, and new consumers drift |
| Opaque, kernels per extension ID | Look up `(function, extension ID)` before the generic kernel | Computes results, but statistics and pruning learn nothing, and typing cannot see a session |
| Type-owned encodings | Each extension type gets its own array encoding | Mixes logical type and encoding, and generic encodings still carry the type |
| Full dtype plugin | The type defines its own arrays, scalars and compute | The most power and the most work, and storage becomes optional |
| Opaque, a declaration on each function and a generic rewrite | A central check matches the claims of a type with an `extension_use` declaration of each function, and one rewrite runs accepted calls on storage (X17 to X19) | Chosen on 2026-10-29, replaced on 2026-11-17: it adds a second type checker beside `return_dtype` (X36) |
| **Opaque, claims checked by each function** | The type declares its claims. Each function checks them in its own `return_dtype` and runs accepted calls on storage | **Chosen (X1, X13, X36, X37)** |

The deciding property is the number of places that must understand extension semantics. Today,
three groups of consumers give three different answers to "is this type ordered?" (EXT-A1,
EXT-B4):

- `compare` and `min_max` treat every extension type as ordered.
- `is_sorted` and `vortex-row` treat no extension type as ordered.
- DuckDB pushdown treats every extension type except `Uuid` as ordered.

Each group guesses from the storage. With claims, every consumer asks the same question of the
type, `DType::holds(Claim::Ord)`, so no two consumers can give different answers. The chosen model
differs from flags in each subsystem in that point: one fact on the type, read by everyone. A
consumer can still forget to ask. The plan covers that risk with code review and an opaque test
type that claims nothing (X36).

## Proposal

### Typing and execution

A call with an extension operand passes two steps:

1. Typing checks the call when code builds the node. The `return_dtype` of the function checks the
   claims that it needs, with `DType::holds` (X36). A call that no claim and no rule of the type
   covers is an error.
2. Execution runs an accepted call on storage. The function does this itself, through shared
   helpers, and gives the result its logical dtype (X37). A rule of the type can replace the call
   with its own operations (X20).

```text
ts_ms < lit(Timestamp(ms) 1500)  =>  compares the storage: ts_ms storage < 1500i64
min(ts_ms)                       =>  min over the storage, and min gives the result the
                                     dtype Timestamp(ms) again
zip(c, ts_a, ts_b)               =>  an extension array over zip(c, ext_storage(ts_a),
                                     ext_storage(ts_b))
cast(ts_ms, Timestamp(ns))       =>  Extension(Timestamp(ns),
                                               checked_mul(ext_storage(ts_ms), 1_000_000))
json_a = json_b                  =>  rejected when it is bound, because Json makes no Eq claim
```

Typing runs wherever a node is built, so the array path, the row path and constant inputs all
reject the same calls. Each storage path must also give the same result on all three paths, and
the path-consistency helper of #11013 checks it. A missing claim gives an error, never a wrong
result. Pushdown, pruning and compression keep working, because the storage paths run ordinary
storage operations.

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
    /// The claims that values of this type make about their storage.
    ///
    /// The default is no claims, so the type is opaque.
    fn claims(metadata: &Self::Metadata) -> Claims {
        Claims::none()
    }
}
```

The method takes the metadata, like `storage_dtype` (X27), so the claims cannot depend on the
top-level nullability. XQ9 records this recommendation. `Claims` is an opaque value: a type writes
`Claims::none().with(Claim::Ord)`, and adding `Ord` also adds `Eq` (X14).

`DType::holds(claim)` answers for any dtype. An extension dtype holds the claims that it declares,
and the claims that they imply. Built-in dtypes follow a fixed table, which matches what compare
supports today: `Null`, `Bool`, `Primitive`, `Decimal`, `Utf8` and `Binary` hold both claims.
`Struct`, `List`, `FixedSizeList` and `Map` hold a claim when their children hold it. `Union` and
`Variant` hold neither.

Each type lists its claims by hand. Claims are never derived from which `=` or `<` kernels exist.
Kernels are registered per encoding, so a missing kernel would silently remove a claim. A type with
its own `=` rule, such as a JSON-aware equality, still cannot claim `Eq`, because `group by` and
hashing would run on the raw storage.

`ExtDType::try_new` checks each claim when it builds the dtype (X15). The declared storage dtype
must hold every claim of the type, and the claims must not depend on the top-level nullability. The
check walks the dtype tree once and stops at nested extension dtypes, which were checked when they
were built. So typing needs to check only the top-level dtype of each argument.

A claim covers values of one dtype (X2). A type stored as floats inherits the float semantics of
Vortex: a total order and bitwise equality.

| Type | Claims |
| --- | --- |
| `Timestamp`, `Date`, `Time` | `Eq`, `Ord` |
| `Duration` (PR #10467, closed as stale on 2026-11-10) | `Eq`, `Ord` |
| `Uuid`, ordered by its bytes | `Eq`, `Ord` |
| `Vector`, `FixedShapeTensor` | `Eq` |
| `Decimal`, which stays built-in but follows the same rules (D1) | `Eq`, `Ord`, by the built-in table |
| `Json`, WKB, native geometry | None |
| `Interval` (PR #10468, closed as stale on 2026-11-10) | None, until a new PR decides what interval equality means |
| Unknown type | None |

The enum is closed on purpose (X14). A claim that a crate outside Vortex defines needs an open form,
such as a struct with one static for each claim. That switch changes interfaces mechanically, and
the logic stays the same.

### Typing in each function

`return_dtype` is the type checker of Vortex. It runs wherever a node is built:
`BoundExpression::try_new_vec`, `ScalarFnArray::try_new_from_slots`, `ScalarFn::validate` and
`AggregateDTypes::try_new`. So each function checks the claims that it needs there (X36). There is
no separate declaration on the function.

```rust
impl ScalarFnVTable for Binary {
    fn return_dtype(&self, operator: &Operator, args: &[DType]) -> VortexResult<DType> {
        // ...
        if let Some(claim) = operator.required_claim() {
            validate_comparison_dtypes(lhs, rhs)?;
            lhs.require_claim(claim, operator)?;
        }
        // ...
    }
}

impl AggregateFnVTable for MinMax {
    fn return_dtype(&self, _options: &EmptyOptions, input: &DType) -> Option<DType> {
        let supported = match input {
            DType::Extension(ext) => {
                input.holds(Claim::Ord) && minmax_supported_dtype(ext.storage_dtype())
            }
            _ => minmax_supported_dtype(input),
        };
        supported.then(|| make_minmax_dtype(input))
    }
}
```

`DType::require_claim` is a shared helper. It returns an error that names the function, the dtype
and the missing claim, so every function reports the same way. `DType::holds` walks nested dtypes, so the
same check covers `List<Json>` in a compare of lists.

Functions fall into four groups (X18). The groups are documentation, not API:

1. Functions that move values, such as `zip`, `case_when` and `fill_null`, need no check. Their
   value operands already need one dtype, ignoring nullability. Functions that never look inside
   an operand, such as `get_item`, `pack` and `is_null`, need nothing either.
2. Functions that need only equality or order check the claim. Examples are compare, `between`,
   `list_contains` over lists of extension values, `min`, `max`, `min_max`, `is_sorted`, sorting
   and the row encoding.
3. Other functions that read values, such as `byte_length` or `like` on `Json`, keep rejecting
   extension operands until a new claim covers them. They match on concrete dtypes, so they reject
   `DType::Extension` today.
4. Functions written for a few named types, such as `replace_time_zone` for `Timestamp`, the
   spatial functions and `json_to_variant`, check those types in their own `return_dtype`, as they
   do today. They match a type by its vtable, for example `ext.is::<Timestamp>()`, never by ID
   string, so a `ForeignExtDType` with the same ID never matches.

The earlier design had a default reject (X17): a new function rejected extension operands until it
declared a use. This design gives that up. A function that matches on concrete dtypes rejects
`DType::Extension` by itself, so a leak needs explicit extension code once the generic storage
unwraps are gone. Code review catches that code, and an opaque test type that claims nothing runs
against each function family.

Claims land first, because they are small (X36). Each function family then switches from today's
storage behavior to a claim check in one PR, and removes its generic storage unwrap in the same PR.
There are no hardcoded lists of types (X35 is superseded).

### Results

Each function gives its own result dtype, and its storage path gives its results that dtype (X37).
There is no `ResultMapping`.

- Compare, `between`, `list_contains` and `is_sorted` return `Bool`, so their storage result needs
  no change.
- `min`, `max` and `min_max` build their result dtypes over the input dtype, for example
  `make_minmax_dtype(input)`. Their storage path wraps each storage result in the input dtype, as
  plain `min_max` does today (`vortex-array/src/aggregate_fn/fns/min_max/extension.rs`).
- A result that is a stored value is a legal value, so the storage path wraps it with
  `ExtensionArray::new_unchecked`. Debug builds run the value check of the type.
- A function whose result is a new value, such as a future arithmetic function, must build it with
  `ExtensionArray::try_new`, because the claims of X13 do not promise that new values are legal.

The rule of X30 still holds: a storage path never inserts a cast to make dtypes match.

### Aggregates

`develop` has a narrower opt-in for aggregates, from
[#10566](https://github.com/vortex-data/vortex/pull/10566) and
[#10616](https://github.com/vortex-data/vortex/pull/10616). `ExtensionStoragePushdown<A, E>` runs
aggregate `A` on the storage of extension type `E`, under a separate aggregate ID such as
`vortex.min_max[vortex.timestamp]`. `register_storage_ordered` registers it by hand for `Date`,
`Time`, `Timestamp` and `Uuid`. Plain `min_max` and `is_sorted` do not consult it, and today only
tests ask for the composite IDs. Its results are storage values (EXT-A3).

The adapter goes away (X37). Each order aggregate checks `Ord` in its own `return_dtype` and runs on
storage itself:

1. Binding fails for an input without the claim, or for an input whose storage dtype the
   aggregate does not support. `AggregateDTypes::try_new` already calls `return_dtype`, so it needs
   no extension code of its own. `Uuid` passes both checks: it claims `Ord`, and `min_max` orders a
   `FixedSizeList` of non-nullable `u8` by its bytes.
2. For an input with the claim, the accumulator unwraps each batch and runs over the storage, so the
   storage kernels still apply. Plain `is_sorted` then gives a correct result for ordered types
   (EXT-A1).
3. A rule of the type can supply its own accumulator, for example `sum` over a decimal-like type
   (X20).

A partial state stays in storage form in memory. It gets the logical form when it leaves the
accumulator: `finalize`, `finalize_scalar` and `to_scalar` wrap, and `partial_from_scalar`
unwraps (X21). So consumers see the same partial dtype as today, and the cost is constant for each
partial. The composite IDs and `register_storage_ordered` go away in the same PR, which fixes
[#11243](https://github.com/vortex-data/vortex/issues/11243).

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
rule of the type wins over the storage path of the function.

Only the crate of a type adds rules for that type, as with the orphan rule of Rust. A crate that
owns neither the function nor the type defines its own function, or its own extension type over
the same storage. A session registry would bring back conflicts between crates.

PRs [#11091](https://github.com/vortex-data/vortex/pull/11091) and
[#11092](https://github.com/vortex-data/vortex/pull/11092) changed one reason for X20: `optimize`
without a session is deprecated, and parent-reduce rules now live in the session registry. One
reason still holds. Under X36, the `return_dtype` of `cast` must know whether a cast rule exists,
and `return_dtype` has no session. XQ10 records this, and RU1 must settle it first.

A cast exists only if a rule covers it (C1). This is the design of issue
[#8570](https://github.com/vortex-data/vortex/issues/8570), with rules in place of closures. Every
cast that the extension work adds is exact (C2). `Timestamp(ms)` to `Timestamp(ns)` fails on
overflow. `Timestamp(ns)` to `Timestamp(ms)` is not a cast: a named function with an explicit
rounding mode does it.

The only generic extension cast changes the top-level nullability (X30). A type cannot override it.
Nested nullability is part of the type, so a change of it is a rule of the type like any other
cast.

No `ext_wrap` expression exists. A rule that produces extension values builds the extension array
itself. It uses `ExtensionArray::new_unchecked` when legal input gives legal output, and
`ExtensionArray::try_new` otherwise. So no user expression can build a wrapper without a check.

Two explicit expressions remain:

- `ext_storage(x)` reads the storage. It exists today. It never checks values.
- `ext_storage_deep(x)` removes every extension layer at every depth. It is new. The shared storage
  helpers use it for operands that hold nested extension values, for example in a comparison of
  `struct{ts: Timestamp}` values. A function family adds it when it first needs it.

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

Rewrites respect one boundary (X33). Any function can move through a generic structural encoding,
because those encodings know nothing about the dtype. Only `ext_storage`, `ext_storage_deep`, the
storage path of a function backed by a claim, or a rule of the type can cross the `Extension`
wrapper. A
rewrite respects opaque types when it gives the same result for every extension type. Keeping the
result dtype is not enough: rewriting `ts_a < ts_b` to `ext_storage(ts_a) < ext_storage(ts_b)`
keeps `Bool`, but it needs the `Ord` claim.

The pushdown of functions through generic encodings is one shared mechanism, next to #10748
(X34):

- `Dict`, `Chunked` and `RunEnd` already push scalar functions into their values. `Masked` and
  `Sparse` get the same kind of rule, for functions that are elementwise, infallible and strict. A
  strict function returns null for a null input, so masked rows stay null.
- `ScalarFnVTable` gets a constant-fold hook, for example `reduce_constant`. The `Constant`
  encoding and the `Sparse` fill value use it.
- `ext_storage` owns only its constant hook and the unwrap rule `ext_storage(Extension(s))` to `s`.
  PR [#11056](https://github.com/vortex-data/vortex/pull/11056) added the constant case.

Type-specific encodings are a separate question. `DateTimeParts` can move under the wrapper as an
`i64` encoding, so that its compare kernel works on storage. The compressor then chooses it from
the extension metadata. This has low priority. Until then, compare dispatches the kernels of an
encoding before its storage path, so the existing `DateTimeParts` compare kernel keeps working with
no extra rule (X32). TY1 keeps that order and tests it.

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

The bloom filter of a zone map is an equality structure, so it needs `Eq`. Today it hashes the
storage of every extension type (EXT-A4). Writers build it only for types that hold `Eq`, and
readers ignore it for other types. Statistics code builds no expression node, so it checks
`DType::holds` directly (X38).

### Validation

Three kinds of validation exist:

- Dtype validation checks the metadata (X28) and the claims, once for each dtype. The type declares
  its storage (X27), so on read Vortex checks that the declared storage equals the stored storage
  (X29).
- Storage-shape validation checks that storage has the storage dtype, for arrays and scalars.
- Value validation (`validate_array`) checks that each non-null value is a legal value of the
  type.

Every extension array that Vortex builds holds only legal values (X7). `ExtensionArray::try_new`
checks the values in one batch with `validate_array`. Today it checks only the storage dtype, which
is a bug (EXT-V1). Kernels that keep legal values use a new `ExtensionArray::new_unchecked`, so they
pay nothing.

The default `validate_array` applies the existing scalar check of the type to each value. A type
can override it with a faster check, for example a range check on the minimum and maximum of a
`Time` column. `Timestamp` accepts every `i64` (X31), so it needs no value check.

File reads follow the existing pattern of the repository (X23). Every deserialize ends in
`Array::try_from_parts`, which runs only the structural `VTable::validate`. So the value check lives
in `ExtensionArray::try_new` only, never in `Extension::validate`. An optional strict read mode
checks values. Because files are trusted, building a scalar from an array always checks the value
and returns an error for an illegal one. No read panics.

The extension type adds no rule about null slots. The storage array must be a valid array of its
own encoding, which decides what a null slot can hold. The value check reads only non-null slots.

### Nullability and the storage declaration

An extension dtype's nullability always equals the top-level nullability of its storage (X25). The
type cannot choose it. Every extension type sits on canonical storage, and every canonical array
can hold nulls, so a nullability change never changes a value. `with_nullability` cannot fail.
Value checks never see a top-level null, and rows under a null slot are not checked. Storage of
`DType::Null` behaves like `DType::Null`, which cannot become non-nullable.

Nested nullability belongs to the type (X26). A nested level is nullable exactly when the type
accepts nulls at that level. These notes call this the total-domain rule. The spatial types follow
GeoArrow, which allows nulls only at the outer level, so all their nested levels are non-null. The
elements of `Vector` and `FixedShapeTensor`, the bytes of `Uuid`, the fields of `Interval`, and the
coordinates of `Point` and `Rect` are non-null too. An author who needs another domain defines a
new extension type.

A type declares its storage instead of validating it (X27, X28):

```rust
pub trait ExtVTable {
    /// The in-memory metadata. It holds every fact that decides the storage dtype.
    type Metadata;

    /// The storage dtype of the type. `ExtDType::try_new` applies the top-level nullability.
    fn storage_dtype(metadata: &Self::Metadata) -> DType;

    /// Checks the metadata. The default accepts every value.
    fn validate_metadata(metadata: &Self::Metadata) -> VortexResult<()> {
        Ok(())
    }

    /// Rebuilds the metadata on read, from the serialized bytes and the stored storage dtype.
    fn deserialize_metadata(
        &self,
        bytes: &[u8],
        stored_storage: &DType,
    ) -> VortexResult<Self::Metadata>;
}
```

`ExtDType::try_new(metadata, nullability)` calls `validate_metadata`, builds the declared storage
with the given nullability, and checks the claims. So a wrong top-level nullability cannot exist,
and `with_nullability` only asks the declaration again. The declaration takes no nullability
parameter, because no nested level depends on the top level.

The in-memory metadata differs from the serialized bytes, as with array vtables, whose
`deserialize` also receives the dtype. Some facts live only in the storage today: the element type
and dimension of `Vector`, the element type of `FixedShapeTensor`, and the dimension of `Point`
and `Rect`. They move into the in-memory metadata. `deserialize_metadata` reads them back from the
stored storage, so the file format does not change. `ForeignExtDType` holds the raw bytes and the
stored storage dtype, and it declares that storage. One hook checks the metadata, because types can
share a metadata type and accept different values. `TimeUnit` serves `Date`, `Time` and
`Duration`, but `Date` accepts only days and milliseconds.

Reads and imports are strict (X29). On read, the declared storage must equal the stored storage
exactly, including nested nullability. A file in another form is read only through a registered
read plugin. An Arrow import of a known type converts children that are marked nullable but hold
no nulls, and it fails when the data falls outside the domain of the type. An Arrow extension name
that Vortex does not know still imports as plain storage.

Storing nullability on the extension dtype, next to a non-nullable storage dtype, was the
alternative (NU2 in the design notes). It is rejected, because it splits one fact across two
places.

PR [#11012](https://github.com/vortex-data/vortex/pull/11012), merged on 2026-11-02, is the
interim fix until the declaration lands in phase 1. `with_nullability` no longer validates again,
and the documentation of `validate_dtype` states that its result must not depend on the top-level
nullability.

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

- Typing runs in `return_dtype`, so every path that builds a node must call it. The types enforce
  it: `BoundExpression` loses its public variants, and `Cast::new` becomes fallible when typing
  rejects casts. The design accepts these API changes for correctness (X8).
- A new function that reads values must remember to check the claim (X36). Code review and an
  opaque test type that claims nothing cover this risk.
- Each storage path must give the same result on every path: the whole array, one row, and constant
  inputs. The path-consistency helper of #11013 checks it.
- Old readers cannot decode `RunEnd` and `Sparse` over extension values. So writers keep the old
  shape of those two until an edition allows the new one.
- Engines must send extension literals for extension columns. U2 (#11016) removed the lone-extension
  compare rule, and #11067 made DataFusion send extension literals.
- Engines decide pushdown when they plan a query. They must read the claims, or a pushed-down
  filter on a type without the claim fails the query instead of running in the engine.
- Some conversions that work today through storage stop until a plugin or a converter covers them.
- The aggregate work coordinates with [#10769](https://github.com/vortex-data/vortex/pull/10769),
  which moves the aggregate implementations into a new `vortex-aggregates` crate, and with #10837.
  Statistics work in [#10661](https://github.com/vortex-data/vortex/pull/10661) (file statistics for
  a compound sort key, which replaces #10482) and
  [#10686](https://github.com/vortex-data/vortex/pull/10686) (comparisons proven true from
  statistics) meets the claims and the compare checks. Scalar functions move into a new
  `vortex-functions` crate in [#10762](https://github.com/vortex-data/vortex/pull/10762), which
  changes the files of the compare and `list_contains` checks.

## Open questions

The full list with recommendations is in [`../decisions.md`](../decisions.md). The questions that
stay open for extension types are:

- XQ1: which arrays the rule of X5 pulls out, and whether canonical form requires canonical storage.
  It has low priority.
- XQ8: what the Rust traits of `Scalar` mean for Vortex.
- XQ9: whether `ExtVTable::claims` takes the metadata or the whole extension dtype.
- XQ10: whether the rules of a type stay on `ExtVTable` after #11091 and #11092.
- Whether a claim beyond `Eq` and `Ord` is needed, for example for text functions on `Json`, or for
  comparisons of a refinement type with plain storage values.
