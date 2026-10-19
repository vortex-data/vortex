<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 3. Design

## Goal

Extension types behave like Rust newtypes. Structural operations always work. A semantic operation
works only where the extension type opts in or provides its own rule. A function that declares that
it understands the type can also allow it. No code path applies storage semantics to an extension
value without that opt-in.

## Rules

1. A semantic operation on an extension type is rejected unless the type or a type-aware function
   opts in.
2. Structural operations are always allowed.
3. One place decides how generic operations treat extension values: the binder. It runs when an
   expression, a scalar function array or an aggregate is bound.
4. Below the binder, the kernels of generic semantic operations, statistics and pruning see
   storage dtypes only. A function that declares an extension type can see that type. Whether
   generic encodings carry extension dtypes depends on the physical form (open, see below).
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
| **Opaque, bind-time rewrite** | The binder rewrites semantic operations into storage expressions | **Proposed** |

The deciding property is the number of places that must understand extension semantics. With
flags, every consumer must read the flag correctly. Today, three groups of consumers give three
different answers to "is this type ordered?" (EXT-A1, EXT-B4):

- `compare` and `min_max` treat every extension type as ordered.
- `is_sorted` and `vortex-row` treat no extension type as ordered.
- DuckDB pushdown treats every extension type except `Uuid` as ordered.

With bind-time rewrites, only the binder needs to understand extension semantics.

## Proposal

### Bind-time rewrite

When an operation is bound and an operand has an extension type, the binder asks the extension type
how to answer. The answer is a rewrite into an expression over storage, or a rejection:

```text
ts_ms < lit(Timestamp(ms) 1500)  =>  ext_storage(ts_ms) < 1500i64
min(ts_ms)                       =>  ext_wrap(min(ext_storage(ts_ms)), Timestamp(ms))
cast(ts_ms, Timestamp(ns))       =>  ext_wrap(checked_mul(ext_storage(ts_ms), 1_000_000),
                                              Timestamp(ns))
json_a = json_b                  =>  rejected, because Json derives no equality
```

The scalar path and the array path then evaluate the same expression, so the results of EXT-P1
cannot differ. Pushdown, pruning and compression keep working, because the rewritten expression is
an ordinary storage expression.

### Derives for one dtype

A derive says that a semantic operation on values of one extension dtype is the same operation on
their storage values. It is a property of one dtype:

```rust
/// Storage behavior that values of `dtype` inherit, like `#[derive]` on a newtype.
///
/// The default inherits nothing, so every semantic operation on this dtype is rejected.
fn derives(dtype: &ExtDType<Self>) -> Derives {
    Derives::default()
}
```

`Derives` has two flags. `EQ` means that storage equality is value equality, which also makes the
storage hash valid. `ORD` means that storage order is value order.

A derive applies only when every operand in a derived position has the same extension dtype:

- A comparison needs two such operands, and `between` needs three.
- A unary aggregate, such as `min` or `is_sorted`, reads the flags of its one input.
- A struct or list that holds extension values inherits a flag only if every nested extension
  type declares it.

So `ts < 5i64` stays rejected, because only one side is an extension.

| Type | Derives |
| --- | --- |
| `Timestamp` | `EQ`, `ORD` |
| `Date`, `Time` | `EQ`, `ORD` |
| `Decimal`, if it follows this design | `EQ`, `ORD` |
| `Uuid` | `EQ`, plus `ORD` if byte order is the chosen order |
| `Vector`, `FixedShapeTensor` | `EQ` |
| `Json`, WKB, native geometry | None |
| Unknown type | None |

### Operations across dtypes

Two dtypes with different metadata are different dtypes, for example `Timestamp(ms)` and
`Timestamp(ns)`. A derive never applies across them, and the binder rejects such an operation by
default. A type can support one only through a rule that the type owns:

- A compatibility rule: the two dtypes compare by storage as they are. An example is two zoned
  timestamps with one unit.
- A coercion: a cast hook of the type converts one operand first.
- A rewrite of the whole call: for example, a comparison of two decimals with different scales
  rescales one side first.

### Function declarations

Each function declares how it uses an operand that contains an extension type:

- Structural: it moves or selects values, for example `fill_null` or `case_when`.
- Derive: it needs `EQ` or `ORD`, for example `=` or `<`.
- Type-aware: it implements the semantics of named extension types itself, for example
  `replace_time_zone` for `Timestamp`.
- Nothing: the default. The binder must rewrite or reject the call.

Only the function owner declares. The binder never infers a declaration from the input types that a
function accepts, because `Binary` accepts extension operands today.

### Casts

The extension type owns its casts. The binder tries the cast hook of the source type, then the cast
hook of the target type, then rejects the cast. The hooks return expressions, not closures, so the
result can be pushed down and evaluated in the same way on both paths. This is the design in issue
[#8570](https://github.com/vortex-data/vortex/issues/8570), with expressions in place of closures.

Two explicit expressions cover everything else:

- `ext_storage(x)` reads the storage. It exists today. It never checks values.
- `ext_wrap(x, dtype, mode)` builds an extension value from storage. It is new. A checked wrap
  validates the values. An unchecked wrap skips the check, and its result stays unvalidated.

A simplification can remove a wrap only if it keeps every error. `ext_storage(ext_wrap(y))`
simplifies to `y` only for an unchecked wrap. With a checked wrap, `y = 0` under a type that rejects
zero fails before the simplification and succeeds after it.

### Scalar identity

`Scalar` implements `PartialEq`, `Eq` and `Hash` as identity: the same dtype and the same stored
value. These traits serve Rust code, such as maps and tests, not SQL semantics. `PartialOrd`
returns `Equal` for identical scalars, and an order only for ordered dtypes. An extension dtype is
ordered only if it declares `ORD`. This keeps the Rust rule that `==` and `partial_cmp` agree.
Expression comparisons follow the binder, not these traits.

### Physical form (open)

Opacity does not decide where the wrapper sits relative to generic encodings such as `Constant`,
`Dict` and `Chunked`. The binder works with either placement.

| | Wrapper on top | Generic encodings on top |
| --- | --- | --- |
| Shapes of one column | One | Several, as today |
| Current code | The compressor uses it for non-constant columns, unless a type-specific encoding such as `DateTimeParts` wins | The compressor and `ExtensionConstantRule` use it for constants |
| Main cost | Reverse `ExtensionConstantRule`, and look through the wrapper wherever code looks for constants | Push `ext_storage` through `Masked` and `Sparse`, and keep extension code in each generic encoding |
| Wire | No new form for extension columns. `Decimal` translates on write | Memory and wire forms match |

These notes recommend the wrapper on top, after the binder lands. No maintainer decision exists.
[`agents/design-notes.md`](agents/design-notes.md), section 6, has the full comparison.

Type-specific encodings are a separate question. `DateTimeParts` moves under the wrapper as an
`i64` encoding, so its compare kernel works on storage. The compressor chooses it from the
extension metadata.

### Statistics

Zone maps and file statistics store storage statistics for extension columns. The rewrites in the
binder produce predicates on `ext_storage(col)`, and pruning uses the storage statistics for them.
`vortex-spatial` already prunes in this way, by hand.

### Validation

Three kinds of validation exist:

- Dtype validation checks the metadata and the storage dtype, once for each dtype.
- Storage-shape validation checks that storage has the storage dtype, for arrays and scalars.
- Value validation (`validate_array`) checks each value, at trust boundaries.

Each extension array is validated or unvalidated. Only a successful value check makes an array
validated, and structural operations keep that state. File reads without a value check give
unvalidated arrays. A read of a row from an unvalidated array checks the value, and returns an
error for an invalid value. Reads never panic.

The default `validate_array` checks each value with the existing scalar check of the type. It
skips rows under a null parent, because builders pad those rows.

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

- Binding must be mandatory. The array API builds scalar function arrays directly, so the rewrite
  must happen when such an array is built, not in an optional optimizer pass. Aggregates that code
  binds directly, such as zone-map statistics, need the binder too.
- `DateTimeParts` changes from a `Timestamp` encoding to an `i64` encoding. That is a file format
  change for the encoding, and it needs an edition.
- Engines must send extension literals for extension columns. Today the lone-extension compare rule
  can hide literals that have the storage type.
- After binding, plans show storage operations, so plans are harder to read.
- The new scalar order must land with or after the change that rejects `min` and `max` without
  `ORD`. Otherwise `min` and `max` can panic, for example in zone maps of `Json` columns. This
  risk comes from reading the code, not from a run.

## Open questions

- Open question: where does binding happen for the array API? One option is a rewrite when the
  array is built, with hooks on `ExtVTable`, so no session is necessary. The other option is a
  mandatory rewrite pass that also covers the row path.
- Open question: can a crate register a rewrite or a declaration for a function and a type that it
  does not own?
- Open question: is the wrapper on top of generic encodings? These notes recommend it. The
  alternative keeps generic encodings above the wrapper, as `ExtensionConstantRule` does for
  constants today.
- Open question: does Vortex coerce operations across dtypes or metadata, or do engines coerce
  first? Rust and the built-in `Decimal` reject such operations today. SQL coerces them.
- Open question: are `EQ` and `ORD` enough? A refinement over a storage dtype can need a third flag
  that allows comparisons with plain storage values.
- Open question: do file reads check values by default?
- Open question: existing files store min and max for extension columns as extension scalars.
  Readers can convert them to storage values, but the switch needs a compatibility decision.
- Open question: is the nullability contract enough, or does nullability move onto the extension
  dtype?
- Open question: does `Uuid` order by bytes, or have no order?
