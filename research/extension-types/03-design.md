<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 3. Design

## Goal

Extension types behave like Rust newtypes. Structural operations always work. A semantic operation
works only where the extension type opts in or provides its own rule. No code path applies storage
semantics to an extension value without that opt-in.

## Rules

1. A semantic operation on an extension type is rejected unless the type opts in.
2. Structural operations are always allowed.
3. One place decides semantics: the binder, which runs when an expression or a scalar function
   array is built.
4. Below the binder, everything is storage. Kernels, encodings, statistics, pruning and compression
   see storage dtypes only.
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
cast(ts_ms, Timestamp(ns))       =>  ext_wrap(ext_storage(ts_ms) * 1_000_000, Timestamp(ns))
json_a = json_b                  =>  rejected, because Json derives no equality
```

The scalar path and the array path then evaluate the same expression, so the results of EXT-P1
cannot differ. Pushdown, pruning and compression keep working, because the rewritten expression is
an ordinary storage expression.

### Pairwise derives

A derive says that a semantic operation on two extension values is the same operation on their
storage values. It depends on the metadata of both operands:

```rust
/// Storage behavior that values of `lhs` and `rhs` share, like `#[derive]` on a newtype.
///
/// The default shares nothing, so every semantic operation on this type is rejected.
fn derives(lhs: &ExtDType<Self>, rhs: &ExtDType<Self>) -> Derives {
    Derives::NONE
}
```

`Derives` has two flags. `EQ` means that storage equality is value equality, which also makes the
storage hash valid. `ORD` means that storage order is value order.

| Type | Pair | Derives |
| --- | --- | --- |
| `Timestamp` | Same unit, and both with a time zone or both without | `EQ`, `ORD` |
| `Date`, `Time` | Same unit | `EQ`, `ORD` |
| `Decimal` | Same scale and same storage width | `EQ`, `ORD` |
| `Uuid` | Any | `EQ`, plus `ORD` if byte order is the chosen order |
| `Vector`, `FixedShapeTensor` | Same shape | `EQ` |
| `Json`, WKB, native geometry | Any | None |
| Unknown type | Any | None |

A derive is a preset rewrite. A type that needs something else writes its own rewrite. An example is
a comparison of two decimals with different scales, which needs a rescale first.

### Casts

The extension type owns its casts. The binder tries the cast hook of the source type, then the cast
hook of the target type, then rejects the cast. The hooks return expressions, not closures, so the
result can be pushed down and evaluated in the same way on both paths. This is the design in issue
[#8570](https://github.com/vortex-data/vortex/issues/8570), with expressions in place of closures.

Two explicit expressions cover everything else:

- `ext_storage(x)` reads the storage. It exists today.
- `ext_wrap(x, dtype)` builds an extension value from storage and checks it. It is new.

### One normal form

Only the wrapper carries an extension dtype. Generic encodings (`Constant`, `Dict`, `Chunked`) and
type-specific encodings (`DateTimeParts`) live under the wrapper and see the storage dtype. This
removes the two shapes of EXT-S1 and EXT-S2. The compressor chooses a type-specific encoding from
the extension metadata.

### Statistics

Zone maps and file statistics store storage statistics for extension columns. The rewrites in the
binder produce predicates on `ext_storage(col)`, and pruning uses the storage statistics for them.
`vortex-spatial` already prunes in this way, by hand.

### Validation

An extension type can define `validate_array`. The core calls it where data enters: `try_new`,
Arrow import, `ext_wrap`, and optionally file reads. Kernels that keep the invariants use an
unchecked constructor. Reads return errors and never panic. Validated arrays also remove the
per-row check that each read does today.

### Nullability

The short-term rule: `validate_dtype` sees the storage dtype without nullability, so a nullability
change never validates the type again. The long-term option: store nullability on the extension
dtype, like every other `DType` variant, and keep the storage dtype non-nullable.

### Authoring and boundaries

The core provides a typed array view, a typed accessor for native values, metadata helpers, and a
conformance test harness for `ExtVTable` implementations. DuckDB, DataFusion and Python get export
plugins like the Arrow registry. Then a new extension type does not need code in each engine crate.

## Costs and risks

- Binding must be mandatory. The array API builds scalar function arrays directly, so the rewrite
  must happen when such an array is built, not in an optional optimizer pass.
- `DateTimeParts` changes from a `Timestamp` encoding to an `i64` encoding. That is a file format
  change for the encoding, and it needs an edition.
- Engines must send extension literals for extension columns. Today the lone-extension compare rule
  can hide literals that have the storage type.
- After binding, plans show storage operations, so plans are harder to read.

## Open questions

- Open question: where does binding happen for the array API? One option is a rewrite when the
  array is built, with hooks on `ExtVTable`, so no session is necessary. The other option is a
  mandatory rewrite pass that also covers the row path.
- Open question: are rewrites owned only by the extension type? Or can a function owner register a
  rewrite for a type that it does not own?
- Open question: are operations on values with different metadata rejected, or coerced to a common
  type? Rust and the built-in `Decimal` reject them today. SQL coerces them.
- Open question: are `EQ` and `ORD` enough? A refinement over a storage dtype can need a third flag
  that allows comparisons with plain storage values.
- Open question: builders pad non-nullable children under null parents with a zero. Does the
  padding have to be a valid extension value?
- Open question: existing files store min and max for extension columns as extension scalars.
  Readers can convert them to storage values, but the switch needs a compatibility decision.
- Open question: is `validate_array` a no-op by default, or a check of each value through
  `validate_scalar_value`?
- Open question: is the nullability contract enough, or is the structural change worth its cost?
- Open question: does `Uuid` order by bytes, or have no order?
- Open question: `Scalar` implements `PartialEq` and `Hash` for use in Rust code. Can these stay
  structural (same dtype and same storage value) if they are documented as identity, not SQL
  equality? `PartialOrd` must follow the derives.
