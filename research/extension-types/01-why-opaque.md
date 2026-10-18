<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 1. Why extension types must be opaque

## Terms

An **extension type** gives new meaning to the values of a **storage dtype**. `Timestamp` stores an
`i64`, and the extension type says that the `i64` counts time units since the Unix epoch. `Json`
stores `Utf8`, and the extension type says that the text is a JSON document.

A **structural operation** does not read values. Slice, take, filter, mask, validity, reading a row,
serialization and compression of the storage are structural. A **semantic operation** depends on
what the values mean. Equality, ordering, arithmetic, casts, aggregates over values, zero values and
export to other systems are semantic.

## The argument

A storage operation computes a result for the storage meaning. A semantic operation on an
extension type can run on storage. The result is then correct only if the storage result and the
extension result agree for every input. Agreement depends on the extension type, on the operation,
and often on the metadata:

| Extension type | Operation | What storage computes | Agrees |
| --- | --- | --- | --- |
| `Timestamp(ms)` | `<` with `Timestamp(ms)` | `i64` order | Yes |
| `Timestamp(ms)` | `<` with `Timestamp(ns)` | Order of counts in different units | No |
| `Decimal(10, 2)` | `<` with `Decimal(10, 2)` | Order of unscaled integers | Yes |
| `Decimal(10, 2)` | `*` with `Decimal(10, 2)` | A product with scale 4 | No, the result type changes |
| `Json` | `=` | Text equality | No, key order and spaces differ |
| WKB geometry | `=` | Byte equality | Only in one direction (see below) |
| Unit vector | `+` | Element sums | No, the sum is not a unit vector |
| `Uuid` | `<` | Byte order | A design decision |

`Decimal` is a built-in dtype, but it has the same shape: an integer storage value plus a scale. The
[Decimal notes](../decimal/README.md) use it as a test of this design.

The meaning lives in the extension type, so the core cannot know which rows of this table are
"yes". The core has two choices:

- **Transparent:** inherit storage behavior unless the type opts out. When an author forgets to opt
  out, the result is a silent wrong value.
- **Opaque:** inherit no semantic behavior unless the type opts in. When an author forgets to opt
  in, the result is an "unsupported" error.

Only the second failure is safe. It also covers types that nobody can audit: third-party types and
unknown types read from a file. Structural operations are the exception. They never read values, so
they are correct for every extension type and always stay available.

Opaque does not mean inaccessible. `ext_storage` remains the explicit way to read the storage, like
`.0` on a public newtype field.

## The newtype model

A Rust newtype has the same split:

| Rust | Vortex extension type |
| --- | --- |
| `struct Meters(f64)` is a separate type | Identity is the ID, the metadata and the storage dtype |
| `#[repr(transparent)]` costs nothing | One storage child, compressed as storage |
| `Vec<Meters>` can slice, filter and index | Structural operations work on the storage child |
| `m.0` reads the field | `ext_storage` |
| `Meters::new(x)` can check invariants | Building an extension value checks the rules of the type |
| Nothing is inherited without `#[derive]` | No semantic operation is inherited without an opt-in |
| `impl From<f64> for Meters` converts on purpose | The extension type owns its casts |

## Equality needs care

Within one extension dtype, equal storage values are equal extension values. The reverse holds only
when each value has exactly one storage form. This document calls that **canonical storage**.
`Timestamp` and `Decimal` storage are canonical. JSON text and WKB bytes are not, because one
document or geometry has many encodings.

So storage equality is safe for compression and for constant detection. `!=`, `DISTINCT`, `GROUP BY`
and joins are safe only when the type declares that its storage is canonical.

## Refinement types are a special case

A **refinement type** restricts a parent type to a subset of its values. Examples are UTF-8 text
over binary, and a unit vector over a vector. Every value of the refinement is a valid value of the
parent, so reads and predicates of the parent stay correct. Operations that produce new values
return the parent type, because the result can leave the subset. In the opaque model, a refinement
type opts in to all of the behavior of its parent and adds a value check.

## Prior art

- **Postgres** has both models. A base type made with `CREATE TYPE` gets ordering only from an
  operator class. A `DOMAIN` is a refinement: it inherits the operators of its base type and adds
  `CHECK` constraints. The `json` type stores text and has no `=` operator. The `jsonb` type stores
  a canonical form, and it defines `=`, ordering and hashing on that form.
- **Parquet** first wrote min and max statistics in the physical sort order. That order was wrong
  for some logical types, and the format later added a declared column order.
- **Haskell** newtypes derive nothing by default. `deriving newtype` and `DerivingVia` opt in.
  Type roles decide when a value can be converted between a newtype and its representation.
- The draft [RFC 0062](https://github.com/vortex-data/rfcs/blob/ngates/types/rfcs/0062-value-types-and-representations.md)
  states the same rule: "An extension does not inherit storage comparison, hashing, arithmetic, or
  parsing by default."
