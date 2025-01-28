# Vortex Data Types

A core principle of Vortex is that its data types (or `dtypes`) are _logical_ rather than _physical_.
This means that the dtype has no bearing on how the data is actually stored in memory, and is instead used to define
the domain of values an array may hold.

For example, a `u32` dtype represents an unsigned integer domain with values between `0` and `2^32 - 1`, even though
the underlying array may store values dictionary-encoded, run-length encoded (RLE), or in any other format!

This principle enables many of Vortex's advanced features. For example, performing compute directly on
compressed data.

:::{admonition} What is a schema?!
:class: tip
It is worth noting now that Vortex has no concept of a _schema_, instead preferring to use a struct dtype to represent
columnar data. This means you can write a Vortex file containing a single integer array just as well as writing one
with many columns.
:::

**Owned vs Viewed**

As with other possibly large recursive data structures in Vortex, dtypes can be either _owned_ or _viewed_.
Owned dtypes are heap-allocated, while viewed dtypes are lazily unwrapped from an underlying FlatBuffer representation.
This allows Vortex to efficiently load and work with very wide data types without needing to deserialize the full type
in memory.

## Logical Types

The following table lists the built-in dtypes in Vortex, each of which can be marked as either nullable or non-nullable.

| Name        | Domain                                      |
|-------------|---------------------------------------------|
| `Null`      | `null`                                      |
| `Bool`      | `true`, `false`                             |
| `Primitive` | See [Primitive](#primitive)                 |
| `UTF8`      | Variable length valid utf-8 encoded strings |
| `Binary`    | Arbitrary variable length bytes             |
| `Struct`    | See [Struct](#struct)                       |
| `List`      | See [List](#list)                           |
| `Extension` | See [Extension](#extension)                 |

:::{note}
There are additional logical types that Vortex does not yet support, for example fixed-length binary, utf-8, and list
types, as well as a map type. These may be added in future versions.
:::

### Primitive

Primitive dtypes are an enumeration of different fixed-width primitive values.

| Name  | Domain                  |
|-------|-------------------------|
| `I8`  | 8-bit signed integer    |
| `I16` | 16-bit signed integer   |
| `I32` | 32-bit signed integer   |
| `I64` | 64-bit signed integer   |
| `U8`  | 8-bit unsigned integer  |
| `U16` | 16-bit unsigned integer |
| `U32` | 32-bit unsigned integer |
| `U64` | 64-bit unsigned integer |
| `F16` | IEEE 754-2008 half      |
| `F32` | IEEE 754-1985 single    |
| `F64` | IEEE 754-1985 double    |

### Struct

A `Struct` dtype is an ordered collection of named fields, each of which has its own logical dtype.

### List

A `List` dtype has a single _element type_, itself a logical dtype, and represents an array of variable-length
sequences of elements of that type.

### Extension

An `Extension` dtype is a logical dtype with an `id`, a `storage` dtype, and a `metadata` field. The `id` and `metadata`
fields together may implicitly restrict the domain of values of the `storage` dtype.

For example, a `vortex.date` type is logically stored as a `U32` representing the number of days since the Unix epoch.

## Vs. Arrow

This section helps those familiar with Apache Arrow to quickly understand the differences vs. Vortex's dtypes.

* In Arrow, nullability is tied to a {obj}`pyarrow.Field` rather than the data type.
  Data types in Vortex instead always define explicit `nullability`.
* In Arrow, there are multiple ways to describe the same logical data type, for example {func}`pyarrow.string` and
  {func}`pyarrow.large_string` both represent UTF-8 values. In Vortex, there is a single `UTF8` dtype.
* In Arrow, encoded data is described with additional data types, for example {func}`pyarrow.dictionary`. In Vortex,
  encodings are a distinct concept from dtypes.
* In Arrow, date and time types are defined as first-class data types. In Vortex, these are represented as `Extension`
  dtypes since that can be composed of other more primitive logical dtypes.
* In Arrow, tables and record batches have a _schema_ that defines the types of the columns. Vortex makes no
  distinction between a data type and a schema. Columnar data can be stored with a struct dtype, and integer data can
  be stored equally well without a top-level struct. 