# Rust Cookbook

This guide provides practical, copy-pasteable examples for common operations in Vortex. For more in-depth information, see the [Rust API documentation](https://docs.rs/vortex).

## Topics Covered

✓ **Creating arrays** - Different types including VarBinArray for strings ([Creating Arrays](#creating-arrays))
✓ **Printing/debugging** - Display methods and inspection ([Inspecting and Debugging](#inspecting-and-debugging-arrays))
✓ **Iterating** - Index-based and `with_iterator` patterns ([Iterating Over Arrays](#iterating-over-arrays))
✓ **Accessing elements** - Getting individual values ([Accessing Elements](#accessing-elements))
✓ **Modifying arrays** - Arrays are immutable, create new ones ([Array Immutability](#array-immutability))
✓ **File I/O** - Reading and writing files ([File I/O](#file-io))
✓ **VarBinArray vs VarBinViewArray** - Comparison and when to use each ([String Arrays](#string-arrays))
✓ **Array trait vs concrete types** - Understanding the type system ([Core Concepts](#core-concepts))

## Quick Reference

### Array Creation

| Type | Code Example |
|------|--------------|
| Primitive integers | `buffer![1i32, 2, 3, 4, 5].into_array()` |
| Primitive floats | `buffer![1.0f64, 2.5, 3.14].into_array()` |
| Strings | `VarBinArray::from(vec!["hello", "world"]).into_array()` |
| Boolean | `BoolArray::from(vec![true, false, true]).into_array()` |
| Null array | `NullArray::new(5)` |
| Struct | `StructArray::from_fields(&[("name", array1), ("age", array2)])` |

### Common Operations

| Operation | Code |
|-----------|------|
| Get element | `array.scalar_at(index)` |
| Get length | `array.len()` |
| Check validity | `array.is_valid(index)` |
| Slice array | `array.slice(start..end)` |
| Print values | `array.display_values()` |
| Show structure | `array.display_tree()` |
| Get dtype | `array.dtype()` |
| Get encoding | `array.encoding().id()` |

## Core Concepts

### Array Trait vs Concrete Types

Understanding the difference between the `Array` trait and concrete array types like `VarBinArray` is fundamental to using Vortex effectively.

#### The Array Trait

`Array` is the core **trait** (interface) that defines what all array types can do:

```rust
pub trait Array: Send + Sync + Debug {
    fn len(&self) -> usize;
    fn dtype(&self) -> &DType;
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar>;
    // ... many other methods
}
```

**Key points:**
- Defines the common interface for all array types
- Provides methods like `len()`, `dtype()`, `scalar_at()`, `slice()`
- Enables polymorphism - write functions that work with any array type
- Usually used as `ArrayRef = Arc<dyn Array>` for type erasure

#### Concrete Array Types

Concrete types like `VarBinArray`, `PrimitiveArray`, `BoolArray` are specific implementations of the `Array` trait:

```rust
// Type hierarchy
Array (trait)
  ├── PrimitiveArray    // for numbers: i32, f64, etc.
  ├── BoolArray         // for booleans
  ├── VarBinArray       // for variable-length strings/binary
  ├── VarBinViewArray   // alternative string encoding
  ├── StructArray       // for struct/record data
  └── ... many more encodings
```

#### Practical Example

.. literalinclude:: ../../vortex/examples/core_concepts.rs
    :language: rust
    :dedent:
    :start-after: [array-trait-vs-concrete]
    :end-before: [array-trait-vs-concrete]

#### Why This Design?

1. **Polymorphism**: Write generic functions

.. literalinclude:: ../../vortex/examples/core_concepts.rs
    :language: rust
    :dedent:
    :start-after: [polymorphism]
    :end-before: [polymorphism]

2. **Multiple encodings for same data**:

.. literalinclude:: ../../vortex/examples/core_concepts.rs
    :language: rust
    :dedent:
    :start-after: [multiple-encodings]
    :end-before: [multiple-encodings]

3. **Heterogeneous collections**:

.. literalinclude:: ../../vortex/examples/core_concepts.rs
    :language: rust
    :dedent:
    :start-after: [heterogeneous-collections]
    :end-before: [heterogeneous-collections]

**Full example:** [core_concepts.rs](../../vortex/examples/core_concepts.rs)

## Creating Arrays

### Primitive Arrays

Create arrays of integers, floats, and other primitive types:

.. literalinclude:: ../../vortex/examples/basic_array_creation.rs
    :language: rust
    :dedent:
    :start-after: [primitive-int]
    :end-before: [primitive-int]

For arrays with null values:

.. literalinclude:: ../../vortex/examples/basic_array_creation.rs
    :language: rust
    :dedent:
    :start-after: [primitive-with-validity]
    :end-before: [primitive-with-validity]

**Full example:** [basic_array_creation.rs](../../vortex/examples/basic_array_creation.rs)

### String Arrays

Vortex has two encodings for variable-length strings:

#### VarBinArray vs VarBinViewArray

| Aspect | VarBinArray | VarBinViewArray |
|--------|-------------|-----------------|
| **Encoding** | Offset-based (like Arrow StringArray) | View-based (like Arrow StringViewArray) |
| **Structure** | Single data buffer + offsets | Multiple buffers + views |
| **Memory** | More compact for small strings | Better for mixed-size strings |
| **Operations** | Good for sequential access | Better for slicing, concatenation |
| **Canonical** | No | Yes (canonical for Utf8 dtype) |
| **When to use** | Input/output, small uniform strings | Processing, frequent slicing |

.. literalinclude:: ../../vortex/examples/string_arrays.rs
    :language: rust
    :dedent:
    :start-after: [array-vs-view]
    :end-before: [array-vs-view]

.. literalinclude:: ../../vortex/examples/string_arrays.rs
    :language: rust
    :dedent:
    :start-after: [varbin-from-vec]
    :end-before: [varbin-from-vec]

With null values:

.. literalinclude:: ../../vortex/examples/string_arrays.rs
    :language: rust
    :dedent:
    :start-after: [varbin-from-iter]
    :end-before: [varbin-from-iter]

**Full example:** [string_arrays.rs](../../vortex/examples/string_arrays.rs)

### Struct Arrays

Structs group multiple fields together:

.. literalinclude:: ../../vortex/examples/struct_arrays.rs
    :language: rust
    :dedent:
    :start-after: [struct-from-fields]
    :end-before: [struct-from-fields]

**Full example:** [struct_arrays.rs](../../vortex/examples/struct_arrays.rs)

### Advanced Array Types

Vortex supports many specialized array types beyond the basics:

#### Constant Arrays

Efficiently represent arrays where all values are the same:

.. literalinclude:: ../../vortex/examples/advanced_array_types.rs
    :language: rust
    :dedent:
    :start-after: [constant-array]
    :end-before: [constant-array]

#### List Arrays

Variable-length nested arrays (like `Vec<Vec<T>>`):

.. literalinclude:: ../../vortex/examples/advanced_array_types.rs
    :language: rust
    :dedent:
    :start-after: [list-array]
    :end-before: [list-array]

#### Fixed-Size List Arrays

Arrays where all lists have the same length:

.. literalinclude:: ../../vortex/examples/advanced_array_types.rs
    :language: rust
    :dedent:
    :start-after: [fixed-size-list]
    :end-before: [fixed-size-list]

#### DateTime Arrays

Temporal data with timezone support:

.. literalinclude:: ../../vortex/examples/advanced_array_types.rs
    :language: rust
    :dedent:
    :start-after: [datetime-array]
    :end-before: [datetime-array]

#### Decimal Arrays

Fixed-precision decimal numbers:

.. literalinclude:: ../../vortex/examples/advanced_array_types.rs
    :language: rust
    :dedent:
    :start-after: [decimal-array]
    :end-before: [decimal-array]

#### Extension Arrays

User-defined custom types with metadata:

.. literalinclude:: ../../vortex/examples/advanced_array_types.rs
    :language: rust
    :dedent:
    :start-after: [extension-array]
    :end-before: [extension-array]

**Full example:** [advanced_array_types.rs](../../vortex/examples/advanced_array_types.rs)

## Inspecting and Debugging Arrays

### Display Methods

Vortex provides several ways to print arrays:

.. literalinclude:: ../../vortex/examples/debug_printing.rs
    :language: rust
    :dedent:
    :start-after: [default-display]
    :end-before: [default-display]

.. literalinclude:: ../../vortex/examples/debug_printing.rs
    :language: rust
    :dedent:
    :start-after: [display-values]
    :end-before: [display-values]

To see the internal encoding structure:

.. literalinclude:: ../../vortex/examples/debug_printing.rs
    :language: rust
    :dedent:
    :start-after: [display-tree]
    :end-before: [display-tree]

**Full example:** [debug_printing.rs](../../vortex/examples/debug_printing.rs)

### Array Properties

Inspect array metadata:

.. literalinclude:: ../../vortex/examples/debug_printing.rs
    :language: rust
    :dedent:
    :start-after: [inspect-properties]
    :end-before: [inspect-properties]

## Accessing Elements

### Getting Individual Elements

Use `scalar_at(index)` to get elements:

.. literalinclude:: ../../vortex/examples/array_iteration.rs
    :language: rust
    :dedent:
    :start-after: [scalar-at]
    :end-before: [scalar-at]

### Extracting Typed Values

Convert scalars to Rust types:

.. literalinclude:: ../../vortex/examples/array_iteration.rs
    :language: rust
    :dedent:
    :start-after: [typed-values]
    :end-before: [typed-values]

### Handling Null Values

Check validity before accessing values:

.. literalinclude:: ../../vortex/examples/array_iteration.rs
    :language: rust
    :dedent:
    :start-after: [iterate-with-validity]
    :end-before: [iterate-with-validity]

**Full example:** [array_iteration.rs](../../vortex/examples/array_iteration.rs)

## Iterating Over Arrays

### Index-based Iteration

The simplest way to iterate over any array:

.. literalinclude:: ../../vortex/examples/array_iteration.rs
    :language: rust
    :dedent:
    :start-after: [scalar-at]
    :end-before: [scalar-at]

### Iterating String Arrays with ArrayAccessor

VarBinArray and VarBinViewArray implement `ArrayAccessor` for efficient iteration:

.. literalinclude:: ../../vortex/examples/array_iteration.rs
    :language: rust
    :dedent:
    :start-after: [array-accessor]
    :end-before: [array-accessor]

### Chunk-based Iteration

For ChunkedArrays, use `to_array_iterator()`:

.. literalinclude:: ../../vortex/examples/array_iteration.rs
    :language: rust
    :dedent:
    :start-after: [array-iterator]
    :end-before: [array-iterator]

**Full example:** [array_iteration.rs](../../vortex/examples/array_iteration.rs)

## Slicing and Transforming

### Array Immutability

**Important:** Vortex arrays are **immutable**. You cannot modify an existing array. Instead, you create new arrays:

.. literalinclude:: ../../vortex/examples/array_immutability.rs
    :language: rust
    :dedent:
    :start-after: [immutability-concept]
    :end-before: [immutability-concept]

#### Creating Modified Arrays

.. literalinclude:: ../../vortex/examples/array_immutability.rs
    :language: rust
    :dedent:
    :start-after: [modify-with-builder]
    :end-before: [modify-with-builder]

#### Compute Operations Return New Arrays

.. literalinclude:: ../../vortex/examples/array_immutability.rs
    :language: rust
    :dedent:
    :start-after: [compute-returns-new]
    :end-before: [compute-returns-new]

**Full example:** [array_immutability.rs](../../vortex/examples/array_immutability.rs)

### Slicing Arrays

Slicing is O(1) and doesn't copy data:

.. literalinclude:: ../../vortex/examples/array_iteration.rs
    :language: rust
    :dedent:
    :start-after: [slice-array]
    :end-before: [slice-array]

### Building New Arrays

Arrays are immutable. To create modified versions, use builders:

.. literalinclude:: ../../vortex/examples/array_iteration.rs
    :language: rust
    :dedent:
    :start-after: [modify-note]
    :end-before: [modify-note]

## File I/O

### Writing Files

Write arrays to disk with compression:

.. literalinclude:: ../../vortex/examples/file_io.rs
    :language: rust
    :dedent:
    :start-after: [basic-write]
    :end-before: [basic-write]

With custom compression:

.. literalinclude:: ../../vortex/examples/file_io.rs
    :language: rust
    :dedent:
    :start-after: [compressed-write]
    :end-before: [compressed-write]

### Reading Files

Read entire files:

.. literalinclude:: ../../vortex/examples/file_io.rs
    :language: rust
    :dedent:
    :start-after: [basic-read]
    :end-before: [basic-read]

With filtering (pushdown):

.. literalinclude:: ../../vortex/examples/file_io.rs
    :language: rust
    :dedent:
    :start-after: [filtered-read]
    :end-before: [filtered-read]

**Full example:** [file_io.rs](../../vortex/examples/file_io.rs)

## Key Concepts

### Encodings vs Data Types

- **DType** is the *logical* type (what the data represents)
- **Encoding** is the *physical* layout (how it's stored)

For example, a `DType::Primitive(i32)` array could be stored in many encodings:
- `PrimitiveEncoding`: Uncompressed array
- `DictEncoding`: Dictionary encoding for repeated values
- `FastLanesEncoding`: Compressed with FastLanes bitpacking

### Canonical Encodings

Each dtype has a canonical encoding that supports zero-copy conversion to/from Arrow:

| DType | Canonical Encoding |
|-------|-------------------|
| Bool | BoolArray |
| Primitive | PrimitiveArray |
| Utf8/Binary | VarBinViewArray |
| Struct | StructArray |
| List | ListArray |

Use `to_canonical()` to convert any array to its canonical form.

### Array vs ArrayView

Some types have both Array and View variants:

- **VarBinArray**: Owned array with offsets
- **VarBinViewArray**: Views into buffers (canonical for strings)

The View variant is generally more efficient for operations like slicing and concatenation.

### Validity

Arrays can have nullable values. The `Validity` enum specifies:

- `Validity::NonNullable`: Array has no nulls
- `Validity::Array(bool_array)`: Boolean mask indicating which values are valid
- `Validity::AllValid`: All values are valid (but type is nullable)

## Common Patterns

### Converting from Arrow

```rust
use arrow_array::RecordBatch;
use vortex::Array;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex_array::arrow::FromArrowArray;

let arrow_batch: RecordBatch = ...;
let dtype = DType::from_arrow(arrow_batch.schema());
let vortex_array = ArrayRef::from_arrow(arrow_batch, false);
```

### Converting to Arrow

```rust
use vortex::ToCanonical;

let vortex_array: ArrayRef = ...;
let canonical = vortex_array.to_canonical()?;
let arrow_array = canonical.into_arrow()?;
```

### Compressing Arrays

```rust
use vortex::compressor::BtrBlocksCompressor;

let array: ArrayRef = ...;
let compressed = BtrBlocksCompressor::default().compress(&array)?;
println!("Compression: {:.2}x", array.nbytes() as f64 / compressed.nbytes() as f64);
```

### Working with Statistics

```rust
use vortex::stats::{Stat, StatsProviderExt};

let array: ArrayRef = ...;

if let Some(min) = array.maybe_min() {
    println!("Min: {}", min);
}

if let Some(is_sorted) = array.maybe_stat(Stat::IsSorted) {
    println!("Is sorted: {}", is_sorted);
}
```

## Running Examples

All examples in this cookbook can be run with:

```bash
# Run a specific example
cargo run --example basic_array_creation
cargo run --example string_arrays
cargo run --example debug_printing
cargo run --example array_iteration
cargo run --example struct_arrays
cargo run --example file_io

# List all examples
cargo run --example
```

## See Also

- [Rust API Documentation](https://docs.rs/vortex)
- [Concepts: Arrays](../concepts/arrays.md)
- [Concepts: Data Types](../concepts/dtypes.md)
- [Writing an Encoding](writing-an-encoding.md)
