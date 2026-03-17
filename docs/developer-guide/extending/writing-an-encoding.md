# Writing a Vortex Encoding

This guide walks through the process of implementing a custom array encoding in Vortex. An
encoding defines how data is physically stored in memory and how it can be decompressed back
to canonical form.

We use the **ZigZag** encoding as a running example throughout. ZigZag maps signed integers to
unsigned integers so that small-magnitude values (positive or negative) have small unsigned
representations, making them more compressible by downstream encodings like bit-packing.

## Prerequisites

Before writing an encoding, you should be familiar with:

- [Arrays](../../concepts/arrays.md) -- the Vortex array tree model
- [Vtables and Dispatch](../internals/vtables.md) -- how Vortex dispatches operations

## Project Structure

Each encoding lives in its own crate under `encodings/`. A typical layout:

```
encodings/my-encoding/
├── Cargo.toml
└── src/
    ├── lib.rs          # Public exports and module declarations
    ├── array.rs        # Array struct, VTable marker, VTable impl, OperationsVTable, ValidityVTable
    ├── compress.rs     # Encode and decode functions
    ├── compute/
    │   └── mod.rs      # FilterReduce, TakeExecute, and other compute implementations
    ├── rules.rs        # ParentRuleSet for optimizer reduce rules
    ├── kernel.rs       # ParentKernelSet for execute-parent kernels
    └── slice.rs        # SliceReduce implementation
```

### Cargo.toml

```toml
[package]
name = "vortex-my-encoding"
# ... workspace metadata ...

[dependencies]
vortex-array = { workspace = true }
vortex-buffer = { workspace = true }
vortex-error = { workspace = true }
vortex-mask = { workspace = true }
vortex-session = { workspace = true }

[dev-dependencies]
rstest = { workspace = true }
vortex-array = { workspace = true, features = ["_test-harness"] }

[lints]
workspace = true
```

### lib.rs

```rust
pub use array::*;
pub use compress::*;

mod array;
mod compress;
mod compute;
mod kernel;
mod rules;
mod slice;
```

## Step 1: Define the Array Struct and VTable Marker

Every encoding needs three things in `array.rs`:

1. A **VTable marker struct** -- a zero-sized type that serves as the type parameter for the
   vtable system.
2. An **ArrayId** -- a unique string identifier for your encoding.
3. An **Array struct** -- holds the encoding-specific data.

```rust
use vortex_array::stats::ArrayStats;
use vortex_array::vtable::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;

/// VTable marker type for ZigZag encoding.
#[derive(Debug)]
pub struct ZigZag;

impl ZigZag {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.zigzag");
}

/// A ZigZag-encoded array of signed integers stored as unsigned integers.
#[derive(Clone, Debug)]
pub struct ZigZagArray {
    dtype: DType,
    encoded: ArrayRef,
    stats_set: ArrayStats,
}
```

Key rules:

- The array struct must derive `Clone` and `Debug`.
- Always include a `stats_set: ArrayStats` field.
- Child arrays are stored as `ArrayRef`.
- Data buffers are stored as `BufferHandle` (from `vortex_array::buffer`).

### Constructors

Provide `try_new` (fallible) and optionally `new` (panicking) constructors that validate
invariants:

```rust
impl ZigZagArray {
    pub fn try_new(encoded: ArrayRef) -> VortexResult<Self> {
        let encoded_dtype = encoded.dtype().clone();
        if !encoded_dtype.is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "unsigned int", encoded_dtype);
        }

        let dtype = DType::from(PType::try_from(&encoded_dtype)?.to_signed())
            .with_nullability(encoded_dtype.nullability());

        Ok(Self {
            dtype,
            encoded,
            stats_set: Default::default(),
        })
    }

    /// Accessor for the encoded child array.
    pub fn encoded(&self) -> &ArrayRef {
        &self.encoded
    }
}
```

## Step 2: Invoke the vtable! Macro

The `vtable!` macro generates the `AsRef<dyn DynArray>`, `Deref`, `IntoArray`, and
`From<...> for ArrayRef` implementations that connect your array struct to the Vortex
type-erased array system:

```rust
use vortex_array::vtable;

vtable!(ZigZag);
```

Place this near the top of `array.rs`, before the `VTable` impl.

## Step 3: Implement the VTable Trait

The `VTable` trait is the core of your encoding. It tells Vortex how to inspect, serialize,
deserialize, and execute your array.

```rust
use vortex_array::vtable::{VTable, OperationsVTable, ValidityVTableFromChild, ValidityChild};
use vortex_array::{EmptyMetadata, ExecutionCtx, ExecutionStep, IntoArray, Precision};
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::StatsSetRef;

impl VTable for ZigZag {
    type Array = ZigZagArray;
    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    // ... method implementations follow ...
}
```

### Associated Types

| Type                | Purpose                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `Array`             | Your concrete array struct                                              |
| `Metadata`          | Serializable metadata. Use `EmptyMetadata` if none needed, or `ProstMetadata<T>` for structured data |
| `OperationsVTable`  | Type implementing `OperationsVTable` (usually `Self`)                   |
| `ValidityVTable`    | How nullability is handled (see [Validity](#validity) below)            |

### Identity and Shape

```rust
fn id(_array: &Self::Array) -> ArrayId {
    Self::ID
}

fn len(array: &ZigZagArray) -> usize {
    array.encoded.len()
}

fn dtype(array: &ZigZagArray) -> &DType {
    &array.dtype
}

fn stats(array: &ZigZagArray) -> StatsSetRef<'_> {
    array.stats_set.to_ref(array.as_ref())
}
```

### Equality and Hashing

Used for array comparison and deduplication:

```rust
fn array_hash<H: Hasher>(array: &ZigZagArray, state: &mut H, precision: Precision) {
    array.dtype.hash(state);
    array.encoded.array_hash(state, precision);
}

fn array_eq(array: &ZigZagArray, other: &ZigZagArray, precision: Precision) -> bool {
    array.dtype == other.dtype && array.encoded.array_eq(&other.encoded, precision)
}
```

### Buffers and Children

Declare how many raw data buffers and child arrays your encoding holds. Vortex uses these for
serialization, traversal, and memory accounting.

```rust
// ZigZag has no direct buffers -- its data lives in the child array.
fn nbuffers(_array: &ZigZagArray) -> usize {
    0
}

fn buffer(_array: &ZigZagArray, idx: usize) -> BufferHandle {
    vortex_panic!("ZigZagArray buffer index {idx} out of bounds")
}

fn buffer_name(_array: &ZigZagArray, idx: usize) -> Option<String> {
    vortex_panic!("ZigZagArray buffer_name index {idx} out of bounds")
}

// ZigZag has one child: the encoded unsigned integer array.
fn nchildren(_array: &ZigZagArray) -> usize {
    1
}

fn child(array: &ZigZagArray, idx: usize) -> ArrayRef {
    match idx {
        0 => array.encoded().clone(),
        _ => vortex_panic!("ZigZagArray child index {idx} out of bounds"),
    }
}

fn child_name(_array: &ZigZagArray, idx: usize) -> String {
    match idx {
        0 => "encoded".to_string(),
        _ => vortex_panic!("ZigZagArray child_name index {idx} out of bounds"),
    }
}
```

### Metadata Serialization

Metadata captures any encoding-specific parameters that are not children or buffers. For
simple encodings, use `EmptyMetadata`:

```rust
fn metadata(_array: &ZigZagArray) -> VortexResult<Self::Metadata> {
    Ok(EmptyMetadata)
}

fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
    Ok(Some(vec![]))
}

fn deserialize(
    _bytes: &[u8], _dtype: &DType, _len: usize,
    _buffers: &[BufferHandle], _session: &VortexSession,
) -> VortexResult<Self::Metadata> {
    Ok(EmptyMetadata)
}
```

For encodings with structured metadata, derive `prost::Message` and wrap it in
`ProstMetadata<T>`:

```rust
#[derive(Clone, prost::Message)]
pub struct RunEndMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub ends_ptype: i32,
    #[prost(uint64, tag = "2")]
    pub num_runs: u64,
    #[prost(uint64, tag = "3")]
    pub offset: u64,
}

// In the VTable impl:
type Metadata = ProstMetadata<RunEndMetadata>;
```

### Build and With-Children

`build` reconstructs the array from deserialized components. `with_children` replaces
children in-place (used by the optimizer).

```rust
fn build(
    dtype: &DType, len: usize, _metadata: &Self::Metadata,
    _buffers: &[BufferHandle], children: &dyn ArrayChildren,
) -> VortexResult<ZigZagArray> {
    if children.len() != 1 {
        vortex_bail!("Expected 1 child, got {}", children.len());
    }
    let ptype = PType::try_from(dtype)?;
    let encoded_type = DType::Primitive(ptype.to_unsigned(), dtype.nullability());
    let encoded = children.get(0, &encoded_type, len)?;
    ZigZagArray::try_new(encoded)
}

fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
    vortex_ensure!(children.len() == 1, "ZigZagArray expects 1 child, got {}", children.len());
    array.encoded = children.into_iter().next().vortex_expect("checked");
    Ok(())
}
```

### Execute (Canonicalize)

The `execute` method decompresses your encoding toward canonical form. This is how Vortex
resolves compressed data into a form that compute kernels can operate on.

```rust
fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
    Ok(ExecutionStep::Done(
        zigzag_decode(array.encoded().clone().execute(ctx)?).into_array(),
    ))
}
```

### Reduce Parent and Execute Parent

These methods integrate with the Vortex optimizer. They allow operations on parent arrays
(like `SliceArray` or `FilterArray`) to be pushed down into your encoding, avoiding
unnecessary decompression.

```rust
fn reduce_parent(
    array: &Self::Array, parent: &ArrayRef, child_idx: usize,
) -> VortexResult<Option<ArrayRef>> {
    RULES.evaluate(array, parent, child_idx)
}

fn execute_parent(
    array: &Self::Array, parent: &ArrayRef, child_idx: usize, ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    PARENT_KERNELS.execute(array, parent, child_idx, ctx)
}
```

## Step 4: Implement OperationsVTable

The `OperationsVTable` provides `scalar_at`, which extracts a single scalar value from your
array at a given index:

```rust
impl OperationsVTable<ZigZag> for ZigZag {
    fn scalar_at(array: &ZigZagArray, index: usize) -> VortexResult<Scalar> {
        let scalar = array.encoded().scalar_at(index)?;
        if scalar.is_null() {
            return scalar.primitive_reinterpret_cast(array.ptype());
        }
        // Decode the unsigned value back to signed
        let pscalar = scalar.as_primitive();
        Ok(match_each_unsigned_integer_ptype!(pscalar.ptype(), |P| {
            Scalar::primitive(
                <<P as ZigZagEncoded>::Int>::decode(
                    pscalar.typed_value::<P>().vortex_expect("zigzag corruption"),
                ),
                array.dtype().nullability(),
            )
        }))
    }
}
```

(validity)=
## Step 5: Implement Validity

Vortex provides several helpers for implementing `ValidityVTable`, depending on how your
encoding represents nulls:

| Helper                                | When to use                                      |
|---------------------------------------|--------------------------------------------------|
| `ValidityVTableFromChild`             | Nullability is tracked by a child array           |
| `ValidityVTableFromValidityHelper`    | You store a `Validity` field directly             |
| `ValidityVTableFromValiditySliceHelper` | You store a `Validity` field and support slicing |

For ZigZag, nullability is inherited from the encoded child:

```rust
use vortex_array::vtable::{ValidityChild, ValidityVTableFromChild};

// Set in VTable impl:
// type ValidityVTable = ValidityVTableFromChild;

impl ValidityChild<ZigZag> for ZigZag {
    fn validity_child(array: &ZigZagArray) -> &ArrayRef {
        array.encoded()
    }
}
```

If your encoding computes validity directly (e.g. `ConstantArray`), implement
`ValidityVTable` yourself:

```rust
impl ValidityVTable<Constant> for Constant {
    fn validity(array: &ConstantArray) -> VortexResult<Validity> {
        Ok(if array.scalar().is_null() {
            Validity::AllInvalid
        } else {
            Validity::AllValid
        })
    }
}
```

## Step 6: Write Encode and Decode Functions

In `compress.rs`, implement the encoding and decoding logic:

```rust
pub fn zigzag_encode(parray: PrimitiveArray) -> VortexResult<ZigZagArray> {
    let validity = parray.validity().clone();
    let encoded = match parray.ptype() {
        PType::I8  => zigzag_encode_primitive::<i8>(parray.into_buffer_mut(), validity),
        PType::I16 => zigzag_encode_primitive::<i16>(parray.into_buffer_mut(), validity),
        PType::I32 => zigzag_encode_primitive::<i32>(parray.into_buffer_mut(), validity),
        PType::I64 => zigzag_encode_primitive::<i64>(parray.into_buffer_mut(), validity),
        _ => vortex_bail!("ZigZag can only encode signed integers, got {}", parray.ptype()),
    };
    ZigZagArray::try_new(encoded.into_array())
}

pub fn zigzag_decode(parray: PrimitiveArray) -> PrimitiveArray {
    let validity = parray.validity().clone();
    match parray.ptype() {
        PType::U8  => zigzag_decode_primitive::<i8>(parray.into_buffer_mut(), validity),
        PType::U16 => zigzag_decode_primitive::<i16>(parray.into_buffer_mut(), validity),
        PType::U32 => zigzag_decode_primitive::<i32>(parray.into_buffer_mut(), validity),
        PType::U64 => zigzag_decode_primitive::<i64>(parray.into_buffer_mut(), validity),
        _ => vortex_panic!("ZigZag can only decode unsigned integers, got {}", parray.ptype()),
    }
}
```

## Step 7: Implement Compute Functions

Compute functions let Vortex perform operations directly on your compressed representation
without full decompression. Implement as many as make sense for your encoding.

### Filter

```rust
// compute/mod.rs
impl FilterReduce for ZigZag {
    fn filter(array: &ZigZagArray, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let encoded = array.encoded().filter(mask.clone())?;
        Ok(Some(ZigZagArray::try_new(encoded)?.into_array()))
    }
}
```

### Take

```rust
impl TakeExecute for ZigZag {
    fn take(
        array: &ZigZagArray, indices: &ArrayRef, _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let encoded = array.encoded().take(indices.to_array())?;
        Ok(Some(ZigZagArray::try_new(encoded)?.into_array()))
    }
}
```

### Slice

```rust
// slice.rs
impl SliceReduce for ZigZag {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(ZigZagArray::new(array.encoded().slice(range)?).into_array()))
    }
}
```

### Mask

```rust
impl MaskReduce for ZigZag {
    fn mask(array: &ZigZagArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let masked_encoded = MaskExpr.try_new_array(
            array.encoded().len(), EmptyOptions,
            [array.encoded().clone(), mask.clone()],
        )?;
        Ok(Some(ZigZagArray::try_new(masked_encoded)?.into_array()))
    }
}
```

The general pattern: push the operation down into the child array, then re-wrap the result in
your encoding. Return `Ok(None)` if the operation cannot be performed on your compressed
representation, and Vortex will fall back to decompressing first.

## Step 8: Define Optimizer Rules and Kernels

### Rules (rules.rs)

Rules tell the optimizer how to push parent operations (slice, filter, cast, mask) down into
your encoding:

```rust
use vortex_array::arrays::filter::FilterReduceAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) static RULES: ParentRuleSet<ZigZag> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(ZigZag)),
    ParentRuleSet::lift(&FilterReduceAdaptor(ZigZag)),
    ParentRuleSet::lift(&MaskReduceAdaptor(ZigZag)),
    ParentRuleSet::lift(&SliceReduceAdaptor(ZigZag)),
]);
```

### Kernels (kernel.rs)

Kernels handle execution of parent operations on your encoding:

```rust
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

pub(crate) const PARENT_KERNELS: ParentKernelSet<ZigZag> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(ZigZag))]);
```

Each adaptor wraps the corresponding compute trait implementation (e.g. `TakeExecuteAdaptor`
wraps `TakeExecute`) so the kernel system can dispatch to it.

## Step 9: Register the Encoding

External encodings (those outside `vortex-array`) must be registered with the Vortex session
so the deserializer can reconstruct them from serialized data:

```rust
// lib.rs
use vortex_session::VortexSession;

pub fn initialize(session: &mut VortexSession) {
    session.arrays().register(MyEncoding::ID, MyEncoding);
}
```

Built-in encodings (inside `vortex-array`) are registered automatically and do not need this
step.

## Step 10: Write Tests

Use `rstest` for parameterized tests. Vortex provides conformance test suites that verify
your encoding behaves correctly for standard operations.

```rust
#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::{ArrayRef, IntoArray, assert_arrays_eq};
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::{ZigZagArray, zigzag_encode};

    #[test]
    fn roundtrip() -> VortexResult<()> {
        let original = PrimitiveArray::from_iter([-100i32, 0, 100]);
        let encoded = zigzag_encode(original.clone())?;
        assert_arrays_eq!(encoded.to_primitive(), original);
        Ok(())
    }

    #[rstest]
    #[case::i32(buffer![-189i32, -160, 1, 42, -73].into_array())]
    #[case::i64(buffer![1000i64, -2000, 3000, -4000, 5000].into_array())]
    fn test_take_conformance(#[case] array: ArrayRef) -> VortexResult<()> {
        use vortex_array::compute::conformance::take::test_take_conformance;
        let zigzag = zigzag_encode(array.to_primitive())?;
        test_take_conformance(&zigzag.into_array());
        Ok(())
    }

    #[rstest]
    #[case::basic(zigzag_encode(PrimitiveArray::from_iter([-128i8, -1, 0, 1, 127])).unwrap())]
    #[case::large(zigzag_encode(PrimitiveArray::from_iter(-500..500)).unwrap())]
    fn test_consistency(#[case] array: ZigZagArray) {
        test_array_consistency(&array.into_array());
    }
}
```

Available conformance test suites:

| Function                     | What it tests                                       |
|------------------------------|-----------------------------------------------------|
| `test_array_consistency`     | Serialization roundtrip, scalar_at, metadata        |
| `test_filter_conformance`    | Filter with various mask patterns                   |
| `test_take_conformance`      | Take with various index patterns                    |
| `test_mask_conformance`      | Null masking                                        |
| `test_binary_numeric_array`  | Binary numeric operations (add, subtract, etc.)     |

Run your tests with:

```bash
cargo test -p vortex-my-encoding
```

## Step 11: Integrate with the BtrBlocks Compressor

The BtrBlocks compressor (`vortex-btrblocks`) is Vortex's cascading compression framework. It
adaptively selects the best encoding for each data block by sampling ~1% of the data,
estimating compression ratios, and choosing the scheme with the highest ratio. To make your
encoding available for automatic compression, you need to implement a **compression scheme**.

### How the Compressor Works

1. Data arrives as a canonical array (e.g. `PrimitiveArray`).
2. The compressor generates statistics about the data.
3. Each registered scheme estimates its compression ratio (via sampling or heuristics).
4. The scheme with the best ratio above 1.0 is selected.
5. The selected scheme compresses the data, optionally cascading into further compression
   of its children (up to 3 levels deep).

### Scheme Types

Schemes are specialized by data type. Each has its own code enum and trait:

| Data Type | Trait           | Code Enum    | Schemes List        |
|-----------|-----------------|--------------|---------------------|
| Integer   | `IntegerScheme` | `IntCode`    | `ALL_INT_SCHEMES`   |
| Float     | `FloatScheme`   | `FloatCode`  | `ALL_FLOAT_SCHEMES` |
| String    | `StringScheme`  | `StringCode` | `ALL_STRING_SCHEMES`|

### Step-by-Step: Adding an Integer Scheme

#### 1. Add a variant to the code enum

In `vortex-btrblocks/src/compressor/integer/mod.rs`, add your encoding to `IntCode`:

```rust
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Sequence, Ord, PartialOrd)]
pub enum IntCode {
    // ... existing variants ...
    /// My encoding - brief description.
    MyEncoding,
}
```

#### 2. Define a scheme struct

```rust
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MyEncodingScheme;
```

#### 3. Implement the `Scheme` trait

The `Scheme` trait has two key methods:

- `expected_compression_ratio` -- estimates how well this scheme compresses the data.
  The default implementation samples ~1% of the data and compresses it. Override this
  if you have a cheaper heuristic.
- `compress` -- applies the encoding and optionally cascades into further compression.

Here is the ZigZag scheme as an example:

```rust
impl Scheme for ZigZagScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        IntCode::ZigZag
    }

    fn expected_compression_ratio(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        ctx: CompressorContext,
        excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // ZigZag is only useful when cascaded with another encoding
        if ctx.allowed_cascading == 0 {
            return Ok(0.0);
        }
        if stats.value_count == 0 {
            return Ok(0.0);
        }
        // ZigZag is only useful when there are negative values
        if !stats.typed.min_is_negative() {
            return Ok(0.0);
        }
        // Fall back to sampling-based estimation
        self.estimate_compression_ratio_with_sampling(compressor, stats, ctx, excludes)
    }

    fn compress(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        ctx: CompressorContext,
        excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        let zag = zigzag_encode(stats.src.clone())?;
        let encoded = zag.encoded().to_primitive();

        // Exclude schemes that should not be applied after ZigZag
        let mut new_excludes = vec![
            ZigZagScheme.code(),
            DictScheme.code(),
            RunEndScheme.code(),
            SparseScheme.code(),
        ];
        new_excludes.extend_from_slice(excludes);

        // Recursively compress the inner values
        let compressed = compressor.compress_canonical(
            Canonical::Primitive(encoded),
            ctx.descend(),
            Excludes::int_only(&new_excludes),
        )?;

        Ok(ZigZagArray::try_new(compressed)?.into_array())
    }
}
```

Key patterns:

- Return `0.0` from `expected_compression_ratio` if the scheme is not applicable to the data.
- Use `ctx.allowed_cascading` to check if further cascading is allowed.
- Use `ctx.descend()` to reduce the remaining cascade depth.
- Add your own scheme code to the excludes list to prevent infinite recursion.

#### 4. Register the scheme

Add your scheme to the `ALL_INT_SCHEMES` array:

```rust
pub const ALL_INT_SCHEMES: &[&dyn IntegerScheme] = &[
    &ConstantScheme,
    &FORScheme,
    &ZigZagScheme,
    &BitPackingScheme,
    // ... other schemes ...
    &MyEncodingScheme,  // Add your scheme here
];
```

The compressor will now automatically consider your encoding when compressing integer data.

## Step 12: Add a Backward-Compatibility Fixture

Vortex maintains backward-compatibility testing infrastructure in the `vortex-compat` crate
(`vortex-test/compat-gen/`). This ensures that files written by older versions can always be
read by newer versions. When adding a new encoding, you should add a compatibility fixture.

### The Fixture Contract

- **Immutable data.** Once a fixture's `build()` method is defined, its output must never
  change. Every version that includes the fixture must produce byte-for-byte identical
  logical arrays.
- **New capabilities get new files.** Never modify an existing fixture. Add a new one.
- **Additive-only.** The fixture list only grows; fixtures are never removed.
- **Deterministic.** `build()` must produce identical output across all runs and versions.

### Adding a Fixture

Create a new file in `vortex-test/compat-gen/src/fixtures/arrays/synthetic/encodings/`:

```rust
// my_encoding.rs
use vortex::array::ArrayRef;
use vortex::array::vtable::ArrayId;
use vortex::error::VortexResult;

use crate::fixtures::ArrayFixture;
use super::N;  // Standard fixture size: 1024 rows

pub struct MyEncodingFixture;

impl ArrayFixture for MyEncodingFixture {
    fn name(&self) -> &str {
        "my_encoding.vortex"
    }

    fn description(&self) -> &str {
        "Data patterns designed to exercise MyEncoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![MyEncoding::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        // Create deterministic test data that exercises your encoding.
        // Wrap multiple columns in a StructArray to test different patterns.
        // IMPORTANT: This output must never change once defined.
        todo!()
    }
}
```

Then register it in `vortex-test/compat-gen/src/fixtures/arrays/synthetic/encodings/mod.rs`:

```rust
mod my_encoding;

pub fn fixtures() -> Vec<Box<dyn ArrayFixture>> {
    vec![
        // ... existing fixtures ...
        Box::new(my_encoding::MyEncodingFixture),
    ]
}
```

### Testing Locally

```bash
# Generate fixtures
cargo run -p vortex-compat --release --bin compat-gen -- \
  --version 0.99.0 --output /tmp/compat-fixtures/v0.99.0/

# Validate fixtures round-trip correctly
cargo run -p vortex-compat --release --bin compat-validate -- \
  --fixtures-dir /tmp/compat-fixtures/
```

### CI Integration

- **Upload workflow** (`compat-gen-upload.yml`): Triggered manually per release. Generates
  fixtures and uploads to S3 with manifest merging.
- **Weekly validation** (`compat-test-weekly.yml`): Runs every Monday, validating all
  historical versions listed in `versions.json`.

See [the compat-gen README](https://github.com/spiraldb/vortex/blob/main/vortex-test/compat-gen/README.md)
for the full workflow.

## Step 13: Add Compression Benchmarks

Vortex uses [Divan](https://github.com/nvzqz/divan) (via `codspeed-divan-compat`) for
benchmarking. You should add compress and decompress benchmarks for your encoding in two
places.

### Encoding-Specific Benchmarks

Create a benchmark file in your encoding crate:

```
encodings/my-encoding/benches/my_compress.rs
```

```rust
#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::compute::warm_up_vtables;

fn main() {
    warm_up_vtables();
    divan::main();
}

const BENCH_ARGS: &[(usize, usize)] = &[
    (1_000, 4),
    (10_000, 8),
    (100_000, 16),
];

#[divan::bench(args = BENCH_ARGS)]
fn compress(bencher: Bencher, (length, param): (usize, usize)) {
    let array = create_test_data(length, param);

    bencher
        .with_inputs(|| &array)
        .bench_refs(|a| my_encode(a).unwrap())
}

#[divan::bench(args = BENCH_ARGS)]
fn decompress(bencher: Bencher, (length, param): (usize, usize)) {
    let array = create_test_data(length, param);
    let compressed = my_encode(&array).unwrap();

    bencher
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical().unwrap())
}
```

Register the benchmark in your encoding's `Cargo.toml`:

```toml
[dev-dependencies]
divan = { workspace = true }
vortex-array = { workspace = true, features = ["_test-harness"] }

[[bench]]
name = "my_compress"
harness = false
```

### Cross-Encoding Throughput Benchmarks

Add compress and decompress benchmarks to `vortex/benches/single_encoding_throughput.rs`,
which collects all encodings in one place for comparison. Follow the existing pattern:

```rust
#[divan::bench(name = "my_encoding_compress_u32")]
fn bench_my_encoding_compress(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| uint_array.clone())
        .bench_values(|a| my_encode(a).unwrap());
}

#[divan::bench(name = "my_encoding_decompress_u32")]
fn bench_my_encoding_decompress(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();
    let compressed = my_encode(uint_array).unwrap();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}
```

### Divan Benchmark Patterns

**Type-parameterized benchmarks:**
```rust
#[divan::bench(types = [f32, f64], args = BENCH_ARGS)]
fn compress<T: NativePType>(bencher: Bencher, (n, fraction): (usize, f64)) { ... }
```

**Byte throughput counters** (skipped in Codspeed CI):
```rust
#[cfg(not(codspeed))]
use divan::counter::BytesCount;

with_byte_counter(bencher, num_bytes)
    .with_inputs(|| data.clone())
    .bench_values(|d| compress(d));
```

### Running Benchmarks

```bash
# Run benchmarks for a specific encoding
cargo bench -p vortex-my-encoding

# Run cross-encoding throughput benchmarks
cargo bench -p vortex -- my_encoding

# Run all benchmarks
cargo bench
```

## Checklist

When implementing a new encoding, make sure you have:

- [ ] Defined the VTable marker struct with an `ArrayId`
- [ ] Defined the Array struct with `Clone`, `Debug`, and `stats_set: ArrayStats`
- [ ] Called `vtable!(MyEncoding)` to generate trait impls
- [ ] Implemented all `VTable` trait methods
- [ ] Implemented `OperationsVTable` (at minimum `scalar_at`)
- [ ] Implemented validity handling via one of the `ValidityVTable` helpers
- [ ] Written encode/decode functions
- [ ] Implemented compute functions (filter, take, slice, mask) where applicable
- [ ] Defined `RULES` and `PARENT_KERNELS` for optimizer integration
- [ ] Added session registration via `initialize()` for external encodings
- [ ] Written tests using conformance suites and `rstest`
- [ ] Implemented a `Scheme` in `vortex-btrblocks` for automatic compression
- [ ] Added a backward-compatibility fixture in `vortex-test/compat-gen`
- [ ] Added compress/decompress benchmarks
- [ ] Verified with `cargo clippy --all-targets --all-features`
- [ ] Formatted with `cargo +nightly fmt --all`

## Reference Implementations

For simple encodings, study:

- `encodings/zigzag/` -- wraps a single child, no metadata

For encodings with structured metadata:

- `encodings/runend/` -- two children, prost-serialized metadata, custom optimizer rules

For built-in canonical encodings:

- `vortex-array/src/arrays/constant/` -- simplest canonical array
- `vortex-array/src/arrays/primitive/` -- foundational primitive array
