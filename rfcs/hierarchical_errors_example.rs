//! DO NOT MERGE! ONLY FOR DEMO PURPOSES!

// Example: Hierarchical Error Types for Vortex using Snafu
//
// This file demonstrates how Vortex's error handling could be migrated from a centralized
// VortexError to a hierarchical, context-rich system using snafu. Each crate defines its
// own error types that can be composed and propagated with full context.

use snafu::{Backtrace, Location, ResultExt, Snafu};
use std::sync::Arc;

// ================================================================================================
// vortex-buffer error types
// ================================================================================================

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum BufferError {
    #[snafu(display("Buffer alignment error: required {required} bytes, found {actual} bytes"))]
    AlignmentError {
        required: usize,
        actual: usize,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Invalid UTF-8 in string buffer at offset {offset}"))]
    InvalidUtf8 {
        offset: usize,
        #[snafu(source)]
        source: std::str::Utf8Error,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Buffer slice {start}..{end} out of bounds for buffer of length {len}"))]
    SliceOutOfBounds {
        start: usize,
        end: usize,
        len: usize,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Failed to allocate buffer of size {size} bytes"))]
    AllocationFailed {
        size: usize,
        location: Location,
        backtrace: Backtrace,
    },
}

// ================================================================================================
// vortex-dtype error types
// ================================================================================================

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum DTypeError {
    #[snafu(display("Cannot cast from {from} to {to}: incompatible types"))]
    IncompatibleCast {
        from: String, // Would be DType in real code
        to: String,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Failed to parse type from flatbuffer"))]
    FlatbufferParse {
        #[snafu(source)]
        source: Box<dyn std::error::Error + Send + Sync>,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Struct field '{field}' not found"))]
    FieldNotFound {
        field: String,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Type mismatch: expected {expected}, found {actual}"))]
    TypeMismatch {
        expected: String,
        actual: String,
        location: Location,
        backtrace: Backtrace,
    },
}

// ================================================================================================
// vortex-scalar error types
// ================================================================================================

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum ScalarError {
    #[snafu(display("Cannot convert null scalar to {target_type}"))]
    NullConversion {
        target_type: String,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Expected {expected} scalar, found {actual}"))]
    UnexpectedType {
        expected: String,
        actual: String,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Scalar value {value} out of range for type {target_type}"))]
    OutOfRange {
        value: String,
        target_type: String,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Failed to serialize scalar to protobuf"))]
    SerializationFailed {
        #[snafu(source)]
        source: Box<dyn std::error::Error + Send + Sync>,
        location: Location,
        backtrace: Backtrace,
    },
}

// ================================================================================================
// vortex-io error types
// ================================================================================================

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum IOError {
    #[snafu(display("Unexpected end of file while reading {expected} bytes"))]
    UnexpectedEof {
        expected: usize,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Read request out of bounds: offset {offset}, length {length}"))]
    ReadOutOfBounds {
        offset: u64,
        length: usize,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Async runtime error: task was cancelled"))]
    TaskCancelled {
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Failed to spawn async task"))]
    SpawnFailed {
        #[snafu(source)]
        source: Box<dyn std::error::Error + Send + Sync>,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Object store operation failed"))]
    ObjectStore {
        #[snafu(source)]
        source: Box<dyn std::error::Error + Send + Sync>,
        location: Location,
        backtrace: Backtrace,
    },
}

// ================================================================================================
// vortex-file error types
// ================================================================================================

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum FileError {
    #[snafu(display("Invalid magic bytes in file header: expected VORTEX, found {found:?}"))]
    InvalidMagic {
        found: Vec<u8>,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Unsupported file version {version}, expected {expected}"))]
    UnsupportedVersion {
        version: u16,
        expected: u16,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Segment length {length} exceeds maximum u32"))]
    SegmentTooLarge {
        length: u64,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("File I/O operation failed"))]
    IoOperation {
        #[snafu(source)]
        source: IOError,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Failed to parse postscript"))]
    PostscriptParse {
        location: Location,
        backtrace: Backtrace,
    },
}

// ================================================================================================
// vortex-array error types - General array operations
// ================================================================================================

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum ArrayError {
    #[snafu(display("Index {index} out of bounds for array of length {len}"))]
    IndexOutOfBounds {
        index: usize,
        len: usize,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Array operation requires non-null values but found {null_count} nulls"))]
    UnexpectedNulls {
        null_count: usize,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Encoding '{encoding}' not found in context"))]
    EncodingNotFound {
        encoding: String,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Statistics bounds do not overlap"))]
    StatsNotOverlapping {
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Buffer operation failed"))]
    BufferOperation {
        #[snafu(source)]
        source: BufferError,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Type operation failed"))]
    TypeOperation {
        #[snafu(source)]
        source: DTypeError,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Scalar operation failed"))]
    ScalarOperation {
        #[snafu(source)]
        source: ScalarError,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Compute operation failed"))]
    ComputeFailed {
        #[snafu(source)]
        source: ComputeError,
        location: Location,
        backtrace: Backtrace,
    },
}

// ================================================================================================
// vortex-array COMPUTE error types - This is the key part showing compute complexity
// ================================================================================================

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum ComputeError {
    // Cast-specific errors
    #[snafu(display("Cannot cast from {from} to {to}: no kernel available"))]
    CastUnsupported {
        from: String,
        to: String,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Cast operation failed due to type incompatibility"))]
    CastTypeMismatch {
        #[snafu(source)]
        source: DTypeError,
        location: Location,
        backtrace: Backtrace,
    },

    // Filter-specific errors
    #[snafu(display("Filter mask length {mask_len} does not match array length {array_len}"))]
    FilterLengthMismatch {
        mask_len: usize,
        array_len: usize,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Filter operation requires boolean mask, found {actual}"))]
    FilterInvalidMask {
        actual: String,
        location: Location,
        backtrace: Backtrace,
    },

    // Take-specific errors
    #[snafu(display("Take indices contain out-of-bounds value {index} for array of length {len}"))]
    TakeIndexOutOfBounds {
        index: usize,
        len: usize,
        location: Location,
        backtrace: Backtrace,
    },

    // Compare-specific errors
    #[snafu(display("Cannot compare arrays of different lengths: {left_len} vs {right_len}"))]
    CompareLengthMismatch {
        left_len: usize,
        right_len: usize,
        location: Location,
        backtrace: Backtrace,
    },

    // General compute errors that can occur in ANY compute function
    #[snafu(display(
        "No compute kernel found for operation '{operation}' on encoding '{encoding}'"
    ))]
    KernelNotFound {
        operation: String,
        encoding: String,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Invalid argument to compute function: {message}"))]
    InvalidArgument {
        message: String,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Compute operation requires canonicalization but failed"))]
    CanonicalizationFailed {
        #[snafu(source)]
        source: Box<ComputeError>,
        location: Location,
        backtrace: Backtrace,
    },

    // Compute operations can fail due to buffer operations
    #[snafu(display("Buffer allocation failed during compute"))]
    BufferAllocation {
        #[snafu(source)]
        source: BufferError,
        location: Location,
        backtrace: Backtrace,
    },

    // Compute operations can fail due to I/O
    #[snafu(display("I/O error during compute operation"))]
    IoError {
        #[snafu(source)]
        source: IOError,
        location: Location,
        backtrace: Backtrace,
    },

    // Compute operations can fail due to scalar conversions
    #[snafu(display("Scalar conversion failed during compute"))]
    ScalarConversion {
        #[snafu(source)]
        source: ScalarError,
        location: Location,
        backtrace: Backtrace,
    },

    // Compute operations might need to read from files
    #[snafu(display("File operation failed during compute"))]
    FileOperation {
        #[snafu(source)]
        source: FileError,
        location: Location,
        backtrace: Backtrace,
    },

    // Recursive compute operations (e.g., canonicalization calling other compute)
    #[snafu(display("Nested compute operation failed: {operation}"))]
    NestedCompute {
        operation: String,
        #[snafu(source)]
        source: Box<ComputeError>,
        location: Location,
        backtrace: Backtrace,
    },
}

// ================================================================================================
// Example usage showing how compute functions handle complex error scenarios
// ================================================================================================

// Mock types for demonstration
struct Array;
struct DType;
struct Mask;

// Example: Cast operation that can fail in multiple ways
fn cast_array(array: &Array, target_dtype: &DType) -> Result<Array, ComputeError> {
    // Cast can fail due to buffer allocation
    let buffer = allocate_buffer(1024).context(BufferAllocationSnafu)?;

    // Cast can fail due to type incompatibility
    check_type_compatibility(&array, &target_dtype).context(CastTypeMismatchSnafu)?;

    // Cast might need to read data from disk
    let data = read_array_data(&array).context(FileOperationSnafu)?;

    // Cast might fail to find appropriate kernel
    let kernel = find_cast_kernel(&array, &target_dtype).ok_or_else(|| {
        CastUnsupportedSnafu {
            from: "current_type".to_string(),
            to: "target_type".to_string(),
        }
        .build()
})?;

    // Cast might need to recursively call other compute operations
    let canonical = canonicalize(&array).context(NestedComputeSnafu {
        operation: "canonicalize",
    })?;

    Ok(Array)
}

// Example: Filter operation with different failure modes
fn filter_array(array: &Array, mask: &Mask) -> Result<Array, ComputeError> {
    // Validate mask length
    ensure_lengths_match(&array, &mask).context(FilterLengthMismatchSnafu {
        mask_len: 100,
        array_len: 200,
    })?;

    // Filter might need buffer operations
    let indices = compute_filtered_indices(&mask).context(BufferAllocationSnafu)?;

    // Filter might need I/O for large arrays
    let filtered_data = stream_filter_operation(&array, &indices).context(IoErrorSnafu)?;

    Ok(Array)
}

// Example: Complex compute pipeline showing error propagation
fn complex_compute_pipeline(array: &Array) -> Result<Array, ComputeError> {
    // Step 1: Cast to intermediate type (can fail multiple ways)
    let casted = cast_array(&array, &DType).context(NestedComputeSnafu { operation: "cast" })?;

    // Step 2: Filter based on condition (can fail multiple ways)
    let mask = compute_mask(&casted).context(NestedComputeSnafu {
        operation: "compute_mask",
    })?;

    let filtered = filter_array(&casted, &mask).context(NestedComputeSnafu {
        operation: "filter",
    })?;

    // Step 3: Take specific indices (can fail with bounds errors)
    let indices = vec![0, 5, 10];
    let taken =
        take_array(&filtered, &indices).context(NestedComputeSnafu { operation: "take" })?;

    Ok(taken)
}

// ================================================================================================
// Backward compatibility: Converting to legacy VortexError
// ================================================================================================

#[derive(Debug)]
pub enum VortexError {
    OutOfBounds(String, std::backtrace::Backtrace),
    ComputeError(String, std::backtrace::Backtrace),
    InvalidArgument(String, std::backtrace::Backtrace),
    Generic(String, std::backtrace::Backtrace),
}

// Each crate's error can convert to VortexError for backward compatibility
impl From<BufferError> for VortexError {
    fn from(err: BufferError) -> Self {
        match err {
            BufferError::SliceOutOfBounds {
                start, end, len, ..
            } => VortexError::OutOfBounds(
                format!("Buffer slice {}..{} exceeds length {}", start, end, len),
                std::backtrace::Backtrace::capture(),
            ),
            _ => VortexError::Generic(err.to_string(), std::backtrace::Backtrace::capture()),
        }
    }
}

impl From<ComputeError> for VortexError {
    fn from(err: ComputeError) -> Self {
        match err {
            ComputeError::TakeIndexOutOfBounds { index, len, .. } => VortexError::OutOfBounds(
                format!("Take index {} out of bounds for length {}", index, len),
                std::backtrace::Backtrace::capture(),
            ),
            ComputeError::InvalidArgument { message, .. } => {
                VortexError::InvalidArgument(message, std::backtrace::Backtrace::capture())
            }
            _ => VortexError::ComputeError(err.to_string(), std::backtrace::Backtrace::capture()),
        }
    }
}

impl From<ArrayError> for VortexError {
    fn from(err: ArrayError) -> Self {
        match err {
            ArrayError::IndexOutOfBounds { index, len, .. } => VortexError::OutOfBounds(
                format!("Index {} out of bounds for length {}", index, len),
                std::backtrace::Backtrace::capture(),
            ),
            ArrayError::ComputeFailed { source, .. } => source.into(),
            _ => VortexError::Generic(err.to_string(), std::backtrace::Backtrace::capture()),
        }
    }
}

// ================================================================================================
// Helper functions (stubs for demonstration)
// ================================================================================================

fn allocate_buffer(size: usize) -> Result<Vec<u8>, BufferError> {
    Ok(vec![0; size])
}

fn check_type_compatibility(array: &Array, dtype: &DType) -> Result<(), DTypeError> {
    Ok(())
}

fn read_array_data(array: &Array) -> Result<Vec<u8>, FileError> {
    Ok(vec![])
}

fn find_cast_kernel(array: &Array, dtype: &DType) -> Option<()> {
    Some(())
}

fn canonicalize(array: &Array) -> Result<Array, ComputeError> {
    Ok(Array)
}

fn ensure_lengths_match(array: &Array, mask: &Mask) -> Result<(), ()> {
    Ok(())
}

fn compute_filtered_indices(mask: &Mask) -> Result<Vec<usize>, BufferError> {
    Ok(vec![])
}

fn stream_filter_operation(array: &Array, indices: &[usize]) -> Result<Vec<u8>, IOError> {
    Ok(vec![])
}

fn compute_mask(array: &Array) -> Result<Mask, ComputeError> {
    Ok(Mask)
}

fn take_array(array: &Array, indices: &[usize]) -> Result<Array, ComputeError> {
    Ok(Array)
}

fn main() {
    // This example demonstrates:
    // 1. Each crate has its own error types with specific variants
    // 2. Compute functions can fail in MANY different ways and capture all context
    // 3. Errors propagate hierarchically with full context preservation
    // 4. Backward compatibility is maintained through From implementations
    // 5. The snafu library makes this ergonomic with context selectors

    println!("Hierarchical error handling example for Vortex");
    println!("================================================");
    println!();
    println!("Key benefits:");
    println!("- Compute functions can express ANY type of error that might occur");
    println!("- Each error includes location and backtrace for debugging");
    println!("- Error context is preserved through the entire call stack");
    println!("- Type safety ensures only relevant errors are handled at each level");
    println!("- Migration can be incremental with backward compatibility");
}

