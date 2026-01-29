// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fuzzer module for testing compressor roundtrip.
//!
//! This module generates arbitrary arrays, compresses them, decompresses them,
//! and verifies that the result matches the original. It also tests serialization
//! roundtrip by writing compressed arrays to a buffer and reading them back.

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ArrayVisitor;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ChunkedVTable;
use vortex_array::arrays::FilterVTable;
use vortex_array::arrays::SliceVTable;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_buffer::ByteBufferMut;
use vortex_dtype::DType;
use vortex_dtype::StructFields;
use vortex_error::VortexExpect;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_set::HashSet;

use crate::RUNTIME;
use crate::SESSION;
use crate::array::CompressorStrategy;
use crate::array::assert_array_eq;
use crate::array::compress_array;
use crate::error::Backtrace;
use crate::error::VortexFuzzError;
use crate::error::VortexFuzzResult;

/// Input for the compressor roundtrip fuzzer.
#[derive(Debug)]
pub struct FuzzCompressor {
    pub array: ArrayRef,
    pub strategy: CompressorStrategy,
}

impl<'a> Arbitrary<'a> for FuzzCompressor {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let array = ArbitraryArray::arbitrary(u)?.0;
        let strategy = CompressorStrategy::arbitrary(u)?;
        Ok(FuzzCompressor { array, strategy })
    }
}

/// Recursively checks if an array or any of its children contains a forbidden encoding.
///
/// Forbidden encodings are those that should not appear in compressor output:
/// - Slice: lazy view encoding, not serializable
/// - Filter: lazy view encoding, not serializable
/// - Chunked: in-memory batching encoding, compressor should produce flat arrays
fn contains_forbidden_encoding(array: &ArrayRef) -> Option<&'static str> {
    let encoding_id = array.encoding_id();

    if encoding_id == SliceVTable::ID {
        return Some("vortex.slice");
    }
    if encoding_id == FilterVTable::ID {
        return Some("vortex.filter");
    }
    if encoding_id == ChunkedVTable::ID {
        return Some("vortex.chunked");
    }

    // Recursively check children
    for child in array.children() {
        if let Some(found) = contains_forbidden_encoding(&child) {
            return Some(found);
        }
    }

    None
}

/// Run the compressor roundtrip fuzzer.
///
/// Returns:
/// - `Ok(true)` - keep in corpus
/// - `Ok(false)` - reject from corpus
/// - `Err(_)` - a bug was found
#[allow(clippy::result_large_err)]
pub fn run_compressor_fuzzer(fuzz: FuzzCompressor) -> VortexFuzzResult<bool> {
    let FuzzCompressor { array, strategy } = fuzz;

    // Store original properties
    let original_len = array.len();
    let original_dtype = array.dtype().clone();

    // Canonicalize first to get a clean baseline
    let canonical = array
        .to_canonical()
        .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?;
    let canonical_array = canonical.into_array();

    // Compress the canonical array
    let compressed = compress_array(&canonical_array, strategy);

    // Check that compressed array doesn't contain forbidden encodings (Slice, Filter, Chunked)
    if let Some(forbidden_encoding) = contains_forbidden_encoding(&compressed) {
        return Err(VortexFuzzError::ForbiddenEncoding(
            forbidden_encoding.to_string(),
            compressed,
            Backtrace::capture(),
        ));
    }

    // Verify dtype is preserved after compression
    if &original_dtype != compressed.dtype() {
        return Err(VortexFuzzError::DTypeMismatch(
            canonical_array,
            compressed,
            0,
            Backtrace::capture(),
        ));
    }

    // Verify len is preserved after compression
    if original_len != compressed.len() {
        return Err(VortexFuzzError::LengthMismatch(
            original_len,
            compressed.len(),
            canonical_array,
            compressed,
            0,
            Backtrace::capture(),
        ));
    }

    // Decompress by converting back to canonical form
    let decompressed = compressed
        .to_canonical()
        .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?;
    let decompressed_array = decompressed.into_array();

    // Verify dtype is preserved after decompression
    if &original_dtype != decompressed_array.dtype() {
        return Err(VortexFuzzError::DTypeMismatch(
            canonical_array,
            decompressed_array,
            1,
            Backtrace::capture(),
        ));
    }

    // Verify len is preserved after decompression
    if original_len != decompressed_array.len() {
        return Err(VortexFuzzError::LengthMismatch(
            original_len,
            decompressed_array.len(),
            canonical_array,
            decompressed_array,
            1,
            Backtrace::capture(),
        ));
    }

    // Verify decompressed array is canonical
    if !decompressed_array.is_canonical() {
        return Err(VortexFuzzError::NotCanonical(
            decompressed_array,
            Backtrace::capture(),
        ));
    }

    // Verify array contents are equal (element-by-element comparison)
    assert_array_eq(&canonical_array, &decompressed_array, 0)?;

    // Skip file roundtrip for arrays that have known issues with file format
    if has_nullable_struct(&original_dtype) || has_duplicate_field_names(&original_dtype) {
        return Ok(true);
    }

    // Test file serialization roundtrip: write compressed array to buffer, read back, decompress
    let mut buffer = ByteBufferMut::empty();
    SESSION
        .write_options()
        .blocking(&*RUNTIME)
        .write(&mut buffer, compressed.to_array_iterator())
        .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?;

    // Read array back from buffer
    let mut output = SESSION
        .open_options()
        .open_buffer(buffer)
        .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?
        .scan()
        .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?
        .into_array_iter(&*RUNTIME)
        .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?;

    let from_file = match output.len() {
        0 => Canonical::empty(&original_dtype).into_array(),
        1 => output.pop().vortex_expect("one output"),
        _ => ChunkedArray::from_iter(output).into_array(),
    };

    // Verify dtype is preserved after file roundtrip
    if &original_dtype != from_file.dtype() {
        return Err(VortexFuzzError::DTypeMismatch(
            canonical_array,
            from_file,
            2,
            Backtrace::capture(),
        ));
    }

    // Verify len is preserved after file roundtrip
    if original_len != from_file.len() {
        return Err(VortexFuzzError::LengthMismatch(
            original_len,
            from_file.len(),
            canonical_array,
            from_file,
            2,
            Backtrace::capture(),
        ));
    }

    // Decompress the array read from file
    let from_file_decompressed = from_file
        .to_canonical()
        .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?;
    let from_file_array = from_file_decompressed.into_array();

    // Verify array contents are equal after file roundtrip
    assert_array_eq(&canonical_array, &from_file_array, 2)?;

    Ok(true)
}

/// Checks if dtype contains a nullable struct (not supported by file format).
fn has_nullable_struct(dtype: &DType) -> bool {
    dtype.is_struct() && dtype.is_nullable()
        || dtype
            .as_struct_fields_opt()
            .map(|sdt| sdt.fields().any(|dtype| has_nullable_struct(&dtype)))
            .unwrap_or(false)
        || dtype
            .as_list_element_opt()
            .map(|e| has_nullable_struct(e.as_ref()))
            .unwrap_or(false)
}

/// Checks if dtype has duplicate field names in struct types.
fn has_duplicate_field_names(dtype: &DType) -> bool {
    if let Some(struct_dtype) = dtype.as_struct_fields_opt() {
        struct_has_duplicate_names(struct_dtype)
    } else if let Some(list_elem) = dtype.as_list_element_opt() {
        has_duplicate_field_names(list_elem)
    } else {
        false
    }
}

fn struct_has_duplicate_names(struct_dtype: &StructFields) -> bool {
    HashSet::<_, DefaultHashBuilder>::from_iter(struct_dtype.names().iter()).len()
        != struct_dtype.names().len()
        || struct_dtype
            .fields()
            .any(|dtype| has_duplicate_field_names(&dtype))
}
