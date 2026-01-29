// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fuzzer module for testing compressor roundtrip.
//!
//! This module generates arbitrary arrays, compresses them, decompresses them,
//! and verifies that the result matches the original.

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ArrayVisitor;
use vortex_array::IntoArray;
use vortex_array::arrays::FilterVTable;
use vortex_array::arrays::SliceVTable;
use vortex_array::arrays::arbitrary::ArbitraryArray;

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

/// Recursively checks if an array or any of its children contains a Slice or Filter encoding.
fn contains_slice_or_filter(array: &ArrayRef) -> Option<&'static str> {
    let encoding_id = array.encoding_id();

    if encoding_id == SliceVTable::ID {
        return Some("vortex.slice");
    }
    if encoding_id == FilterVTable::ID {
        return Some("vortex.filter");
    }

    // Recursively check children
    for child in array.children() {
        if let Some(found) = contains_slice_or_filter(&child) {
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

    // Check that compressed array doesn't contain Slice or Filter encodings
    if let Some(forbidden_encoding) = contains_slice_or_filter(&compressed) {
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

    // Verify array contents are equal (element-by-element comparison)
    assert_array_eq(&canonical_array, &decompressed_array, 0)?;

    Ok(true)
}
