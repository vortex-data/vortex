// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fuzzer module for testing compressed encoding canonicalization.
//!
//! This module generates arbitrary instances of compressed encodings (DictArray, etc.),
//! then verifies that `to_canonical()` works and produces correct `len` and `dtype`.

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::constant::ArbitraryConstantArray;
use vortex_array::arrays::dict::ArbitraryDictArray;
use vortex_runend::ArbitraryRunEndArray;

/// Which compressed encoding to generate.
#[derive(Debug, Clone, Copy)]
pub enum EncodingKind {
    Dict,
    Constant,
    RunEnd,
}

impl<'a> Arbitrary<'a> for EncodingKind {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        match u.int_in_range(0..=2)? {
            0 => Ok(EncodingKind::Dict),
            1 => Ok(EncodingKind::Constant),
            2 => Ok(EncodingKind::RunEnd),
            _ => unreachable!(),
        }
    }
}

/// Input for the compressed encoding canonicalization fuzzer.
#[derive(Debug)]
pub struct FuzzCompressRoundtrip {
    pub array: ArrayRef,
}

impl<'a> Arbitrary<'a> for FuzzCompressRoundtrip {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let kind: EncodingKind = u.arbitrary()?;

        let array = match kind {
            EncodingKind::Dict => ArbitraryDictArray::arbitrary(u)?.0.into_array(),
            EncodingKind::Constant => ArbitraryConstantArray::arbitrary(u)?.0.into_array(),
            EncodingKind::RunEnd => ArbitraryRunEndArray::arbitrary(u)?.0.into_array(),
        };

        Ok(FuzzCompressRoundtrip { array })
    }
}

/// Run the compressed encoding canonicalization fuzzer.
///
/// Returns:
/// - `Ok(true)` - keep in corpus
/// - `Ok(false)` - reject from corpus
/// - `Err(_)` - a bug was found
#[allow(clippy::result_large_err)]
pub fn run_compress_roundtrip(fuzz: FuzzCompressRoundtrip) -> crate::error::VortexFuzzResult<bool> {
    use crate::error::Backtrace;
    use crate::error::VortexFuzzError;

    let FuzzCompressRoundtrip { array } = fuzz;

    // Store original properties
    let original_len = array.len();
    let original_dtype = array.dtype().clone();

    // Try to canonicalize - this is the main thing we're testing
    let canonical = match array.to_canonical() {
        Ok(c) => c,
        Err(e) => {
            // Canonicalization failed - this is a bug
            return Err(VortexFuzzError::VortexError(e, Backtrace::capture()));
        }
    };

    let canonical_array: ArrayRef = canonical.into_array();

    // Verify dtype is preserved
    if &original_dtype != canonical_array.dtype() {
        return Err(VortexFuzzError::DTypeMismatch(
            array,
            canonical_array,
            0,
            Backtrace::capture(),
        ));
    }

    // Verify len is preserved
    if original_len != canonical_array.len() {
        return Err(VortexFuzzError::LengthMismatch(
            original_len,
            canonical_array.len(),
            array,
            canonical_array,
            0,
            Backtrace::capture(),
        ));
    }

    Ok(true)
}
