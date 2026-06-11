// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fuzzer module for testing compressed encoding canonicalization.
//!
//! This module generates arbitrary instances of compressed encodings (DictArray, etc.),
//! then verifies that `to_canonical()` works and produces correct `len` and `dtype`.

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::constant::ArbitraryConstantArray;
use vortex_array::arrays::dict::ArbitraryDictArray;
use vortex_runend::ArbitraryRunEndArray;

use crate::SESSION;

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
#[expect(clippy::result_large_err)]
pub fn run_compress_roundtrip(fuzz: FuzzCompressRoundtrip) -> crate::error::VortexFuzzResult<bool> {
    use crate::error::Backtrace;
    use crate::error::VortexFuzzError;

    let FuzzCompressRoundtrip { array } = fuzz;

    // Store original properties
    let original_len = array.len();
    let original_dtype = array.dtype().clone();

    let mut ctx = SESSION.create_execution_ctx();

    // Try to canonicalize - this is the main thing we're testing
    let canonical = match array.clone().execute::<Canonical>(&mut ctx) {
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

#[cfg(test)]
mod tests {
    use arbitrary::Arbitrary;
    use arbitrary::Unstructured;

    use super::FuzzCompressRoundtrip;
    use super::run_compress_roundtrip;

    /// Deterministic pseudo-random bytes for driving [`Unstructured`].
    fn pseudo_random_bytes(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 24) as u8
            })
            .collect()
    }

    /// End-to-end smoke test of the compress roundtrip pipeline, covering the arbitrary
    /// dtypes (including temporal extension dtypes) without needing a fuzzing engine.
    #[test]
    fn compress_roundtrip_pipeline_smoke() {
        let mut ran = 0;
        for seed in 0..256 {
            let bytes = pseudo_random_bytes(16 * 1024, seed);
            let mut u = Unstructured::new(&bytes);
            let Ok(fuzz) = FuzzCompressRoundtrip::arbitrary(&mut u) else {
                continue;
            };
            if let Err(e) = run_compress_roundtrip(fuzz) {
                panic!("seed {seed}: {e}");
            }
            ran += 1;
        }
        assert!(ran > 0, "no compress roundtrips were generated");
    }
}
