// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fuzzer for FSST compression roundtrip: build a `VarBinArray` of arbitrary
//! bytestrings, compress it with FSST, canonicalize the result back to
//! `VarBinViewArray`, and assert that every element (including nulls) matches
//! the original input.
//!
//! The string generator mirrors `fsst_like.rs` (small alphabet biased toward
//! repeated substrings so the FSST symbol table actually fires), with optional
//! null slots and an occasional injection of long strings so we exercise the
//! buffer-growth path in `fsst_compress_iter`.

use std::sync::LazyLock;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::session::ArraySession;
use vortex_fsst::FSSTArray;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_session::VortexSession;

use crate::error::Backtrace;
use crate::error::VortexFuzzError;
use crate::error::VortexFuzzResult;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// Maximum number of byte slots per fuzz iteration. Keeps the per-iteration
/// budget bounded so libfuzzer can stay fast.
const MAX_VALUES: usize = 512;

/// Default per-slot length cap. Most values stay short to encourage FSST to
/// build a useful symbol table.
const DEFAULT_MAX_LEN: usize = 256;

/// Occasional "huge" slot length, which forces `fsst_compress_iter` through
/// its reallocation branch.
const LARGE_MAX_LEN: usize = 16 * 1024;

/// Fuzz input: a vector of optional bytestrings plus a nullability flag.
#[derive(Debug)]
pub struct FuzzFsstRoundtrip {
    /// `None` slots are kept when `nullable` is true. When `nullable` is false
    /// they are coerced to empty strings so we can still build a
    /// `NonNullable` array from the same generator.
    pub values: Vec<Option<Vec<u8>>>,
    pub nullable: bool,
    pub utf8: bool,
}

impl<'a> Arbitrary<'a> for FuzzFsstRoundtrip {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let nullable: bool = u.arbitrary()?;
        // Bias toward Utf8 since that is the most common FSST target, but
        // exercise Binary as well so we cover non-utf8 bytes.
        let utf8: bool = u.ratio(3, 4)?;

        let n: usize = u.int_in_range(1..=MAX_VALUES)?;
        let mut values = Vec::with_capacity(n);

        // A small alphabet drives repeated substrings into the symbol table.
        let alpha_lo: u8 = b'a';
        let alpha_hi: u8 = b'h';

        for _ in 0..n {
            // ~10% nulls when nullable.
            if nullable && u.ratio(1, 10)? {
                values.push(None);
                continue;
            }

            // ~2% of slots use the long-string path.
            let max_len = if u.ratio(1, 50)? {
                LARGE_MAX_LEN
            } else {
                DEFAULT_MAX_LEN
            };
            let len: usize = u.int_in_range(0..=max_len)?;

            let mut bytes = Vec::with_capacity(len);
            for _ in 0..len {
                if utf8 {
                    bytes.push(u.int_in_range(alpha_lo..=alpha_hi)?);
                } else {
                    // For binary we allow the full byte range, but bias toward
                    // small alphabets so FSST training still has signal.
                    if u.ratio(3, 4)? {
                        bytes.push(u.int_in_range(alpha_lo..=alpha_hi)?);
                    } else {
                        bytes.push(u.arbitrary::<u8>()?);
                    }
                }
            }
            values.push(Some(bytes));
        }

        Ok(FuzzFsstRoundtrip {
            values,
            nullable,
            utf8,
        })
    }
}

/// Run the FSST roundtrip fuzzer.
///
/// Returns:
/// - `Ok(true)` — keep in corpus
/// - `Ok(false)` — reject (degenerate input)
/// - `Err(_)` — mismatch or unexpected error
#[expect(clippy::result_large_err)]
pub fn run_fsst_roundtrip_fuzz(fuzz: FuzzFsstRoundtrip) -> VortexFuzzResult<bool> {
    let FuzzFsstRoundtrip {
        values,
        nullable,
        utf8,
    } = fuzz;

    if values.is_empty() {
        return Ok(false);
    }

    let nullability = if nullable {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    let dtype = if utf8 {
        DType::Utf8(nullability)
    } else {
        DType::Binary(nullability)
    };

    // Materialize the input. When the array is non-nullable, replace `None`
    // slots with empty strings so the generated `Arbitrary` distribution is
    // still useful for the non-nullable case.
    let materialized: Vec<Option<Vec<u8>>> = if nullable {
        values
    } else {
        values
            .into_iter()
            .map(|v| Some(v.unwrap_or_default()))
            .collect()
    };

    let varbin = VarBinArray::from_iter(
        materialized
            .iter()
            .map(|opt| opt.as_ref().map(|v| v.as_slice())),
        dtype.clone(),
    );

    let mut ctx = SESSION.create_execution_ctx();

    // Train a compressor on the input and compress.
    let compressor = fsst_train_compressor(&varbin);
    let fsst: FSSTArray = fsst_compress(varbin.clone(), varbin.len(), &dtype, &compressor, &mut ctx);

    // Sanity: length and dtype must round-trip through the compressed form.
    if fsst.len() != materialized.len() {
        return Err(VortexFuzzError::VortexError(
            vortex_error::vortex_err!(
                "FSST length mismatch after compress: expected {}, got {}",
                materialized.len(),
                fsst.len(),
            ),
            Backtrace::capture(),
        ));
    }
    if fsst.dtype() != &dtype {
        return Err(VortexFuzzError::VortexError(
            vortex_error::vortex_err!(
                "FSST dtype mismatch after compress: expected {:?}, got {:?}",
                dtype,
                fsst.dtype(),
            ),
            Backtrace::capture(),
        ));
    }

    // Canonicalize the FSST array back to a VarBinView and compare every slot
    // to the original input.
    let canonical = fsst
        .into_array()
        .execute::<Canonical>(&mut ctx)
        .map_err(|err| VortexFuzzError::VortexError(err, Backtrace::capture()))?;
    let view: VarBinViewArray = canonical.into_varbinview();

    let actual: Vec<Option<Vec<u8>>> =
        view.with_iterator(|iter| iter.map(|opt| opt.map(|b| b.to_vec())).collect());

    if actual.len() != materialized.len() {
        return Err(VortexFuzzError::VortexError(
            vortex_error::vortex_err!(
                "FSST canonicalize length mismatch: expected {}, got {}",
                materialized.len(),
                actual.len(),
            ),
            Backtrace::capture(),
        ));
    }

    for (idx, (expected, got)) in materialized.iter().zip(actual.iter()).enumerate() {
        if expected != got {
            return Err(VortexFuzzError::VortexError(
                vortex_error::vortex_err!(
                    "FSST roundtrip mismatch at index {idx}:\n  expected: {expected:?}\n  got:      {got:?}",
                ),
                Backtrace::capture(),
            ));
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_ok(values: Vec<Option<Vec<u8>>>, nullable: bool, utf8: bool) {
        let res = run_fsst_roundtrip_fuzz(FuzzFsstRoundtrip {
            values,
            nullable,
            utf8,
        });
        match res {
            Ok(_) => {}
            Err(e) => panic!("expected roundtrip ok, got error: {e}"),
        }
    }

    #[test]
    fn smoke_nullable_utf8() {
        let values = (0..32)
            .map(|i| {
                if i % 7 == 0 {
                    None
                } else {
                    Some(format!("abcabc{i}").into_bytes())
                }
            })
            .collect();
        run_ok(values, true, true);
    }

    #[test]
    fn smoke_nonnullable_utf8() {
        let values = (0..16)
            .map(|i| Some(format!("hello-world-{i}").into_bytes()))
            .collect();
        run_ok(values, false, true);
    }

    #[test]
    fn smoke_nullable_binary() {
        let values = vec![
            Some(vec![0u8, 1, 2, 3, 4]),
            None,
            Some(vec![255u8; 32]),
            Some(b"abcabcabc".to_vec()),
        ];
        run_ok(values, true, false);
    }

    #[test]
    fn smoke_with_large_string() {
        // Exercise the buffer-growth path inside fsst_compress_iter.
        let big: Vec<u8> = "abcabcabc".repeat(200_000).into_bytes();
        let values = vec![
            Some(b"a".to_vec()),
            Some(b"ab".to_vec()),
            Some(big),
            Some(b"abc".to_vec()),
        ];
        run_ok(values, false, true);
    }

    #[test]
    fn smoke_all_empty() {
        let values = (0..16).map(|_| Some(vec![])).collect();
        run_ok(values, false, true);
    }
}
