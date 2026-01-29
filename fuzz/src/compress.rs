// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fuzzer module for testing compressed encoding canonicalization.
//!
//! This module generates arbitrary instances of compressed encodings (DictArray, etc.),
//! then verifies that `to_canonical()` works and produces correct `len` and `dtype`.
//!
//! It also tests that applying arbitrary expressions to compressed arrays produces
//! the same results as applying them to canonical arrays.

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ArbitraryConstantArray;
use vortex_array::arrays::ArbitraryDictArray;
use vortex_array::expr::Expression;
use vortex_array::expr::arbitrary::arb_filter_expr;
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

/// Input for the compressed encoding expression roundtrip fuzzer.
///
/// This tests that applying an arbitrary expression to a compressed array
/// produces the same result as applying it to the canonical form.
pub struct FuzzCompressExprRoundtrip {
    pub array: ArrayRef,
    pub expr: Expression,
}

impl std::fmt::Debug for FuzzCompressExprRoundtrip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuzzCompressExprRoundtrip")
            .field("array", &self.array)
            .field("expr", &self.expr.to_string())
            .finish()
    }
}

impl<'a> Arbitrary<'a> for FuzzCompressExprRoundtrip {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let kind: EncodingKind = u.arbitrary()?;

        let array = match kind {
            EncodingKind::Dict => ArbitraryDictArray::arbitrary(u)?.0.into_array(),
            EncodingKind::Constant => ArbitraryConstantArray::arbitrary(u)?.0.into_array(),
            EncodingKind::RunEnd => ArbitraryRunEndArray::arbitrary(u)?.0.into_array(),
        };

        // Generate an expression that returns Bool (filter expression)
        // Use depth 3 for reasonable complexity
        let expr = match arb_filter_expr(u, array.dtype(), 3)? {
            Some(e) => e,
            None => {
                // If we couldn't generate a filter expr, just use is_null(root())
                use vortex_array::expr::is_null;
                use vortex_array::expr::root;
                is_null(root())
            }
        };

        Ok(FuzzCompressExprRoundtrip { array, expr })
    }
}

/// Run the compressed encoding expression roundtrip fuzzer.
///
/// This tests that applying an expression to a compressed array produces
/// the same results as applying it to the canonical form of that array.
///
/// Returns:
/// - `Ok(true)` - keep in corpus
/// - `Ok(false)` - reject from corpus (e.g., expression execution failed for expected reasons)
/// - `Err(_)` - a bug was found (results differ between compressed and canonical)
#[allow(clippy::result_large_err)]
pub fn run_compress_expr_roundtrip(
    fuzz: FuzzCompressExprRoundtrip,
) -> crate::error::VortexFuzzResult<bool> {
    use crate::error::Backtrace;
    use crate::error::VortexFuzzError;

    let FuzzCompressExprRoundtrip { array, expr } = fuzz;

    // Skip empty arrays
    if array.is_empty() {
        return Ok(false);
    }

    // Canonicalize the array
    let canonical = match array.to_canonical() {
        Ok(c) => c,
        Err(e) => {
            return Err(VortexFuzzError::VortexError(e, Backtrace::capture()));
        }
    };
    let canonical_array: ArrayRef = canonical.into_array();

    // Create execution context
    let session = crate::SESSION.clone();
    let mut ctx = ExecutionCtx::new(session);

    // Apply expression to compressed array
    let compressed_applied = match array.apply(&expr) {
        Ok(a) => a,
        Err(_e) => {
            // Expression application failed - might be expected for some dtype/expr combos
            // Return false to not keep in corpus
            return Ok(false);
        }
    };

    let compressed_result = match compressed_applied.execute::<Canonical>(&mut ctx) {
        Ok(r) => r.into_array(),
        Err(_e) => {
            // Execution failed - might be expected
            return Ok(false);
        }
    };

    // Apply expression to canonical array
    let canonical_applied = match canonical_array.apply(&expr) {
        Ok(a) => a,
        Err(e) => {
            // If it worked on compressed but not canonical, that's a bug
            return Err(VortexFuzzError::VortexError(e, Backtrace::capture()));
        }
    };

    let canonical_result = match canonical_applied.execute::<Canonical>(&mut ctx) {
        Ok(r) => r.into_array(),
        Err(e) => {
            // If it worked on compressed but not canonical, that's a bug
            return Err(VortexFuzzError::VortexError(e, Backtrace::capture()));
        }
    };

    // Compare results
    if compressed_result.len() != canonical_result.len() {
        return Err(VortexFuzzError::LengthMismatch(
            compressed_result.len(),
            canonical_result.len(),
            compressed_result,
            canonical_result,
            0,
            Backtrace::capture(),
        ));
    }

    // Compare element by element
    for i in 0..compressed_result.len() {
        let compressed_scalar = match compressed_result.scalar_at(i) {
            Ok(s) => s,
            Err(e) => return Err(VortexFuzzError::VortexError(e, Backtrace::capture())),
        };
        let canonical_scalar = match canonical_result.scalar_at(i) {
            Ok(s) => s,
            Err(e) => return Err(VortexFuzzError::VortexError(e, Backtrace::capture())),
        };

        if compressed_scalar != canonical_scalar {
            return Err(VortexFuzzError::ArrayNotEqual(
                compressed_scalar,
                canonical_scalar,
                i,
                compressed_result,
                canonical_result,
                0,
                Backtrace::capture(),
            ));
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use arbitrary::Unstructured;
    use vortex_array::Array;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ArbitraryDictArray;
    use vortex_array::expr::is_null;
    use vortex_array::expr::root;

    use super::*;

    #[test]
    fn test_compress_expr_roundtrip_runs() {
        // Test with several different random seeds
        for seed in 0u8..10 {
            let data = vec![seed; 1024];
            let mut u = Unstructured::new(&data);

            if let Ok(fuzz) = FuzzCompressExprRoundtrip::arbitrary(&mut u) {
                // The fuzzer should not panic
                let result = run_compress_expr_roundtrip(fuzz);
                // Result can be Ok(true), Ok(false), or Err - all are valid
                // We just want to make sure it doesn't panic
                match result {
                    Ok(true) => {}  // kept in corpus
                    Ok(false) => {} // rejected from corpus (empty array or failed expr)
                    Err(e) => {
                        // This would indicate a potential bug - print it for debugging
                        panic!("Fuzzer found potential bug: {:?}", e);
                    }
                }
            }
        }
    }

    #[test]
    fn test_compress_roundtrip_runs() {
        // Test the original roundtrip fuzzer too
        for seed in 0u8..10 {
            let data = vec![seed; 512];
            let mut u = Unstructured::new(&data);

            if let Ok(fuzz) = FuzzCompressRoundtrip::arbitrary(&mut u) {
                let result = run_compress_roundtrip(fuzz);
                match result {
                    Ok(_) => {}
                    Err(e) => {
                        panic!("Compress roundtrip found potential bug: {:?}", e);
                    }
                }
            }
        }
    }

    #[test]
    fn test_explicit_evaluation() {
        // Create a DictArray with known values
        let data = vec![100u8; 512];
        let mut u = Unstructured::new(&data);

        let dict_array = ArbitraryDictArray::arbitrary(&mut u)
            .expect("should create dict array")
            .0
            .into_array();

        println!("\n=== Explicit Evaluation Test ===");
        println!("Array encoding: {}", dict_array.encoding_id());
        println!("Array dtype: {}", dict_array.dtype());
        println!("Array len: {}", dict_array.len());

        // Canonicalize
        let canonical = dict_array.to_canonical().expect("should canonicalize");
        let canonical_array = canonical.into_array();

        // Create a simple expression: is_null(root())
        let expr = is_null(root());
        println!("Expression: {expr}");

        // Apply to both
        let compressed_applied = dict_array.apply(&expr).expect("apply to compressed");
        let canonical_applied = canonical_array.apply(&expr).expect("apply to canonical");

        // Execute both
        let session = crate::SESSION.clone();
        let mut ctx = ExecutionCtx::new(session);

        let compressed_result = compressed_applied
            .execute::<Canonical>(&mut ctx)
            .expect("execute compressed")
            .into_array();
        let canonical_result = canonical_applied
            .execute::<Canonical>(&mut ctx)
            .expect("execute canonical")
            .into_array();

        println!("Compressed result len: {}", compressed_result.len());
        println!("Canonical result len: {}", canonical_result.len());

        // Compare first few elements
        let limit = compressed_result.len().min(10);
        println!("\nFirst {limit} results:");
        for i in 0..limit {
            let c = compressed_result.scalar_at(i).unwrap();
            let n = canonical_result.scalar_at(i).unwrap();
            let match_str = if c == n { "✓" } else { "✗ MISMATCH" };
            println!("  [{i}] compressed={c}, canonical={n} {match_str}");
        }

        // Verify they match
        assert_eq!(
            compressed_result.len(),
            canonical_result.len(),
            "lengths should match"
        );
        for i in 0..compressed_result.len() {
            let c = compressed_result.scalar_at(i).unwrap();
            let n = canonical_result.scalar_at(i).unwrap();
            assert_eq!(c, n, "mismatch at index {i}");
        }

        println!("\n✓ All {} elements match!", compressed_result.len());
    }

    #[test]
    fn test_generated_expr_evaluation() {
        println!("\n=== Generated Expression Evaluation Test ===\n");

        let mut successes = 0;
        for seed in 0u8..20 {
            let data = vec![seed; 1024];
            let mut u = Unstructured::new(&data);

            // Generate a compressed array
            let Ok(fuzz) = FuzzCompressExprRoundtrip::arbitrary(&mut u) else {
                continue;
            };

            let array = &fuzz.array;
            let expr = &fuzz.expr;

            // Skip empty arrays
            if array.is_empty() {
                continue;
            }

            // Canonicalize
            let Ok(canonical) = array.to_canonical() else {
                continue;
            };
            let canonical_array = canonical.into_array();

            // Apply expression
            let Ok(compressed_applied) = array.apply(expr) else {
                continue;
            };
            let Ok(canonical_applied) = canonical_array.apply(expr) else {
                continue;
            };

            // Execute
            let session = crate::SESSION.clone();
            let mut ctx = ExecutionCtx::new(session);

            let Ok(compressed_result) = compressed_applied.execute::<Canonical>(&mut ctx) else {
                continue;
            };
            let compressed_result = compressed_result.into_array();

            let session2 = crate::SESSION.clone();
            let mut ctx2 = ExecutionCtx::new(session2);
            let Ok(canonical_result) = canonical_applied.execute::<Canonical>(&mut ctx2) else {
                continue;
            };
            let canonical_result = canonical_result.into_array();

            // Compare
            if compressed_result.len() != canonical_result.len() {
                panic!("Length mismatch for expr: {expr}");
            }

            let mut all_match = true;
            for i in 0..compressed_result.len() {
                let c = compressed_result.scalar_at(i).unwrap();
                let n = canonical_result.scalar_at(i).unwrap();
                if c != n {
                    all_match = false;
                    break;
                }
            }

            if all_match {
                println!(
                    "Seed {:2}: {} ({}, len={}) -> ✓ {} elements match",
                    seed,
                    array.encoding_id(),
                    array.dtype(),
                    array.len(),
                    compressed_result.len()
                );
                println!("         Expr: {expr}");
                successes += 1;
            } else {
                panic!("Mismatch for expr: {expr}");
            }
        }

        println!("\n✓ {successes} expressions evaluated correctly!");
        assert!(
            successes > 0,
            "Should have at least one successful evaluation"
        );
    }
}
