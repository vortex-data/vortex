// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Common logic for comparing ALP-encoded arrays with scalar values.
//!
//! This module contains shared logic used by both eager comparison (compare.rs)
//! and lazy comparison pushdown (expr_pushdown.rs).

use vortex_array::compute::Operator;
use vortex_dtype::NativePType;
use vortex_scalar::Scalar;

use crate::{ALPFloat, Exponents};

/// Result of encoding a scalar value for comparison with an ALP-encoded array.
#[derive(Debug, Clone, Copy)]
pub(super) enum EncodedComparison<T> {
    /// The scalar encoded cleanly - compare using the encoded value with the original operator
    Encoded { value: T, operator: Operator },
    /// The scalar doesn't encode - return a constant result for all elements
    Constant(bool),
}

/// Determine how to compare an ALP-encoded array with a scalar value.
///
/// This encapsulates the core logic for ALP scalar comparisons:
/// - If the scalar encodes cleanly in the ALP domain, compare using the encoded value
/// - If not encodable, handle special cases based on the operator:
///   - Eq/NotEq: constant result (false/true)
///   - Gt/Gte: use encode_above with Gte operator (handles IEEE 754 totalOrder)
///   - Lt/Lte: use encode_below with Lte operator (handles IEEE 754 totalOrder)
///
/// # Examples
///
/// ```ignore
/// let exponents = Exponents { e: 3, f: 0 };
/// match encode_for_comparison(1.234f32, exponents, Operator::Gt) {
///     EncodedComparison::Encoded { value, operator } => {
///         // Compare encoded array with encoded value using operator
///     }
///     EncodedComparison::Constant(result) => {
///         // Return constant result for all elements
///     }
/// }
/// ```
pub(super) fn encode_for_comparison<F: ALPFloat + Into<Scalar>>(
    value: F,
    exponents: Exponents,
    operator: Operator,
) -> EncodedComparison<F::ALPInt>
where
    F::ALPInt: Into<Scalar>,
{
    // Try to encode the scalar into the ALP domain
    let encoded = F::encode_single(value, exponents);

    match encoded {
        Some(encoded_value) => EncodedComparison::Encoded {
            value: encoded_value,
            operator,
        },
        None => {
            // Value doesn't encode cleanly - handle special cases
            match operator {
                // Since this value is not encodable it cannot be equal to any value in the encoded array
                Operator::Eq => EncodedComparison::Constant(false),
                // Since this value is not encodable it is not equal to all values in the encoded array
                Operator::NotEq => EncodedComparison::Constant(true),
                Operator::Gt | Operator::Gte => {
                    // Per IEEE 754 totalOrder semantics: -NaN < -Inf < finite < +Inf < +NaN
                    // All values in the encoded array are definitely finite
                    let is_not_finite =
                        NativePType::is_infinite(value) || NativePType::is_nan(value);

                    if is_not_finite {
                        // Comparing finite values to non-finite:
                        // - finite > -Inf is true, finite > +Inf is false
                        // - finite > -NaN is true, finite > +NaN is false
                        // Result depends on the sign of the non-finite value
                        EncodedComparison::Constant(value.is_sign_negative())
                    } else {
                        // For finite unencodable values, use encode_above
                        // Since the encoded value is unencodable, Gte is equivalent to Gt
                        // Consider a value v between two encodable values v_l (just less) and
                        // v_a (just above), then for all encodable values u: v > u <=> v_a >= u
                        EncodedComparison::Encoded {
                            value: F::encode_above(value, exponents),
                            operator: Operator::Gte,
                        }
                    }
                }
                Operator::Lt | Operator::Lte => {
                    // Per IEEE 754 totalOrder semantics: -NaN < -Inf < finite < +Inf < +NaN
                    // All values in the encoded array are definitely finite
                    let is_not_finite =
                        NativePType::is_infinite(value) || NativePType::is_nan(value);

                    if is_not_finite {
                        // Comparing finite values to non-finite:
                        // - finite < +Inf is true, finite < -Inf is false
                        // - finite < +NaN is true, finite < -NaN is false
                        // Result depends on the sign of the non-finite value (opposite of Gt/Gte)
                        EncodedComparison::Constant(value.is_sign_positive())
                    } else {
                        // For finite unencodable values, use encode_below
                        // Since the encoded value is unencodable, Lte is equivalent to Lt
                        // See Gt | Gte for further explanation
                        EncodedComparison::Encoded {
                            value: F::encode_below(value, exponents),
                            operator: Operator::Lte,
                        }
                    }
                }
            }
        }
    }
}
